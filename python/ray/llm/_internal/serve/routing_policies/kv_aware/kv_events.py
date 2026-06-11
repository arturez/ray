import importlib.util
import logging
from typing import Any, Dict, List, Optional

from ray import serve
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve.exceptions import RayServeException

logger = logging.getLogger(SERVE_LOGGER_NAME)

# experimental_configs keys overriding the per-node base ports.
KV_EVENTS_PORT_BASE_KEY = "KV_EVENTS_PORT_BASE"
KV_EVENT_CONSOLIDATOR_PORT_BASE_KEY = "KV_EVENT_CONSOLIDATOR_PORT_BASE"
DEFAULT_KV_EVENTS_PORT_BASE = 5557
DEFAULT_KV_EVENT_CONSOLIDATOR_PORT_BASE = 57001

# Dynamo's KVBM vLLM connector (consumed through vLLM's KVTransferConfig).
DYNAMO_KV_CONNECTOR = "DynamoConnector"
DYNAMO_KV_CONNECTOR_MODULE_PATH = "kvbm.vllm_integration.connector"
CONSOLIDATOR_ENDPOINTS_KEY = "consolidator_endpoints"


def configure_kv_events_for_kv_routing(llm_config: LLMConfig) -> None:
    """Enable engine KV-cache events for a KV-aware-routed deployment.

    Sets up ``engine_kwargs`` so the engine publishes KV-cache events over
    ZMQ, and selects Dynamo's KVBM connector when it is installed. Called at
    deployment build time; the endpoint is finalized per replica by
    :func:`assign_replica_kv_events_endpoint`.
    """
    engine_kwargs = llm_config.engine_kwargs
    if engine_kwargs.get("enable_prefix_caching") is False:
        logger.warning(
            "KV-aware routing is configured but enable_prefix_caching is False; "
            "the engine will not emit KV-cache events."
        )

    kv_events_config = engine_kwargs.get("kv_events_config")
    if kv_events_config is None:
        llm_config.update_engine_kwargs(
            kv_events_config={
                "enable_kv_cache_events": True,
                "publisher": "zmq",
                "endpoint": _default_kv_events_endpoint(llm_config),
            }
        )
    elif isinstance(kv_events_config, dict) and not kv_events_config.get(
        "enable_kv_cache_events"
    ):
        logger.warning(
            "KV-aware routing is configured but the user-provided kv_events_config "
            "disables KV-cache events; the KV router will not see cached blocks."
        )

    _pin_block_hash_seed(llm_config)
    _maybe_enable_dynamo_connector(llm_config)


def _pin_block_hash_seed(llm_config: LLMConfig) -> None:
    """Make engine block hashes content-deterministic across replicas.

    The KV router's global indexer chains and dedups blocks by the engines'
    block hashes, so identical content must hash identically on every
    replica. vLLM salts its block-hash chain root per process unless
    ``PYTHONHASHSEED`` is set, so pin it deployment-wide (a user-provided
    value is respected).
    """
    runtime_env = dict(llm_config.runtime_env or {})
    env_vars = dict(runtime_env.get("env_vars") or {})
    env_vars.setdefault("PYTHONHASHSEED", "0")
    runtime_env["env_vars"] = env_vars
    llm_config.runtime_env = runtime_env


def _maybe_enable_dynamo_connector(llm_config: LLMConfig) -> None:
    """Select Dynamo's KVBM connector via vLLM's KVTransferConfig.

    Only injected when the ``kvbm`` package is installed (vLLM imports the
    connector module eagerly while building its engine config) and the user
    has not configured a connector themselves.
    """
    if llm_config.engine_kwargs.get("kv_transfer_config") is not None:
        return
    if importlib.util.find_spec("kvbm") is None:
        logger.info(
            "Dynamo KVBM connector is not installed; KV-aware routing will "
            "consume the engine's KV events directly."
        )
        return
    llm_config.update_engine_kwargs(
        kv_transfer_config={
            "kv_connector": DYNAMO_KV_CONNECTOR,
            "kv_connector_module_path": DYNAMO_KV_CONNECTOR_MODULE_PATH,
            "kv_role": "kv_both",
        }
    )


def assign_replica_kv_events_endpoint(llm_config: LLMConfig) -> None:
    """Pin the engine's KV-events endpoint to a per-replica port.

    Replicas of a deployment share one ``engine_kwargs``, so colocated
    replicas would otherwise bind the same ZMQ port. Offsets the configured
    base port by the replica's rank, mirroring vLLM's own per-data-parallel
    rank offsetting (when ``data_parallel_rank`` is set, vLLM applies that
    offset itself and the base endpoint is left untouched).

    Must run in the replica process before the engine config is built, and
    only once (the offset is relative to the configured base endpoint).
    """
    kv_events_config = _enabled_kv_events_config(llm_config)
    if kv_events_config is None:
        return
    base_endpoint = kv_events_config.get("endpoint") or _default_kv_events_endpoint(
        llm_config
    )
    if _engine_data_parallel_rank(llm_config) is not None:
        endpoint = base_endpoint
    else:
        endpoint = _offset_endpoint_port(base_endpoint, _replica_rank())
    llm_config.update_engine_kwargs(
        kv_events_config={**kv_events_config, "endpoint": endpoint}
    )


def resolve_kv_event_source_endpoint(llm_config: LLMConfig) -> Optional[str]:
    """The ZMQ endpoint a replica's ``KvEventPublisher`` should consume.

    With Dynamo's KVBM connector active this is the consolidator's output
    stream (which carries the engine's events bridged with KVBM's own);
    otherwise it is the engine's KV-events endpoint. ``None`` when KV-cache
    events are not enabled.
    """
    kv_events_config = _enabled_kv_events_config(llm_config)
    if kv_events_config is None:
        return None
    if _uses_dynamo_connector(llm_config):
        return resolve_consolidator_endpoints(llm_config)[2]
    return _engine_event_connect_endpoint(llm_config, kv_events_config)


def resolve_consolidator_endpoints(llm_config: LLMConfig) -> List[str]:
    """Per-replica KVBM consolidator endpoints (RFC §4.2).

    Returns ``[engine_event_endpoint, output_bind_endpoint,
    output_connect_endpoint]``: the consolidator subscribes to the engine's
    KV events and republishes the consolidated stream on its output port.
    """
    kv_events_config = _enabled_kv_events_config(llm_config)
    if kv_events_config is None:
        raise ValueError(
            "The KVBM consolidator requires kv_events_config with "
            "enable_kv_cache_events=True in engine_kwargs."
        )
    dp_rank = _engine_data_parallel_rank(llm_config)
    offset = dp_rank if dp_rank is not None else _replica_rank()
    consolidator_port = (
        _experimental_port_base(
            llm_config,
            KV_EVENT_CONSOLIDATOR_PORT_BASE_KEY,
            DEFAULT_KV_EVENT_CONSOLIDATOR_PORT_BASE,
        )
        + offset
    )
    return [
        _engine_event_connect_endpoint(llm_config, kv_events_config),
        f"tcp://0.0.0.0:{consolidator_port}",
        f"tcp://127.0.0.1:{consolidator_port}",
    ]


def _engine_event_connect_endpoint(
    llm_config: LLMConfig, kv_events_config: Dict[str, Any]
) -> str:
    """The localhost endpoint the engine's KV events are consumable from.

    The engine offsets its bind port by ``data_parallel_rank`` itself; the
    replica-rank case is already offset in the configured endpoint by
    :func:`assign_replica_kv_events_endpoint`.
    """
    endpoint = kv_events_config["endpoint"]
    dp_rank = _engine_data_parallel_rank(llm_config)
    if dp_rank is not None:
        endpoint = _offset_endpoint_port(endpoint, dp_rank)
    return _to_connect_endpoint(endpoint)


def _uses_dynamo_connector(llm_config: LLMConfig) -> bool:
    kv_transfer_config = llm_config.engine_kwargs.get("kv_transfer_config")
    return (
        isinstance(kv_transfer_config, dict)
        and kv_transfer_config.get("kv_connector") == DYNAMO_KV_CONNECTOR
    )


def _enabled_kv_events_config(llm_config: LLMConfig) -> Optional[Dict[str, Any]]:
    kv_events_config = llm_config.engine_kwargs.get("kv_events_config")
    if isinstance(kv_events_config, dict) and kv_events_config.get(
        "enable_kv_cache_events"
    ):
        return kv_events_config
    return None


def _engine_data_parallel_rank(llm_config: LLMConfig) -> Optional[int]:
    dp_rank = llm_config.engine_kwargs.get("data_parallel_rank")
    return dp_rank if isinstance(dp_rank, int) and dp_rank >= 0 else None


def _replica_rank() -> int:
    """This replica's rank on its node (ports are node-local), 0 outside a replica."""
    try:
        return serve.get_replica_context().rank.local_rank
    except RayServeException:
        return 0


def _experimental_port_base(llm_config: LLMConfig, key: str, default: int) -> int:
    return int(llm_config.experimental_configs.get(key, default))


def _default_kv_events_endpoint(llm_config: LLMConfig) -> str:
    port_base = _experimental_port_base(
        llm_config, KV_EVENTS_PORT_BASE_KEY, DEFAULT_KV_EVENTS_PORT_BASE
    )
    return f"tcp://*:{port_base}"


def _offset_endpoint_port(endpoint: str, offset: int) -> str:
    """Offset a TCP endpoint's port (vLLM's ZmqEventPublisher convention)."""
    if offset == 0:
        return endpoint
    if not endpoint.startswith("tcp://") or ":" not in endpoint[len("tcp://") :]:
        raise ValueError(
            f"Cannot offset port of non-TCP KV events endpoint: {endpoint}"
        )
    base, port = endpoint.rsplit(":", 1)
    return f"{base}:{int(port) + offset}"


def _to_connect_endpoint(endpoint: str) -> str:
    """Convert a bind-form TCP endpoint to its localhost connect form."""
    if endpoint.startswith("tcp://"):
        host, port = endpoint[len("tcp://") :].rsplit(":", 1)
        if host in ("*", "0.0.0.0", "::"):
            return f"tcp://127.0.0.1:{port}"
    return endpoint
