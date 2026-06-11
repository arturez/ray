import logging
import os
import re
import shutil
import tempfile
from typing import Optional

from dynamo.runtime import DistributedRuntime

from ray.serve._private.common import DeploymentID
from ray.serve._private.constants import SERVE_LOGGER_NAME

logger = logging.getLogger(SERVE_LOGGER_NAME)

# Process config required on both the replica (KvEventPublisher) and the
# router actor (KvRouter / KvEventConsumer) sides of Dynamo's event plane
# (RFC §4.3.3). Set before the DistributedRuntime is created.
DYN_FILE_KV_ENV = "DYN_FILE_KV"
KV_EVENT_PLANE_ENV_DEFAULTS = {
    "DYN_EVENT_PLANE": "zmq",
    "DYN_ROUTER_USE_KV_EVENTS": "true",
    "DYN_ROUTER_DURABLE_KV_EVENTS": "false",
}

# The deployment-scoped Dynamo component endpoint. Replicas publish KV events
# scoped to it and the router actor's consumer subscribes to the same scope.
KV_EVENTS_ENDPOINT_SUFFIX = "backend.generate"


def dynamo_namespace(deployment_id: DeploymentID) -> str:
    """The Dynamo namespace for a deployment.

    Per-deployment so KV events of different models never share an event
    scope. Derived from the deployment identity alone, so replicas and the
    deployment actor compute it independently.
    """
    raw = f"ray_llm_{deployment_id.app_name}_{deployment_id.name}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)


def kv_events_endpoint_path(namespace: str) -> str:
    """The Dynamo endpoint path scoping a deployment's KV events."""
    return f"{namespace}.{KV_EVENTS_ENDPOINT_SUFFIX}"


def configure_kv_event_plane_env(namespace: str) -> None:
    """Configure Dynamo's discovery and event plane for this process.

    Uses file-based discovery (no etcd/NATS) with a per-deployment directory
    and ZMQ as the event plane, the local-deployment setup from RFC §4.3.3.
    Pre-set ``DYN_*`` variables are respected, which is how multi-node
    clusters point discovery at a shared filesystem path.
    """
    os.environ.setdefault(DYN_FILE_KV_ENV, _default_discovery_dir(namespace))
    for key, value in KV_EVENT_PLANE_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def reset_kv_event_plane_dir(namespace: str) -> None:
    """Purge stale discovery state from prior deployment incarnations.

    Dead replicas cannot unregister their event channels and local-indexer
    endpoints; the router would keep connecting to and scheduling recovery
    against their corpses, starving and staling live workers' recovery.
    Called by the ``KVRouterActor`` on startup, before any replica of its
    incarnation registers. Only applies to the default per-deployment
    directory; an externally managed ``DYN_FILE_KV`` is left untouched.
    """
    default_dir = _default_discovery_dir(namespace)
    if os.environ.get(DYN_FILE_KV_ENV, default_dir) != default_dir:
        return
    shutil.rmtree(default_dir, ignore_errors=True)
    os.makedirs(default_dir, exist_ok=True)


def create_kv_event_plane_runtime(loop) -> DistributedRuntime:
    """Create the Dynamo runtime backing this process's KV event plane.

    File-based discovery with the TCP request plane; must be created on a
    running asyncio event loop, after :func:`configure_kv_event_plane_env`.
    """
    return DistributedRuntime(loop, "file", "tcp", False)


def resolve_kv_event_block_size(engine_block_size: int, additional_config) -> int:
    """The block size KV events are chunked at.

    Mirrors Dynamo's ``get_configured_kv_event_block_size``: KVBM can re-chunk
    events at ``additional_config["dynamo_kv_event_block_size"]``; otherwise
    events use the engine's cache block size.
    """
    if isinstance(additional_config, dict):
        return int(
            additional_config.get("dynamo_kv_event_block_size", engine_block_size)
        )
    return engine_block_size


def _default_discovery_dir(namespace: str) -> str:
    """Per-deployment discovery directory for Dynamo's file KV store.

    Node-local by default, which assumes a single-node cluster; multi-node
    deployments must set ``DYN_FILE_KV`` to a shared filesystem path.
    """
    path = os.path.join(tempfile.gettempdir(), "ray_serve_llm_kv_events", namespace)
    os.makedirs(path, exist_ok=True)
    return path


def replica_deployment_id() -> Optional[DeploymentID]:
    """The deployment id when called inside a Serve replica, else ``None``."""
    from ray import serve
    from ray.serve.exceptions import RayServeException

    try:
        return serve.get_replica_context().replica_id.deployment_id
    except RayServeException:
        return None
