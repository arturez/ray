import asyncio
import logging
from typing import Optional

from dynamo.llm import KvEventPublisher

from ray import serve
from ray.actor import ActorHandle
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
    KV_ROUTER_ACTOR_NAME,
    get_worker_id,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_event_plane import (
    configure_kv_event_plane_env,
    create_kv_event_plane_runtime,
    dynamo_namespace,
    kv_events_endpoint_path,
    replica_deployment_id,
    resolve_kv_event_block_size,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (
    resolve_kv_event_source_endpoint,
)
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve.exceptions import RayServeException

logger = logging.getLogger(SERVE_LOGGER_NAME)

# Number of no-op events publishing burns before the engine's events start.
_PRIMING_EVENTS = 4
# Removing a block hash that is never stored is a state no-op on every indexer.
_PRIMING_BLOCK_HASH = -(0x52415945 << 16)


class ReplicaKvEventPublisher:
    """Replica-side Dynamo ``KvEventPublisher`` (RFC §4.3.1).

    Bridges the engine's (or, with KVBM, the consolidator's) ZMQ KV-event
    stream into Dynamo's ``kv-events`` event plane, where the deployment's
    ``KVRouterActor``-hosted ``KvRouter`` consumes it into the global KV
    indexer. Ray owns replica identity, so the Ray-derived ``worker_id`` is
    supplied to Dynamo (RFC §4.3) and registered with the ``KVRouterActor``.

    Must be started in a running asyncio event loop.
    """

    def __init__(
        self,
        kv_router_actor: ActorHandle,
        replica_id: str,
        worker_id: int,
        namespace: str,
        zmq_endpoint: str,
        kv_block_size: int,
        dp_rank: int = 0,
    ):
        self._kv_router_actor = kv_router_actor
        self._replica_id = replica_id
        self._worker_id = worker_id
        self._namespace = namespace
        self._zmq_endpoint = zmq_endpoint
        self._kv_block_size = kv_block_size
        self._dp_rank = dp_rank
        self._runtime = None
        self._publisher = None

    @property
    def worker_id(self) -> int:
        """This replica's Dynamo worker id."""
        return self._worker_id

    async def start(self) -> None:
        # Register first: this instantiates the actor's KvRouter (and its
        # KvEventConsumer) before this replica appears on the event plane,
        # so the start of its stream is not dropped.
        await self._kv_router_actor.register_kv_event_worker.remote(
            self._worker_id, self._replica_id, self._kv_block_size
        )

        configure_kv_event_plane_env(self._namespace)
        self._runtime = create_kv_event_plane_runtime(asyncio.get_running_loop())
        endpoint = self._runtime.endpoint(kv_events_endpoint_path(self._namespace))

        # enable_local_indexer=True is required for the publisher to publish
        # through the event plane; it also maintains the worker-local indexer
        # the router re-syncs this worker's view from (RFC §4.5).
        self._publisher = KvEventPublisher(
            endpoint,
            worker_id=self._worker_id,
            kv_block_size=self._kv_block_size,
            zmq_endpoint=self._zmq_endpoint,
            zmq_topic="",
            enable_local_indexer=True,
            dp_rank=self._dp_rank,
        )
        self._prime_event_stream()
        logger.info(
            "Dynamo KvEventPublisher started for worker %d (replica %s), "
            "consuming KV events from %s.",
            self._worker_id,
            self._replica_id,
            self._zmq_endpoint,
        )

    def _prime_event_stream(self) -> None:
        """Burn the first event ids so the engine's events are never stale.

        The router restores a newly seen worker from its local indexer, and a
        restore against a still-empty worker reports ``last_event_id=0``,
        marking event id 0 as already applied: a real first event would be
        silently discarded as stale, orphaning every block chained on it.
        After priming, real events carry ids strictly above any id a restore
        can claim, so the router either applies them in order or detects a
        gap and re-syncs from this worker's (always-current) local indexer.
        """
        for _ in range(_PRIMING_EVENTS):
            self._publisher.publish_removed([_PRIMING_BLOCK_HASH])

    def close(self) -> None:
        """Shut down the publisher and its Dynamo runtime."""
        if self._publisher is not None:
            self._publisher.shutdown()
            self._publisher = None
        if self._runtime is not None:
            self._runtime.shutdown()
            self._runtime = None


async def maybe_start_kv_event_publisher(
    llm_config: LLMConfig, engine_block_size: int
) -> Optional[ReplicaKvEventPublisher]:
    """Start this replica's KvEventPublisher if KV-aware routing is set up.

    Requires KV-cache events enabled in ``engine_kwargs`` and the
    deployment-scoped ``KVRouterActor`` (attached when the deployment routes
    with ``KVAwareRouter``); returns ``None`` otherwise.
    """
    zmq_endpoint = resolve_kv_event_source_endpoint(llm_config)
    if zmq_endpoint is None:
        return None
    try:
        kv_router_actor = serve.get_deployment_actor(KV_ROUTER_ACTOR_NAME)
        replica_context = serve.get_replica_context()
    except (RayServeException, ValueError):
        # Outside a replica, or the actor is not attached to this deployment
        # (KV events were enabled for an external consumer).
        logger.info(
            "KV-cache events are enabled but no %s deployment actor is "
            "reachable; not publishing KV events.",
            KV_ROUTER_ACTOR_NAME,
        )
        return None

    publisher = ReplicaKvEventPublisher(
        kv_router_actor=kv_router_actor,
        replica_id=replica_context.replica_id.to_full_id_str(),
        worker_id=get_worker_id(replica_context.replica_id.unique_id),
        namespace=dynamo_namespace(replica_deployment_id()),
        zmq_endpoint=zmq_endpoint,
        kv_block_size=resolve_kv_event_block_size(
            engine_block_size, llm_config.engine_kwargs.get("additional_config")
        ),
        dp_rank=llm_config.engine_kwargs.get("data_parallel_rank") or 0,
    )
    await publisher.start()
    return publisher
