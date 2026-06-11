import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import ray
from ray import serve
from ray.llm._internal.serve.routing_policies.kv_aware.kv_event_plane import (
    configure_kv_event_plane_env,
    create_kv_event_plane_runtime,
    dynamo_namespace,
    kv_events_endpoint_path,
    reset_kv_event_plane_dir,
)
from ray.serve._private.common import DeploymentTargetInfo, ReplicaID
from ray.serve._private.constants import (
    SERVE_CONTROLLER_NAME,
    SERVE_LOGGER_NAME,
    SERVE_NAMESPACE,
)
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace

logger = logging.getLogger(SERVE_LOGGER_NAME)

KV_ROUTER_ACTOR_NAME = "serve_llm_kv_router"


def get_worker_id(replica_unique_id: str) -> int:
    """
    Deterministically derive a Dynamo worker id from a replica's unique id.
    """
    return int.from_bytes(
        hashlib.blake2b(replica_unique_id.encode(), digest_size=8).digest(), "big"
    )


@ray.remote
class KVRouterActor:
    """Deployment-scoped Ray actor hosting the KV-aware router.

    KVRouterActor, independent of any replica's lifetime, is attached to the LLMServer
    deployment via Serve's DeploymentActorConfig. It owns the deployment's Dynamo
    ``KvRouter``, whose ``KvEventConsumer`` consumes the replicas' KV events from the
    event plane into the global KV indexer. It exposes the KV-aware routing interfaces:
    - Replica membership tracking
    - KV-aware scoring

    TODO (jeffreywang): Scoring routes to Dynamo once ``rank_workers`` lands (RFC §4.4.1).
    """

    def __init__(self):
        self._replica_id_by_worker: Dict[int, str] = {}
        self._long_poll_client: Optional[LongPollClient] = None
        self._kv_router = None
        self._kv_router_runtime = None
        self._kv_router_block_size: Optional[int] = None
        self._replica_by_kv_event_worker: Dict[int, str] = {}
        # This actor starts with the deployment, before any replica of its
        # incarnation registers, so it owns clearing the previous
        # incarnation's discovery state.
        reset_kv_event_plane_dir(self._kv_event_plane_namespace())
        self._start_replica_tracking()

    def _start_replica_tracking(self) -> None:
        """Subscribe to this deployment's running replicas via LongPollClient."""
        deployment_id = serve.get_deployment_actor_context().deployment_id
        controller = ray.get_actor(SERVE_CONTROLLER_NAME, namespace=SERVE_NAMESPACE)
        self._long_poll_client = LongPollClient(
            controller,
            {
                (
                    LongPollNamespace.DEPLOYMENT_TARGETS,
                    deployment_id,
                ): self._on_deployment_targets,
            },
            call_in_event_loop=asyncio.get_event_loop(),
            client_id=f"{type(self).__name__}:{deployment_id}",
        )

    def _on_deployment_targets(self, target_info: DeploymentTargetInfo) -> None:
        """LongPoll listener: sync the worker mapping to the running replicas.

        Diffs the snapshot against the current mapping and updates the reverse
        worker->replica map.
        """
        new = {
            get_worker_id(r.replica_id.unique_id): r.replica_id.to_full_id_str()
            for r in target_info.running_replicas
        }
        current = set(self._replica_id_by_worker)
        added = new.keys() - current
        removed = current - new.keys()

        for worker_id in removed:
            self.remove_worker(worker_id)
            self._replica_id_by_worker.pop(worker_id, None)
            # A removed replica's KV-event registration goes with it. Keyed
            # by worker id, not the running set: replicas register while
            # still STARTING, before they appear in this snapshot.
            self._replica_by_kv_event_worker.pop(worker_id, None)
        for worker_id in added:
            self._replica_id_by_worker[worker_id] = new[worker_id]
            self.add_worker(worker_id)

        if added or removed:
            logger.info(
                "KV router replica membership updated: +%d -%d, tracking %d worker(s).",
                len(added),
                len(removed),
                len(self._replica_id_by_worker),
            )

    def get_tracked_worker_id(self, replica_id_full_str: str) -> Optional[int]:
        """Return the worker id for a currently-tracked replica, or ``None``."""
        unique_id = ReplicaID.from_full_id_str(replica_id_full_str).unique_id
        worker_id = get_worker_id(unique_id)
        return worker_id if worker_id in self._replica_id_by_worker else None

    def get_replica_id(self, worker_id: int) -> Optional[str]:
        """Return the replica's full id string for ``worker_id``, or ``None``."""
        return self._replica_id_by_worker.get(worker_id)

    def get_candidate_worker_ids(self) -> List[int]:
        """Return the currently tracked worker ids, sorted ascending."""
        return sorted(self._replica_id_by_worker)

    def add_worker(self, worker_id: int) -> None:
        """Register a worker with the KV router when a replica is added."""
        pass

    def remove_worker(self, worker_id: int) -> None:
        """Deregister a worker from the KV router when a replica is removed."""
        pass

    async def register_kv_event_worker(
        self, worker_id: int, replica_id: str, kv_block_size: int
    ) -> None:
        """Register a replica's KV-event identity before it publishes.

        Called by each replica's ``ReplicaKvEventPublisher`` on startup with
        the Ray-derived worker id its events are keyed by (RFC §4.3).
        Instantiates the deployment's ``KvRouter`` on first registration, so
        its ``KvEventConsumer`` subscribes to the event plane before any event
        is published.
        """
        self._ensure_kv_router(kv_block_size)
        self._replica_by_kv_event_worker[worker_id] = replica_id
        logger.info(
            "Registered KV event worker %d for replica %s (%d registered).",
            worker_id,
            replica_id,
            len(self._replica_by_kv_event_worker),
        )

    def _ensure_kv_router(self, kv_block_size: int) -> None:
        """Create the Dynamo ``KvRouter`` consuming this deployment's KV events.

        Lazy: the actor starts before any replica, and the router's block size
        must match the engines' KV-event block size, which replicas resolve.
        """
        # Imported here, not at module scope: Ray pickles this actor class by
        # value, and Dynamo's pyo3 classes cannot be pickled as its globals.
        from dynamo.llm import KvRouter, KvRouterConfig

        if self._kv_router is not None:
            if kv_block_size != self._kv_router_block_size:
                logger.warning(
                    "KV event worker registered with block size %d but the "
                    "KvRouter indexes at block size %d; its events will not "
                    "match.",
                    kv_block_size,
                    self._kv_router_block_size,
                )
            return

        namespace = self._kv_event_plane_namespace()
        configure_kv_event_plane_env(namespace)
        self._kv_router_runtime = create_kv_event_plane_runtime(
            asyncio.get_running_loop()
        )
        endpoint = self._kv_router_runtime.endpoint(kv_events_endpoint_path(namespace))
        # durable_kv_events=False: events arrive over the event plane (RFC
        # §4.3.3), not NATS JetStream.
        self._kv_router = KvRouter(
            endpoint=endpoint,
            block_size=kv_block_size,
            kv_router_config=KvRouterConfig(
                use_kv_events=True,
                durable_kv_events=False,
            ),
        )
        self._kv_router_block_size = kv_block_size
        logger.info(
            "Dynamo KvRouter created for namespace %s (block size %d).",
            namespace,
            kv_block_size,
        )

    def _kv_event_plane_namespace(self) -> str:
        """The Dynamo namespace scoping this deployment's KV events."""
        return dynamo_namespace(serve.get_deployment_actor_context().deployment_id)

    def get_kv_event_worker_replicas(self) -> Dict[int, str]:
        """The registered Dynamo worker id -> replica full id mapping."""
        return dict(self._replica_by_kv_event_worker)

    async def get_kv_indexer_events(self) -> List[Dict[str, Any]]:
        """The KV events applied to the router's global indexer.

        Dynamo's ``KvRouter.dump_events``: each entry carries the worker id,
        storage tier, and the stored/removed event payload (block hashes and
        per-block token hashes). Empty before the router exists.
        """
        if self._kv_router is None:
            return []
        return json.loads(await self._kv_router.dump_events())

    async def get_kv_event_worker_ids(self) -> List[int]:
        """Workers with at least one event in the global indexer, sorted."""
        return sorted(
            {event["worker_id"] for event in await self.get_kv_indexer_events()}
        )

    async def get_kv_overlap_blocks(self, token_ids: List[int]) -> Dict[int, int]:
        """Per-worker device-tier KV overlap blocks for a token sequence.

        The global indexer's view of how many leading blocks of ``token_ids``
        each worker has cached: the overlap input to KV-aware scoring.
        """
        if self._kv_router is None:
            return {}
        scores = await self._kv_router.get_overlap_scores(token_ids)
        return {
            worker["worker_id"]: worker["device_blocks"] for worker in scores["workers"]
        }

    async def select_worker(
        self,
        request_id: str,
        token_ids: List[int],
        allowed_worker_ids: List[int],
    ) -> Dict[str, Any]:
        """Score the allowed workers for a request based on KV-cache overlap and
        load and pick the best one.

        Args:
            request_id: Unique identifier for the request being routed.
            token_ids: Prompt token ids used to compute KV-cache overlap.
            allowed_worker_ids: Candidate worker ids the router may select from.

        Returns:
            A dict describing the selected worker:
            ``worker_id`` (int): the chosen worker.
            ``dp_rank`` (int): data-parallel rank within the worker.
            ``overlap_blocks`` (int): KV blocks already cached on that worker.
            ``score`` (float): the worker's routing score (higher is better).
        """
        raise NotImplementedError("KVRouterActor.select_worker is not implemented")

    async def add_request(
        self,
        request_id: str,
        expected_output_tokens: Optional[int] = None,
    ) -> None:
        """Commit a routed request to the worker chosen by ``select_worker``.

        Args:
            request_id: Unique identifier for the request.
            expected_output_tokens: Predicted number of output tokens, used to
                estimate the request's decode-phase KV footprint.
        """
        raise NotImplementedError("KVRouterActor.add_request is not implemented")

    async def mark_prefill_complete(self, request_id: str) -> None:
        """Signals the transition from prefill to decode.

        Args:
            request_id: Unique identifier for the request.
        """
        raise NotImplementedError(
            "KVRouterActor.mark_prefill_complete is not implemented"
        )

    async def free(self, request_id: str) -> None:
        """Release the KV-cache accounting for a finished request.

        Args:
            request_id: Unique identifier for the request.
        """
        raise NotImplementedError("KVRouterActor.free is not implemented")
