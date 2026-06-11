import sys
import uuid

import pytest
from dynamo.llm import compute_block_hash_for_seq
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)

import ray
from ray._common.test_utils import async_wait_for_condition
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.core.server.builder import build_llm_deployment
from ray.llm._internal.serve.engines.vllm.kv_transfer.factory import (
    KVConnectorBackendFactory,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
    KVRouterActor,
    get_worker_id,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_event_plane import (
    dynamo_namespace,
    resolve_kv_event_block_size,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_event_publisher import (
    ReplicaKvEventPublisher,
    maybe_start_kv_event_publisher,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (
    DYNAMO_KV_CONNECTOR,
    DYNAMO_KV_CONNECTOR_MODULE_PATH,
    assign_replica_kv_events_endpoint,
    configure_kv_events_for_kv_routing,
    resolve_consolidator_endpoints,
    resolve_kv_event_source_endpoint,
)
from ray.serve._private.common import DeploymentID, DeploymentTargetInfo, ReplicaID
from ray.serve.llm.request_router import KVAwareRouter

BLOCK_SIZE = 16


def make_llm_config(**kwargs) -> LLMConfig:
    return LLMConfig(
        model_loading_config={
            "model_id": "qwen-0.5b",
            "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        accelerator_type=None,
        **kwargs,
    )


def make_kv_aware_llm_config(**kwargs) -> LLMConfig:
    return make_llm_config(
        deployment_config={
            "autoscaling_config": {"min_replicas": 1, "max_replicas": 1},
            "request_router_config": {"request_router_class": KVAwareRouter},
        },
        **kwargs,
    )


class TestConfigureKvEvents:
    def test_build_enables_kv_events(self):
        """Building a KVAwareRouter deployment enables engine KV events."""
        llm_config = make_kv_aware_llm_config()
        build_llm_deployment(llm_config)

        assert llm_config.engine_kwargs["kv_events_config"] == {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": "tcp://*:5557",
        }
        # The Dynamo KVBM connector is not installed in this environment.
        assert "kv_transfer_config" not in llm_config.engine_kwargs

    def test_build_without_kv_aware_router_is_untouched(self):
        llm_config = make_llm_config(
            deployment_config={
                "autoscaling_config": {"min_replicas": 1, "max_replicas": 1}
            },
        )
        build_llm_deployment(llm_config)

        assert "kv_events_config" not in llm_config.engine_kwargs

    def test_user_kv_events_config_is_respected(self):
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={
                "kv_events_config": {
                    "enable_kv_cache_events": True,
                    "publisher": "zmq",
                    "endpoint": "tcp://*:6000",
                    "buffer_steps": 5,
                }
            },
        )
        build_llm_deployment(llm_config)

        assert llm_config.engine_kwargs["kv_events_config"]["endpoint"] == (
            "tcp://*:6000"
        )
        assert llm_config.engine_kwargs["kv_events_config"]["buffer_steps"] == 5

    def test_port_base_override(self):
        llm_config = make_kv_aware_llm_config(
            experimental_configs={"KV_EVENTS_PORT_BASE": 21000},
        )
        configure_kv_events_for_kv_routing(llm_config)

        assert llm_config.engine_kwargs["kv_events_config"]["endpoint"] == (
            "tcp://*:21000"
        )

    def test_block_hash_seed_pinned(self):
        """Replicas must hash identical content identically: the router's
        global indexer chains blocks by the engines' block hashes."""
        llm_config = make_kv_aware_llm_config()
        build_llm_deployment(llm_config)

        assert llm_config.runtime_env["env_vars"]["PYTHONHASHSEED"] == "0"

        user_config = make_kv_aware_llm_config(
            runtime_env={"env_vars": {"PYTHONHASHSEED": "7"}},
        )
        configure_kv_events_for_kv_routing(user_config)

        assert user_config.runtime_env["env_vars"]["PYTHONHASHSEED"] == "7"

    def test_dynamo_connector_injected_when_kvbm_installed(self, monkeypatch):
        """With the kvbm package importable, the Dynamo connector is selected."""
        monkeypatch.setattr(
            "ray.llm._internal.serve.routing_policies.kv_aware.kv_events."
            "importlib.util.find_spec",
            lambda name: object() if name == "kvbm" else None,
        )
        llm_config = make_kv_aware_llm_config()
        configure_kv_events_for_kv_routing(llm_config)

        assert llm_config.engine_kwargs["kv_transfer_config"] == {
            "kv_connector": DYNAMO_KV_CONNECTOR,
            "kv_connector_module_path": DYNAMO_KV_CONNECTOR_MODULE_PATH,
            "kv_role": "kv_both",
        }

    def test_user_kv_transfer_config_is_respected(self, monkeypatch):
        monkeypatch.setattr(
            "ray.llm._internal.serve.routing_policies.kv_aware.kv_events."
            "importlib.util.find_spec",
            lambda name: object(),
        )
        user_config = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={"kv_transfer_config": dict(user_config)},
        )
        configure_kv_events_for_kv_routing(llm_config)

        assert llm_config.engine_kwargs["kv_transfer_config"] == user_config


class TestReplicaEndpoints:
    @pytest.fixture
    def replica_rank(self, monkeypatch):
        def set_rank(rank):
            monkeypatch.setattr(
                "ray.llm._internal.serve.routing_policies.kv_aware.kv_events."
                "_replica_rank",
                lambda: rank,
            )

        return set_rank

    def test_no_kv_events_is_noop(self):
        llm_config = make_llm_config()
        assign_replica_kv_events_endpoint(llm_config)

        assert "kv_events_config" not in llm_config.engine_kwargs
        assert resolve_kv_event_source_endpoint(llm_config) is None

    def test_replica_rank_offsets_port(self, replica_rank):
        """Colocated replicas must bind distinct KV-events ports."""
        replica_rank(2)
        llm_config = make_kv_aware_llm_config()
        configure_kv_events_for_kv_routing(llm_config)
        assign_replica_kv_events_endpoint(llm_config)

        assert llm_config.engine_kwargs["kv_events_config"]["endpoint"] == (
            "tcp://*:5559"
        )
        assert resolve_kv_event_source_endpoint(llm_config) == "tcp://127.0.0.1:5559"

    def test_data_parallel_rank_is_offset_by_vllm(self, replica_rank):
        """With data_parallel_rank, vLLM offsets the bind port internally, so
        the configured endpoint stays at the base and only the subscriber
        endpoint is offset."""
        replica_rank(5)
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={"data_parallel_rank": 3},
        )
        configure_kv_events_for_kv_routing(llm_config)
        assign_replica_kv_events_endpoint(llm_config)

        endpoint = llm_config.engine_kwargs["kv_events_config"]["endpoint"]
        assert endpoint == "tcp://*:5557"
        assert resolve_kv_event_source_endpoint(llm_config) == "tcp://127.0.0.1:5560"
        offset_by_vllm = ZmqEventPublisher.offset_endpoint_port(endpoint, 3)
        assert offset_by_vllm == "tcp://*:5560"

    def test_consolidator_endpoints_with_dynamo_connector(self, replica_rank):
        """The DynamoConnector backend wires per-replica consolidator
        endpoints into additional_config and the publisher consumes the
        consolidated stream."""
        replica_rank(1)
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={
                "kv_transfer_config": {
                    "kv_connector": DYNAMO_KV_CONNECTOR,
                    "kv_connector_module_path": DYNAMO_KV_CONNECTOR_MODULE_PATH,
                    "kv_role": "kv_both",
                }
            },
        )
        configure_kv_events_for_kv_routing(llm_config)
        assign_replica_kv_events_endpoint(llm_config)

        backend = KVConnectorBackendFactory.create_backend(
            DYNAMO_KV_CONNECTOR, llm_config
        )
        backend.setup()

        assert llm_config.engine_kwargs["additional_config"][
            "consolidator_endpoints"
        ] == [
            "tcp://127.0.0.1:5558",
            "tcp://0.0.0.0:57002",
            "tcp://127.0.0.1:57002",
        ]
        assert resolve_kv_event_source_endpoint(llm_config) == "tcp://127.0.0.1:57002"

    def test_user_consolidator_endpoints_are_respected(self, replica_rank):
        replica_rank(0)
        user_endpoints = ["tcp://127.0.0.1:1", "tcp://0.0.0.0:2", "tcp://127.0.0.1:2"]
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={
                "kv_transfer_config": {"kv_connector": DYNAMO_KV_CONNECTOR},
                "additional_config": {"consolidator_endpoints": user_endpoints},
            },
        )
        configure_kv_events_for_kv_routing(llm_config)
        assign_replica_kv_events_endpoint(llm_config)
        KVConnectorBackendFactory.create_backend(
            DYNAMO_KV_CONNECTOR, llm_config
        ).setup()

        assert (
            llm_config.engine_kwargs["additional_config"]["consolidator_endpoints"]
            is user_endpoints
        )

    def test_consolidator_requires_kv_events(self):
        llm_config = make_kv_aware_llm_config()
        with pytest.raises(ValueError, match="kv_events_config"):
            resolve_consolidator_endpoints(llm_config)


class TestKvEventPlaneConfig:
    def test_namespace_is_deployment_scoped_and_sanitized(self):
        deployment_id = DeploymentID(name="LLMServer:qwen-0.5b", app_name="my.app")
        assert dynamo_namespace(deployment_id) == "ray_llm_my_app_LLMServer_qwen-0_5b"

    def test_kv_event_block_size_honors_kvbm_override(self):
        """KVBM can re-chunk events at dynamo_kv_event_block_size."""
        assert resolve_kv_event_block_size(16, None) == 16
        assert resolve_kv_event_block_size(16, {"dynamo_kv_event_block_size": 64}) == 64

    @pytest.mark.asyncio
    async def test_no_publisher_without_kv_events(self):
        assert await maybe_start_kv_event_publisher(make_llm_config(), 16) is None

    @pytest.mark.asyncio
    async def test_no_publisher_outside_replica(self):
        llm_config = make_kv_aware_llm_config()
        configure_kv_events_for_kv_routing(llm_config)
        assert await maybe_start_kv_event_publisher(llm_config, 16) is None


@pytest.fixture(scope="module")
def ray_instance():
    if not ray.is_initialized():
        ray.init(address="auto")
    yield


@ray.remote(num_cpus=0)
class LocalKVRouterActor(KVRouterActor.__ray_actor_class__):
    """The real KVRouterActor with a fixed Dynamo namespace and replica
    tracking disabled (no Serve controller in these tests)."""

    def __init__(self, namespace: str):
        self._namespace = namespace
        super().__init__()

    def _start_replica_tracking(self) -> None:
        pass

    def _kv_event_plane_namespace(self) -> str:
        return self._namespace

    def apply_running_replicas(self, replica_full_ids) -> None:
        """Feed a replica-membership snapshot as the LongPoll listener would."""
        from ray.serve._private.common import RunningReplicaInfo

        self._on_deployment_targets(
            DeploymentTargetInfo(
                is_available=True,
                running_replicas=[
                    RunningReplicaInfo(
                        replica_id=ReplicaID.from_full_id_str(full_id),
                        node_id=None,
                        node_ip=None,
                        availability_zone=None,
                        actor_name=f"actor-{full_id}",
                        max_ongoing_requests=10,
                    )
                    for full_id in replica_full_ids
                ],
            )
        )


@ray.remote(num_cpus=0)
class ReplicaStandIn:
    """A replica stand-in: vLLM's production ZmqEventPublisher as the engine
    and the real ReplicaKvEventPublisher bridging it to the event plane."""

    def __init__(self, kv_router_actor, replica_id, worker_id, namespace, port):
        self._engine_pub = ZmqEventPublisher(
            data_parallel_rank=0, endpoint=f"tcp://*:{port}", topic=""
        )
        self._publisher = ReplicaKvEventPublisher(
            kv_router_actor=kv_router_actor,
            replica_id=replica_id,
            worker_id=worker_id,
            namespace=namespace,
            zmq_endpoint=f"tcp://127.0.0.1:{port}",
            kv_block_size=BLOCK_SIZE,
        )

    async def start(self) -> int:
        await self._publisher.start()
        return self._publisher.worker_id

    def publish_stored(self, block_hashes, token_ids):
        self._engine_pub.publish(
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockStored(
                        block_hashes=list(block_hashes),
                        parent_block_hash=None,
                        token_ids=list(token_ids),
                        block_size=BLOCK_SIZE,
                        lora_id=None,
                        medium="GPU",
                        lora_name=None,
                    )
                ],
            )
        )

    def publish_removed(self, block_hashes):
        self._engine_pub.publish(
            KVEventBatch(
                ts=2.0,
                events=[BlockRemoved(block_hashes=list(block_hashes), medium="GPU")],
            )
        )

    def publish_cleared(self):
        self._engine_pub.publish(KVEventBatch(ts=3.0, events=[AllBlocksCleared()]))

    def close(self):
        self._publisher.close()
        self._engine_pub.shutdown()


def stored_block_hashes(indexer_events, worker_id):
    """Engine block hashes currently stored for a worker in the indexer dump."""
    hashes = set()
    for entry in indexer_events:
        if entry["worker_id"] != worker_id:
            continue
        for block in entry["event"]["data"]["stored"]["blocks"]:
            hashes.add(block["block_hash"])
    return hashes


def stored_tokens_hashes(indexer_events, worker_id):
    """Dynamo per-block token hashes stored for a worker in the indexer dump."""
    return {
        block["tokens_hash"]
        for entry in indexer_events
        if entry["worker_id"] == worker_id
        for block in entry["event"]["data"]["stored"]["blocks"]
    }


async def wait_for_indexer(actor, predicate, publish=None, timeout=20):
    """Wait until the actor's indexer dump satisfies ``predicate``.

    ``publish`` re-publishes the step's event while waiting, and must be
    idempotent: events published before the event-plane subscription is live
    can be dropped, and the router pulls worker snapshots asynchronously, so
    a single publish observed via the dump is racy. Re-storing the same
    blocks, re-removing an absent block, and re-clearing are all state
    no-ops that still trigger a fresh snapshot pull.
    """

    async def condition():
        if publish is not None:
            publish()
        return predicate(await actor.get_kv_indexer_events.remote())

    await async_wait_for_condition(condition, timeout=timeout, retry_interval_ms=500)


class TestDynamoKvEventPipeline:
    """End-to-end over Dynamo primitives: vLLM's production ZMQ publisher ->
    dynamo KvEventPublisher -> event plane -> KvRouter's KvEventConsumer."""

    @pytest.fixture
    def namespace(self):
        return f"test_kv_events_{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_stored_and_removed_reach_router_indexer(
        self, ray_instance, namespace
    ):
        worker_id = 7001
        actor = LocalKVRouterActor.remote(namespace)
        replica = ReplicaStandIn.remote(actor, "replica-A", worker_id, namespace, 23817)
        try:
            # Events are keyed by the Ray-supplied worker id (RFC §4.3).
            await replica.start.remote()
            assert await actor.get_kv_event_worker_replicas.remote() == {
                worker_id: "replica-A"
            }

            token_ids = list(range(2 * BLOCK_SIZE))
            await wait_for_indexer(
                actor,
                lambda events: stored_block_hashes(events, worker_id) == {11, 22},
                publish=lambda: replica.publish_stored.remote([11, 22], token_ids),
            )
            assert await actor.get_kv_event_worker_ids.remote() == [worker_id]

            # The indexer stores Dynamo's content hash per block: the dumped
            # hashes must match hashing the engine's token stream directly.
            events = await actor.get_kv_indexer_events.remote()
            assert stored_tokens_hashes(events, worker_id) == set(
                compute_block_hash_for_seq(token_ids, BLOCK_SIZE)
            )

            # Removing the leaf drops only it from the indexer's chain (vLLM
            # evicts leaf-first, and removing a parent prunes its subtree).
            await wait_for_indexer(
                actor,
                lambda events: stored_block_hashes(events, worker_id) == {11},
                publish=lambda: replica.publish_removed.remote([22]),
            )

            # AllBlocksCleared (e.g. /reset_prefix_cache) empties the view.
            await wait_for_indexer(
                actor,
                lambda events: stored_block_hashes(events, worker_id) == set(),
                publish=lambda: replica.publish_cleared.remote(),
            )
        finally:
            await replica.close.remote()
            for a in (replica, actor):
                ray.kill(a, no_restart=True)

    @pytest.mark.asyncio
    async def test_per_worker_isolation(self, ray_instance, namespace):
        """Two replicas' events land in the same indexer keyed by worker."""
        actor = LocalKVRouterActor.remote(namespace)
        worker_ids = {"replica-A": 7001, "replica-B": 7002}
        replicas = {
            replica_id: ReplicaStandIn.remote(
                actor, replica_id, worker_id, namespace, 21813 + i
            )
            for i, (replica_id, worker_id) in enumerate(worker_ids.items())
        }
        try:
            for replica in replicas.values():
                await replica.start.remote()
            assert await actor.get_kv_event_worker_replicas.remote() == {
                worker_id: replica_id for replica_id, worker_id in worker_ids.items()
            }

            blocks = {"replica-A": 100, "replica-B": 200}
            tokens = {
                "replica-A": list(range(BLOCK_SIZE)),
                "replica-B": list(range(BLOCK_SIZE, 2 * BLOCK_SIZE)),
            }
            for replica_id, replica in replicas.items():

                def consumed(events, worker_id=worker_ids[replica_id]):
                    return stored_block_hashes(events, worker_id) == {
                        blocks[replica_id]
                    }

                await wait_for_indexer(
                    actor,
                    consumed,
                    publish=lambda replica=replica, replica_id=replica_id: (
                        replica.publish_stored.remote(
                            [blocks[replica_id]], tokens[replica_id]
                        )
                    ),
                )
            assert await actor.get_kv_event_worker_ids.remote() == sorted(
                worker_ids.values()
            )

            # The router scores per-worker overlap from the consumed events:
            # each worker overlaps only the tokens its replica cached.
            for replica_id, worker_id in worker_ids.items():
                overlaps = await actor.get_kv_overlap_blocks.remote(tokens[replica_id])
                assert overlaps[worker_id] == 1
                other = (worker_ids.keys() - {replica_id}).pop()
                assert overlaps.get(worker_ids[other], 0) == 0
        finally:
            for replica in replicas.values():
                await replica.close.remote()
                ray.kill(replica, no_restart=True)
            ray.kill(actor, no_restart=True)

    @pytest.mark.asyncio
    async def test_worker_registration_purged_with_replica(
        self, ray_instance, namespace
    ):
        """A tracked replica's removal drops its KV-event registration.

        Registration precedes tracking (replicas register while STARTING),
        so only the removal of a previously tracked replica purges.
        """
        actor = LocalKVRouterActor.remote(namespace)
        replicas = {
            get_worker_id(f"u{i}"): ReplicaID(
                unique_id=f"u{i}", deployment_id=DeploymentID("d", "a")
            ).to_full_id_str()
            for i in range(2)
        }
        (keep_worker, keep_replica), (drop_worker, drop_replica) = replicas.items()
        try:
            # Replicas register before the controller reports them running.
            for worker_id, replica_id in replicas.items():
                await actor.register_kv_event_worker.remote(
                    worker_id, replica_id, BLOCK_SIZE
                )
            await actor.apply_running_replicas.remote(list(replicas.values()))
            assert await actor.get_kv_event_worker_replicas.remote() == replicas

            await actor.apply_running_replicas.remote([keep_replica])
            assert await actor.get_kv_event_worker_replicas.remote() == {
                keep_worker: keep_replica
            }
        finally:
            ray.kill(actor, no_restart=True)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
