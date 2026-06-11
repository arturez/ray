import sys

import pytest
from vllm.distributed.kv_events import ZmqEventPublisher

from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.core.server.builder import build_llm_deployment
from ray.llm._internal.serve.engines.vllm.kv_transfer.factory import (
    KVConnectorBackendFactory,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (
    DYNAMO_KV_CONNECTOR,
    DYNAMO_KV_CONNECTOR_MODULE_PATH,
    assign_replica_kv_events_endpoint,
    configure_kv_events_for_kv_routing,
    resolve_consolidator_endpoints,
    resolve_kv_event_source_endpoint,
)
from ray.serve.llm.request_router import KVAwareRouter


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


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
