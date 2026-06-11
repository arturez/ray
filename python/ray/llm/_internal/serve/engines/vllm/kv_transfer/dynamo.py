from ray.llm._internal.serve.engines.vllm.kv_transfer.base import (
    BaseConnectorBackend,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (
    CONSOLIDATOR_ENDPOINTS_KEY,
    resolve_consolidator_endpoints,
)


class DynamoConnectorBackend(BaseConnectorBackend):
    """Backend for Dynamo's KVBM vLLM connector.

    KVBM's connector leader reads ``additional_config["consolidator_endpoints"]``
    to start its KV event consolidator: it subscribes to the engine's KV
    events and republishes the consolidated stream (engine events bridged
    with KVBM's own tier events) on the output endpoint, which the replica's
    KvEventPublisher then consumes.
    """

    def setup(self) -> None:
        engine_kwargs = self.llm_config.engine_kwargs
        additional_config = dict(engine_kwargs.get("additional_config") or {})
        if CONSOLIDATOR_ENDPOINTS_KEY in additional_config:
            return
        additional_config[CONSOLIDATOR_ENDPOINTS_KEY] = resolve_consolidator_endpoints(
            self.llm_config
        )
        self.llm_config.update_engine_kwargs(additional_config=additional_config)
