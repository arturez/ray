"""GPU end-to-end test for the Dynamo KV event pipeline.

Deploys two real vLLM replicas with KV-cache events enabled, sends prompts
directly to each replica's backend HTTP server, and asserts the engines'
events flow through each replica's Dynamo ``KvEventPublisher`` into the
``KVRouterActor``-hosted ``KvRouter``: its global indexer must attribute the
prompt's exact block chain to the right worker and score per-worker overlap.

``KVAwareRouter`` replica selection lands in a later branch, so the actor is
attached via ``deployment_actors`` and the engine KV-events configuration is
applied directly (the builder applies it when KVAwareRouter is selected,
covered by CPU tests); event bridging is router-independent.
"""

import asyncio
import os
import sys
import tempfile

import pytest
import requests

os.environ["RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING"] = "1"
os.environ["RAY_SERVE_ENABLE_DIRECT_INGRESS"] = "1"

from dynamo.llm import compute_block_hash_for_seq  # noqa: E402

import ray  # noqa: E402
from ray import serve  # noqa: E402
from ray._common.test_utils import async_wait_for_condition  # noqa: E402
from ray.llm._internal.serve.core.ingress.builder import (  # noqa: E402
    _build_direct_streaming_llm_deployment,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (  # noqa: E402
    KV_ROUTER_ACTOR_NAME,
    KVRouterActor,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (  # noqa: E402
    configure_kv_events_for_kv_routing,
)
from ray.serve._private.constants import (  # noqa: E402
    SERVE_DEPLOYMENT_ACTOR_PREFIX,
    SERVE_NAMESPACE,
)
from ray.serve.config import DeploymentActorConfig  # noqa: E402
from ray.serve.llm import LLMConfig, ModelLoadingConfig  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
APP_NAME = "kv_events_gpu_test"
NUM_REPLICAS = 2
BLOCK_SIZE = 16
MAX_TOKENS = 50
MESSAGES = [
    {
        "role": "user",
        "content": (
            "Repeat the following sentence five times: the quick brown fox "
            "jumps over the lazy dog while the cat watches from the fence."
        ),
    }
]


def discover_deployment_actor(app_name, deployment_name, actor_name):
    """Resolve a deployment-scoped actor by its registered name.

    The test driver isn't a replica, so ``get_deployment_actor`` is
    unavailable; match the name's stable prefix/suffix instead (the middle
    embeds an opaque code_version).
    """
    prefix = f"{SERVE_DEPLOYMENT_ACTOR_PREFIX}{app_name}::{deployment_name}::"
    suffix = f"::{actor_name}"
    for entry in ray.util.list_named_actors(all_namespaces=True):
        name = entry.get("name") or ""
        if (
            entry.get("namespace") == SERVE_NAMESPACE
            and name.startswith(prefix)
            and name.endswith(suffix)
        ):
            return ray.get_actor(name, namespace=SERVE_NAMESPACE)
    return None


def post_chat(endpoint, messages=MESSAGES, max_tokens=MAX_TOKENS):
    host, port = endpoint
    response = requests.post(
        f"http://{host}:{port}/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    return response.json()


def tokenize_prompt(endpoint, messages=MESSAGES):
    """The engine's exact token ids for a chat-templated prompt."""
    host, port = endpoint
    response = requests.post(
        f"http://{host}:{port}/tokenize",
        json={"model": MODEL_ID, "messages": messages, "add_generation_prompt": True},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    return response.json()["tokens"]


def prompt_block_hashes(token_ids):
    """Dynamo's content hashes for the full blocks of a token sequence."""
    return compute_block_hash_for_seq(list(token_ids), BLOCK_SIZE)


def stored_tokens_hashes(indexer_events, worker_id):
    """Dynamo per-block token hashes stored for a worker in the indexer dump."""
    return {
        block["tokens_hash"]
        for entry in indexer_events
        if entry["worker_id"] == worker_id
        for block in entry["event"]["data"]["stored"]["blocks"]
    }


class TestKvEventsGPU:
    @pytest.fixture(scope="class")
    def deployed_handle(self):
        """Deploy two direct-streaming LLMServer replicas with KV events on."""
        if not ray.is_initialized():
            # An empty working_dir keeps the runtime-env package tiny; the
            # repo root would exceed the upload size limit.
            ray.init(
                address="auto",
                runtime_env={"working_dir": tempfile.mkdtemp(prefix="kv_ev_wd_")},
            )
        serve.shutdown()  # ensure no prior app is holding GPU memory

        llm_config = LLMConfig(
            model_loading_config=ModelLoadingConfig(
                model_id=MODEL_ID,
                model_source=MODEL_ID,
            ),
            deployment_config=dict(
                autoscaling_config=dict(
                    min_replicas=NUM_REPLICAS, max_replicas=NUM_REPLICAS
                ),
                deployment_actors=[
                    DeploymentActorConfig(
                        name=KV_ROUTER_ACTOR_NAME,
                        actor_class=KVRouterActor,
                        actor_options={"num_cpus": 0},
                    )
                ],
            ),
            engine_kwargs=dict(
                block_size=BLOCK_SIZE,
                enable_prefix_caching=True,
                max_model_len=2048,
                enforce_eager=True,
                gpu_memory_utilization=0.4,
                use_tqdm_on_load=False,
            ),
            experimental_configs={"KV_EVENTS_PORT_BASE": 21557},
            placement_group_config={"bundles": [{"GPU": 1}]},
            # The replica worker process reads these constants at import time.
            runtime_env=dict(
                env_vars={
                    "VLLM_DISABLE_COMPILE_CACHE": "1",
                    "RAY_SERVE_ENABLE_DIRECT_INGRESS": "1",
                    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
                    # /reset_prefix_cache is a vLLM dev-mode endpoint.
                    "VLLM_SERVER_DEV_MODE": "1",
                },
            ),
            log_engine_metrics=False,
        )
        configure_kv_events_for_kv_routing(llm_config)

        app = _build_direct_streaming_llm_deployment(llm_config)
        handle = serve.run(app, name=APP_NAME)
        yield handle
        serve.shutdown()

    async def _discover_replicas(self, handle):
        """Map each replica's full id to its backend HTTP endpoint."""
        endpoints = {}
        for _ in range(120):
            async with handle.choose_replica() as selection:
                replica = selection._replica
                if replica.backend_http_endpoint is not None:
                    replica_id = replica.replica_id.to_full_id_str()
                    endpoints[replica_id] = replica.backend_http_endpoint
            if len(endpoints) == NUM_REPLICAS:
                return endpoints
            await asyncio.sleep(0.5)
        raise AssertionError(
            f"Expected {NUM_REPLICAS} replicas with backend endpoints, "
            f"found {len(endpoints)}."
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_kv_events_reach_router_actor(self, deployed_handle):
        actor = discover_deployment_actor(
            APP_NAME, deployed_handle.deployment_name, KV_ROUTER_ACTOR_NAME
        )
        assert actor is not None, "KV router actor was not discoverable"

        replica_endpoints = await self._discover_replicas(deployed_handle)

        # Every replica registered its Dynamo worker identity on startup.
        replica_by_worker = await actor.get_kv_event_worker_replicas.remote()
        assert sorted(replica_by_worker.values()) == sorted(replica_endpoints)
        endpoints = {
            worker_id: replica_endpoints[replica_id]
            for worker_id, replica_id in replica_by_worker.items()
        }
        worker_ids = sorted(endpoints)
        # The Ray-supplied worker ids agree with LongPoll replica tracking.
        assert await actor.get_candidate_worker_ids.remote() == worker_ids

        # The same prompt on each replica caches the same content.
        usages = {}
        for worker_id in worker_ids:
            usages[worker_id] = post_chat(endpoints[worker_id])["usage"]

        prompt_token_ids = tokenize_prompt(endpoints[worker_ids[0]])
        expected_hashes = prompt_block_hashes(prompt_token_ids)
        num_prompt_blocks = len(expected_hashes)
        assert num_prompt_blocks >= 2

        # The engines' events reach the router's global indexer: it scores
        # the full prompt overlap for both workers.
        async def both_workers_fully_overlap():
            overlaps = await actor.get_kv_overlap_blocks.remote(prompt_token_ids)
            return {
                worker_id: blocks
                for worker_id, blocks in overlaps.items()
                if blocks == num_prompt_blocks
            }.keys() == set(worker_ids)

        await async_wait_for_condition(both_workers_fully_overlap, timeout=30)
        assert await actor.get_kv_event_worker_ids.remote() == worker_ids

        # Content check: each worker's indexed chain contains exactly the
        # tokenized prompt's block hashes plus its decode blocks. All full
        # blocks of the prompt+completion stream are cached (the block that
        # fills on the final step may not be committed); the decode blocks'
        # events trail the response over the event plane, so wait for them.
        total_blocks = (len(prompt_token_ids) + MAX_TOKENS) // BLOCK_SIZE

        async def all_blocks_indexed():
            events = await actor.get_kv_indexer_events.remote()
            return all(
                len(stored_tokens_hashes(events, worker_id)) >= total_blocks - 1
                for worker_id in worker_ids
            )

        await async_wait_for_condition(all_blocks_indexed, timeout=30)
        events = await actor.get_kv_indexer_events.remote()
        for worker_id in worker_ids:
            usage = usages[worker_id]
            assert usage["prompt_tokens"] == len(prompt_token_ids)
            assert usage["completion_tokens"] == MAX_TOKENS

            hashes = stored_tokens_hashes(events, worker_id)
            assert set(expected_hashes) <= hashes
            assert len(hashes) in (total_blocks, total_blocks - 1)

        # Resetting one replica's prefix cache clears only its worker's view.
        # The engine drains queued KV events on scheduler steps, so a small
        # follow-up request also flushes the AllBlocksCleared event.
        reset_worker, untouched_worker = worker_ids
        host, port = endpoints[reset_worker]
        response = requests.post(f"http://{host}:{port}/reset_prefix_cache", timeout=60)
        assert response.status_code == 200, response.text
        flush_messages = [{"role": "user", "content": "Hi."}]
        post_chat(endpoints[reset_worker], messages=flush_messages, max_tokens=2)

        # After the clear, the reset worker holds only the flush request's
        # chain: its overlap with the original prompt falls back to the chat
        # template prefix the two prompts share.
        flush_token_ids = tokenize_prompt(endpoints[reset_worker], flush_messages)
        shared_prefix = next(
            (
                i
                for i, (a, b) in enumerate(zip(prompt_token_ids, flush_token_ids))
                if a != b
            ),
            min(len(prompt_token_ids), len(flush_token_ids)),
        )
        shared_blocks = shared_prefix // BLOCK_SIZE

        async def reset_worker_cleared():
            overlaps = await actor.get_kv_overlap_blocks.remote(prompt_token_ids)
            return overlaps.get(reset_worker) == shared_blocks

        await async_wait_for_condition(reset_worker_cleared, timeout=30)
        overlaps = await actor.get_kv_overlap_blocks.remote(prompt_token_ids)
        assert overlaps.get(untouched_worker) == num_prompt_blocks


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(address="auto")
    sys.exit(pytest.main(["-v", "-s", __file__]))
