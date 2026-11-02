"""Checks the Graphiti backend with YOUR model and embedder before a long run: one tiny graph, one episode, one search.

Run with the backend's environment (the notebook does):  python backend/scripts/graphiti_smoke_test.py
It prints one line per step and, for a failing step, what to change. It needs the FalkorDB server running
(``notebooks/falkordb_server.py``) and the GRAPHITI_* / LLM_* settings in ``.env``.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("GRAPH_BACKEND", "graphiti")

from app.config import Config  # noqa: E402
from app.utils import graphiti_backend as gb  # noqa: E402

HINTS = {
    "401": "the API key was rejected: check the key (Secrets) and that LLM_BASE_URL belongs to that provider",
    "403": "the provider refused this client or model: pick another model or provider",
    "404": "the model name or base URL is wrong for this provider",
    "429": "rate limited: use a model with more quota, or lower GRAPHITI_CONCURRENCY",
    "json": "the model did not return valid JSON: use a stronger/instruct model for GRAPHITI_LLM_MODEL_NAME",
    "schema": "the model rejects JSON-schema output: keep GRAPHITI_STRUCTURED_OUTPUT=json_object",
    "embedding": "the embedding endpoint failed: check GRAPHITI_EMBEDDING_* (or unset them to embed locally)",
}


def hint(error: BaseException) -> str:
    text = f"{type(error).__name__} {error}".lower()
    for key in ("401", "403", "404", "429"):
        if key in text:
            return HINTS[key]
    if "embed" in text:
        return HINTS["embedding"]
    if "json" in text or "validation" in text:
        return HINTS["json"]
    return ""


def main() -> int:
    settings = gb.GraphitiSettings.from_env()
    print(f"LLM       : {settings.llm_model} @ {settings.llm_base_url} (structured output: {settings.structured_output_mode})")
    print("Embedder  :", f"{settings.embedding_model} @ {settings.embedding_base_url}" if settings.embedding_base_url
          else f"local {settings.local_embedding_model}")
    print(f"FalkorDB  : {settings.db_host}:{settings.db_port}\n")
    failed = False
    client = None

    def step(name, fn):
        nonlocal failed
        started = time.time()
        try:
            result = fn()
            print(f"[ ok ] {name} ({time.time() - started:.1f}s){': ' + str(result) if result is not None else ''}", flush=True)
            return result
        except Exception as error:  # noqa: BLE001
            failed = True
            print(f"[FAIL] {name}: {type(error).__name__}: {str(error)[:300]}", flush=True)
            if hint(error):
                print(f"       -> {hint(error)}", flush=True)
            traceback.print_exc(limit=2)
            return None

    client = step("connect to FalkorDB and set up Graphiti", gb.get_graphiti_client)
    if client is None:
        return 1
    graph_id = f"smoketest_{int(time.time())}"
    step("create a graph", lambda: client.graph.create(graph_id=graph_id).graph_id)
    episode = step("queue one episode (about 4 model calls)", lambda: client.graph.add(
        graph_id=graph_id, type="text", data="Priya Nair, the chief analyst at Meridian Securities, said on Monday that "
        "Meridian Securities would downgrade Halcyon Pharma after the regulator opened an inquiry into Halcyon Pharma.",
        source_description="smoke test"))

    def wait():
        for _ in range(240):
            if client.graph.episode.get(uuid_=episode.uuid_).processed:
                return "extracted"
            time.sleep(1)
        raise TimeoutError("the episode was not processed within 4 minutes")

    if episode is not None:
        step("Graphiti extracts entities and facts with your model", wait)
        nodes = step("read the nodes back", lambda: [n.name for n in client.graph.node.with_raw_response.get_by_graph_id(graph_id, limit=50).data])
        if nodes is not None and not nodes:
            failed = True
            print("[FAIL] the model extracted no entities: use a stronger model for GRAPHITI_LLM_MODEL_NAME")
        step("search the facts", lambda: [e.fact[:70] for e in client.graph.search(graph_id=graph_id, query="who downgrades Halcyon", limit=3, scope="edges").edges])
    step("delete the graph", lambda: client.graph.delete(graph_id=graph_id))
    gb.reset_graphiti_client()   # stops the background workers cleanly
    print("\nSMOKE TEST " + ("FAILED: fix the lines marked FAIL before a long run" if failed else "PASSED: your model and embedder work with Graphiti"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
