"""The Graphiti-backed stand-in for the Zep SDK, run against a fake engine (no database, no LLM).

The real engine is verified on Colab; here the point is that everything the app does with a Zep client keeps working
through ``GraphitiZepClient``: the graph builder's Batch API flow, cursor paging, the entity reader, the episode
``processed`` flag, ontology persistence, and the ``get_zep_client`` switch.
"""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from zep_cloud import NotFoundError

from app.utils import graphiti_backend as gb
from app.utils.graphiti_backend import GraphitiSettings, GraphitiZepClient, build_graphiti_types, serialize_ontology


class FakeEngine:
    """An in-memory Graphiti: one node per episode, an edge between consecutive nodes of a graph."""

    def __init__(self, delay=0.0, fail_when=None):
        self.delay, self.fail_when = delay, fail_when
        self.graphs, self.nodes, self.edges, self.calls = set(), {}, {}, []
        self.counter = 0
        self.order = []
        self.running = 0
        self.max_running = 0

    def _id(self):
        self.counter += 1
        return f"u{self.counter:05d}"

    async def list_graphs(self):
        return sorted(self.graphs)

    async def create_graph(self, graph_id):
        self.graphs.add(graph_id)
        self.nodes.setdefault(graph_id, [])
        self.edges.setdefault(graph_id, [])

    async def delete_graph(self, graph_id):
        self.graphs.discard(graph_id)
        self.nodes.pop(graph_id, None)
        self.edges.pop(graph_id, None)

    async def add_episode(self, *, graph_id, episode_uuid, name, body, source_description, reference_time,
                          entity_types, edge_types, edge_type_map):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail_when and self.fail_when(body):
                raise ValueError("model returned garbage")
            self.calls.append(dict(graph_id=graph_id, episode_uuid=episode_uuid, body=body, reference_time=reference_time,
                                   entity_types=entity_types, edge_types=edge_types, edge_type_map=edge_type_map))
            self.order.append(body)
            label = next(iter(entity_types), None) if entity_types else None
            node = SimpleNamespace(uuid=self._id(), name=f"E:{body[:12]}", labels=["Entity"] + ([label] if label else []),
                                   summary=body, attributes={"k": "v"}, created_at=reference_time, group_id=graph_id)
            previous = self.nodes[graph_id][-1] if self.nodes.get(graph_id) else None
            self.nodes.setdefault(graph_id, []).append(node)
            if previous is not None:
                self.edges.setdefault(graph_id, []).append(SimpleNamespace(
                    uuid=self._id(), name="RELATES_TO", fact=f"{previous.summary} relates to {node.summary}",
                    source_node_uuid=previous.uuid, target_node_uuid=node.uuid, attributes={}, created_at=reference_time,
                    valid_at=reference_time, invalid_at=None, expired_at=None, episodes=[episode_uuid], group_id=graph_id))
        finally:
            self.running -= 1

    async def search(self, graph_id, query, scope, limit):
        self.calls.append(dict(search=(graph_id, query, scope, limit)))
        words = query.lower().split()
        edges = [e for e in self.edges.get(graph_id, []) if any(w in e.fact.lower() for w in words)][:limit]
        nodes = [n for n in self.nodes.get(graph_id, []) if any(w in n.summary.lower() for w in words)][:limit]
        return edges, nodes

    async def _page(self, items, limit, cursor):
        ordered = sorted(items, key=lambda i: i.uuid, reverse=True)
        if cursor:
            ordered = [i for i in ordered if i.uuid < cursor]
        return ordered[:limit]

    async def list_nodes(self, graph_id, limit, cursor):
        return await self._page(self.nodes.get(graph_id, []), limit, cursor)

    async def list_edges(self, graph_id, limit, cursor):
        return await self._page(self.edges.get(graph_id, []), limit, cursor)

    async def get_node(self, graph_id, node_uuid):
        return next((n for n in self.nodes.get(graph_id, []) if n.uuid == node_uuid), None)

    async def node_edges(self, graph_id, node_uuid):
        return [e for e in self.edges.get(graph_id, []) if node_uuid in (e.source_node_uuid, e.target_node_uuid)]

    async def close(self):
        pass


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def make_client(tmp_path):
    def make(engine=None, state_dir=None):
        engine = engine or FakeEngine()
        return GraphitiZepClient(engine, str(state_dir or tmp_path)), engine

    return make


# --- the graph and its ontology ---------------------------------------------------------------------------------

ONTOLOGY = {
    "entity_types": [
        {"name": "Company", "description": "A listed company.", "attributes": [{"name": "ticker", "description": "Exchange ticker"}]},
        {"name": "Regulator", "description": "A market regulator.", "attributes": []},
    ],
    "edge_types": [
        {"name": "regulates", "description": "Supervises.", "attributes": [],
         "source_targets": [{"source": "Regulator", "target": "Company"}]},
    ],
}


def build_ontology_with_the_real_builder(client, graph_id):
    from app.services import graph_builder as gbuilder

    class Builder(gbuilder.GraphBuilderService):
        def __init__(self):
            self.client = client

    Builder().set_ontology(graph_id, ONTOLOGY)


def test_create_get_delete_and_not_found(make_client):
    client, engine = make_client()
    with pytest.raises(NotFoundError):
        client.graph.get("g1")
    client.graph.create(graph_id="g1", name="n", description="d")
    assert client.graph.get("g1").graph_id == "g1" and "g1" in engine.graphs
    client.graph.delete(graph_id="g1")
    assert "g1" not in engine.graphs
    with pytest.raises(NotFoundError):
        client.graph.get("g1")


def test_the_apps_own_ontology_code_works_through_the_client_and_reaches_every_episode(make_client):
    client, engine = make_client()
    client.graph.create(graph_id="g1")
    build_ontology_with_the_real_builder(client, "g1")

    episode = client.graph.add(graph_id="g1", type="text", data="Persistent Systems raised capital",
                               created_at="2026-10-26T12:00:00Z", source_description="x", metadata={"source": "t"})
    assert wait_for(lambda: client.graph.episode.get(uuid_=episode.uuid_).processed)

    call = engine.calls[0]
    assert set(call["entity_types"]) == {"Company", "Regulator"}
    assert call["entity_types"]["Company"].__doc__ == "A listed company."
    assert "ticker" in call["entity_types"]["Company"].model_fields
    assert set(call["edge_types"]) == {"regulates"}
    assert call["edge_type_map"] == {("Regulator", "Company"): ["regulates"]}
    assert call["reference_time"].year == 2026 and call["reference_time"].tzinfo is not None
    assert call["episode_uuid"] == episode.uuid_                      # the app polls this very uuid


def test_the_ontology_survives_a_restart(make_client, tmp_path):
    client, _engine = make_client()
    client.graph.create(graph_id="g1")
    build_ontology_with_the_real_builder(client, "g1")

    second, engine2 = make_client(state_dir=tmp_path)                 # a new process, same state dir
    second.graph.create(graph_id="g1")
    ep = second.graph.add(graph_id="g1", type="text", data="after the restart")
    assert wait_for(lambda: second.graph.episode.get(uuid_=ep.uuid_).processed)
    assert set(engine2.calls[0]["entity_types"]) == {"Company", "Regulator"}


def test_ontology_helpers_round_trip():
    from zep_cloud.external_clients.ontology import EntityModel

    spec = serialize_ontology({"Thing": type("Thing", (EntityModel,), {"__doc__": "A thing."})}, None)
    entity_types, edge_types, edge_map = build_graphiti_types(spec)
    assert list(entity_types) == ["Thing"] and edge_types == {} and edge_map == {}
    assert build_graphiti_types({}) == ({}, {}, {})


# --- the episode queue --------------------------------------------------------------------------------------------

def test_an_episode_is_not_processed_until_the_engine_is_done(make_client):
    client, engine = make_client(FakeEngine(delay=0.3))
    client.graph.create(graph_id="g1")
    ep = client.graph.add(graph_id="g1", type="text", data="slow")
    assert client.graph.episode.get(uuid_=ep.uuid_).processed is False        # Zep's "still pending"
    assert wait_for(lambda: client.graph.episode.get(uuid_=ep.uuid_).processed)


def test_episodes_of_one_graph_run_one_at_a_time_and_in_order(make_client):
    client, engine = make_client(FakeEngine(delay=0.02))
    client.graph.create(graph_id="g1")
    eps = [client.graph.add(graph_id="g1", type="text", data=f"d{i}") for i in range(6)]
    assert wait_for(lambda: all(client.graph.episode.get(uuid_=e.uuid_).processed for e in eps))
    assert engine.order == [f"d{i}" for i in range(6)]
    assert engine.max_running == 1


def test_a_failed_episode_is_reported_not_silently_pending_forever(make_client):
    client, _ = make_client(FakeEngine(fail_when=lambda body: "bad" in body))
    client.graph.create(graph_id="g1")
    ok, bad = client.graph.add(graph_id="g1", type="text", data="fine"), client.graph.add(graph_id="g1", type="text", data="bad one")
    assert wait_for(lambda: client.graph.episode.get(uuid_=ok.uuid_).processed)
    assert wait_for(lambda: client._episodes[bad.uuid_].status == "failed")
    with pytest.raises(RuntimeError, match="garbage"):
        client.graph.episode.get(uuid_=bad.uuid_)
    after = client.graph.add(graph_id="g1", type="text", data="still works")      # the worker survived
    assert wait_for(lambda: client.graph.episode.get(uuid_=after.uuid_).processed)


def test_unknown_episode_is_not_found(make_client):
    client, _ = make_client()
    with pytest.raises(NotFoundError):
        client.graph.episode.get(uuid_="nope")


# --- Zep's Batch API, driven by the app's own graph builder ----------------------------------------------------------

def test_the_graph_builders_batch_flow_runs_unchanged_on_graphiti(make_client, monkeypatch):
    from app.services import graph_builder as gbuilder

    client, engine = make_client(FakeEngine(delay=0.01))
    client.graph.create(graph_id="g1")
    build_ontology_with_the_real_builder(client, "g1")

    class Builder(gbuilder.GraphBuilderService):
        def __init__(self):
            self.client = client

    builder = Builder()
    chunks = [f"chunk number {i} about the QIP" for i in range(7)]
    seen = []
    submission = builder.add_text_batches("g1", chunks, batch_size=3, batch_created_callback=lambda b, o: seen.append((b, o)))
    assert submission.item_count == 7 and len(submission.episode_uuids) == 7 and seen[-1][0] == submission.batch_id

    uuids = builder._wait_for_batch(submission, timeout=20)
    assert uuids == submission.episode_uuids
    builder._wait_for_episodes(uuids, timeout=20)                       # the existing "processed?" polling
    assert engine.order == chunks                                        # in submission order
    assert client.batch.get(batch_id=submission.batch_id).status == "succeeded"


def test_a_failed_chunk_makes_the_batch_partial(make_client):
    from zep_cloud import BatchAddItem

    client, _ = make_client(FakeEngine(fail_when=lambda body: body == "boom"))
    client.graph.create(graph_id="g1")
    batch = client.batch.create(metadata={"graph_id": "g1"})
    items = [BatchAddItem(type="graph_episode", graph_id="g1", data=d, data_type="text") for d in ("a", "boom", "c")]
    details = client.batch.add(batch_id=batch.batch_id, items=items)
    assert [d.sequence_index for d in details] == [0, 1, 2] and all(d.episode_uuid for d in details)
    client.batch.process(batch_id=batch.batch_id)
    assert wait_for(lambda: client.batch.get(batch_id=batch.batch_id).status in gb.BATCH_TERMINAL)
    assert client.batch.get(batch_id=batch.batch_id).status == "partial"
    statuses = {i.sequence_index: i.status for i in client.batch.list_items(batch_id=batch.batch_id).items}
    assert statuses == {0: "succeeded", 1: "failed", 2: "succeeded"}


def test_batch_rules_match_zep(make_client):
    client, _ = make_client()
    with pytest.raises(NotFoundError):
        client.batch.get(batch_id="missing")
    batch = client.batch.create(metadata={"mirofish_operation_id": "op1", "graph_id": "g1"})
    assert client.batch.get(batch_id=batch.batch_id).status == "draft"
    found = [b for b in client.batch.list(limit=100, cursor=None).batches if b.metadata.get("mirofish_operation_id") == "op1"]
    assert len(found) == 1                                              # how the builder re-finds a batch after a lost reply
    client.batch.process(batch_id=batch.batch_id)                       # an empty batch finishes at once
    assert client.batch.get(batch_id=batch.batch_id).status == "succeeded"
    with pytest.raises(RuntimeError):
        client.batch.add(batch_id=batch.batch_id, items=[])


# --- reading the graph --------------------------------------------------------------------------------------------

def fill(client, graph_id, n):
    client.graph.create(graph_id=graph_id)
    eps = [client.graph.add(graph_id=graph_id, type="text", data=f"fact number {i} about regulation") for i in range(n)]
    assert wait_for(lambda: all(client.graph.episode.get(uuid_=e.uuid_).processed for e in eps))


def test_the_apps_paging_helpers_read_every_node_and_edge(make_client):
    from app.utils.zep_paging import fetch_all_edges, fetch_all_nodes

    client, _ = make_client()
    fill(client, "g1", 7)
    nodes = fetch_all_nodes(client, "g1", page_size=3)
    edges = fetch_all_edges(client, "g1", page_size=3)
    assert len(nodes) == 7 and len({n.uuid_ for n in nodes}) == 7          # no page repeated, none skipped
    assert len(edges) == 6
    first = nodes[0]
    assert first.uuid_ == first.uuid and first.labels and first.summary and isinstance(first.attributes, dict)


def test_the_entity_reader_filters_by_the_ontology_label(make_client, monkeypatch):
    from app.services import zep_entity_reader as reader_module

    client, _ = make_client()
    client.graph.create(graph_id="g1")
    build_ontology_with_the_real_builder(client, "g1")
    eps = [client.graph.add(graph_id="g1", type="text", data=f"entity {i}") for i in range(3)]
    assert wait_for(lambda: all(client.graph.episode.get(uuid_=e.uuid_).processed for e in eps))

    monkeypatch.setattr(reader_module, "get_zep_client", lambda *_a, **_k: client)
    monkeypatch.setattr(reader_module.Config, "ZEP_API_KEY", "graphiti-local")
    reader = reader_module.ZepEntityReader()
    filtered = reader.filter_defined_entities("g1")
    # nodes carry the custom label, which is what persona generation looks for: 0 here would mean no agents at all
    assert filtered.filtered_count == 3 and filtered.entity_types == {"Company"}


def test_node_lookups_and_edges_find_the_right_graph(make_client):
    client, _ = make_client()
    fill(client, "g1", 3)
    fill(client, "g2", 2)
    node = client.graph.node.with_raw_response.get_by_graph_id("g1", limit=100).data[0]
    assert client.graph.node.get(uuid_=node.uuid_).uuid_ == node.uuid_
    assert isinstance(client.graph.node.get_edges(node_uuid=node.uuid_), list)
    with pytest.raises(NotFoundError):
        client.graph.node.get(uuid_="missing")
    with pytest.raises(NotFoundError):
        client.graph.node.get_edges(node_uuid="missing")


def test_search_returns_zep_shaped_edges_or_nodes_by_scope(make_client):
    client, engine = make_client()
    fill(client, "g1", 4)
    edges = client.graph.search(graph_id="g1", query="regulation", limit=3, scope="edges", reranker="cross_encoder")
    assert 0 < len(edges.edges) <= 3 and edges.nodes is None
    e = edges.edges[0]
    assert e.uuid_ and e.fact and e.valid_at is not None and e.invalid_at is None and e.expired_at is None
    nodes = client.graph.search(graph_id="g1", query="regulation", limit=2, scope="nodes", reranker="rrf")
    assert 0 < len(nodes.nodes) <= 2 and nodes.edges is None
    assert nodes.nodes[0].name and nodes.nodes[0].labels
    assert ("g1", "regulation", "nodes", 2) in [c["search"] for c in engine.calls if "search" in c]


# --- wiring ---------------------------------------------------------------------------------------------------------

def test_get_zep_client_returns_the_graphiti_client_only_when_asked(monkeypatch):
    from app.config import Config
    from app.utils import zep

    sentinel = object()
    monkeypatch.setattr(gb, "get_graphiti_client", lambda: sentinel)
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "graphiti")
    assert zep.get_zep_client() is sentinel                              # no Zep key, no Zep account involved
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "zep")
    monkeypatch.setattr(Config, "ZEP_API_KEY", "")
    with pytest.raises(ValueError):
        zep.get_zep_client()                                             # the default path is unchanged


def test_settings_default_to_the_apps_llm_and_a_local_embedder():
    s = GraphitiSettings.from_env({"LLM_API_KEY": "ignored-here"})
    assert s.db_host == "localhost" and s.db_port == 6379 and s.structured_output_mode == "json_object"
    assert s.embedding_base_url is None and s.local_embedding_model == "BAAI/bge-small-en-v1.5"
    s = GraphitiSettings.from_env({
        "GRAPHITI_DB_PORT": "6390", "GRAPHITI_LLM_MODEL_NAME": "m", "GRAPHITI_EMBEDDING_BASE_URL": "https://e/v1",
        "GRAPHITI_EMBEDDING_MODEL_NAME": "bge", "GRAPHITI_EMBEDDING_DIM": "768", "GRAPHITI_CONCURRENCY": "bad",
    })
    assert (s.db_port, s.llm_model, s.embedding_base_url, s.embedding_dim, s.concurrency) == (6390, "m", "https://e/v1", 768, 4)


def test_graphiti_follows_the_main_model_then_the_twitter_model(monkeypatch):
    from app.config import Config

    for name, value in dict(LLM_API_KEY="tk", LLM_BASE_URL="https://t/v1", LLM_MODEL_NAME="twitter-model",
                            LLM_PREP_API_KEY=None, LLM_PREP_BASE_URL=None, LLM_PREP_MODEL_NAME=None).items():
        monkeypatch.setattr(Config, name, value)
    s = GraphitiSettings.from_env({})
    assert (s.llm_model, s.llm_base_url, s.llm_api_key) == ("twitter-model", "https://t/v1", "tk")      # no MAIN model: Twitter's

    monkeypatch.setattr(Config, "LLM_PREP_API_KEY", "mk")
    monkeypatch.setattr(Config, "LLM_PREP_MODEL_NAME", "main-model")
    s = GraphitiSettings.from_env({})
    assert (s.llm_model, s.llm_base_url, s.llm_api_key) == ("main-model", "https://t/v1", "mk")          # MAIN's own base falls back to Twitter's

    s = GraphitiSettings.from_env({"GRAPHITI_LLM_MODEL_NAME": "extractor"})
    assert s.llm_model == "extractor"                                                                      # an explicit Graphiti model wins


def test_calls_from_many_threads_share_one_event_loop(make_client):
    client, _ = make_client(FakeEngine(delay=0.005))
    client.graph.create(graph_id="g1")
    errors, uuids = [], []

    def work(i):
        try:
            for j in range(5):
                uuids.append(client.graph.add(graph_id="g1", type="text", data=f"t{i}-{j}").uuid_)
                client.graph.search(graph_id="g1", query="t", limit=2, scope="edges")
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(set(uuids)) == 30
    assert wait_for(lambda: all(client.graph.episode.get(uuid_=u).processed for u in uuids))


# --- what a real model does to the attribute answers ------------------------------------------------------------------

def attribute_model(*names):
    return gb.build_graphiti_types({"entities": {"Company": {"doc": "A company.", "fields": {n: "" for n in names}}}})[0]["Company"]


def test_a_nested_extra_key_is_dropped_so_the_database_never_sees_it():
    # the production failure: "Property values can only be of primitive types or arrays of primitive types"
    model = attribute_model("ticker", "sector")
    answer = {"ticker": "PSYS", "sector": "IT", "notes": {"source": {"line": 3, "tags": ["a"]}}, "confidence": 0.9}
    assert gb.sanitize_attributes(answer, model) == {"ticker": "PSYS", "sector": "IT"}


def test_an_answer_that_echoes_the_schema_layout_is_unwrapped():
    model = attribute_model("ticker")
    assert gb.sanitize_attributes({"title": "Attributes", "type": "object", "properties": {"ticker": "PSYS"}}, model) == {"ticker": "PSYS"}
    # ... and a piece of the schema itself is not an answer
    assert gb.sanitize_attributes({"ticker": {"type": "string", "description": "Ticker"}}, model) == {"ticker": None}


def test_numbers_lists_and_objects_become_text_because_every_attribute_is_declared_as_text():
    model = attribute_model("a", "b", "c", "d", "e", "f")
    got = gb.sanitize_attributes({"a": 42, "b": 1.5, "c": True, "d": ["x", "y", 3], "e": [{"k": "v"}], "f": {"k": [1, 2]}}, model)
    assert got == {"a": "42", "b": "1.5", "c": "true", "d": "x, y, 3", "e": '[{"k": "v"}]', "f": '{"k": [1, 2]}'}
    assert gb.sanitize_attributes({"a": None, "b": []}, model) == {"a": None, "b": None}


def test_only_the_users_own_models_are_sanitised_not_graphitis():
    assert gb.is_custom_type(attribute_model("ticker")) is True
    internal = type("ExtractedEntities", (), {"__module__": "graphiti_core.prompts.extract_nodes"})
    assert gb.is_custom_type(internal) is False
    assert gb.sanitize_attributes("not a dict", attribute_model("ticker")) == "not a dict"          # left for Graphiti to reject
    assert gb.sanitize_attributes({"x": 1}, type("NoFields", (), {"model_fields": {}})) == {}


def test_a_chunk_that_fails_once_is_retried_and_a_chunk_that_keeps_failing_is_reported(tmp_path):
    class Flaky(FakeEngine):
        def __init__(self, failures):
            super().__init__()
            self.failures_left = failures

        async def add_episode(self, **kwargs):
            if self.failures_left:
                self.failures_left -= 1
                raise ValueError("Property values can only be of primitive types")
            await super().add_episode(**kwargs)

    for failures, retries, expected in [(1, 1, "processed"), (2, 1, "failed"), (1, 0, "failed"), (2, 2, "processed")]:
        engine = Flaky(failures)
        client = GraphitiZepClient(engine, str(tmp_path / f"s{failures}{retries}"), episode_retries=retries, retry_delay=0.01)
        client.graph.create(graph_id="g")
        episode = client.graph.add(graph_id="g", type="text", data="chunk")
        assert wait_for(lambda: client._episodes[episode.uuid_].status in ("processed", "failed")), (failures, retries)
        assert client._episodes[episode.uuid_].status == expected, (failures, retries)
        if expected == "failed":
            with pytest.raises(RuntimeError, match="primitive"):
                client.graph.episode.get(uuid_=episode.uuid_)


def test_the_retry_setting_accepts_zero():
    assert GraphitiSettings.from_env({}).episode_retries == 1
    assert GraphitiSettings.from_env({"GRAPHITI_EPISODE_RETRIES": "0"}).episode_retries == 0
    assert GraphitiSettings.from_env({"GRAPHITI_EPISODE_RETRIES": "3"}).episode_retries == 3
    assert GraphitiSettings.from_env({"GRAPHITI_EPISODE_RETRIES": "many"}).episode_retries == 1


def test_graphitis_own_answer_that_echoes_the_schema_layout_is_unwrapped():
    from pydantic import BaseModel

    class EdgeDuplicate(BaseModel):
        duplicate_facts: list
        contradicted_facts: list

    wrapped = {"properties": {"duplicate_facts": [], "contradicted_facts": []}}
    assert gb.unwrap_layout(wrapped, EdgeDuplicate) == {"duplicate_facts": [], "contradicted_facts": []}
    fine = {"duplicate_facts": [1], "contradicted_facts": []}
    assert gb.unwrap_layout(fine, EdgeDuplicate) is fine                    # a correct answer is never touched
    assert gb.unwrap_layout({"x": {"y": 1}}, EdgeDuplicate) == {"x": {"y": 1}}
