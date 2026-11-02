"""Graphiti + FalkorDB behind the Zep Cloud SDK calls MiroFish makes.

``GRAPH_BACKEND=graphiti`` makes ``get_zep_client()`` return a ``GraphitiZepClient`` instead of the Zep SDK client. The
rest of the app keeps calling ``client.graph.add(...)``, ``client.batch.process(...)``, ``client.graph.search(...)`` and
so on, and gets back objects with the attribute names Zep uses (``uuid_``, ``fact``, ``valid_at``, ``labels`` ...).
Zep stays the default and its code path is untouched.

Two layers:

* ``GraphitiZepClient`` holds everything that imitates Zep: the ontology store, the episode queue and its
  ``processed`` flag, Zep's Batch API, cursor paging, Zep-shaped result objects, ``NotFoundError``. It needs no
  database and is unit-tested against a fake engine.
* ``GraphitiEngine`` is the thin layer over the real ``graphiti-core`` and a FalkorDB server. It is verified on Colab.

One Graphiti ``group_id`` (= one MiroFish ``graph_id``) is one FalkorDB graph, so graphs are isolated and a delete is
one ``GRAPH.DELETE``.

Behavioural differences from Zep Cloud that matter:

* Graphiti finishes an episode inside ``add_episode`` (several LLM calls), so ingestion cost is paid in your LLM
  quota. Episodes of one graph run one after another, in order, on a background worker; ``episode.get`` reports
  ``processed`` once the worker is done, so the app's existing waiting logic works unchanged.
* Search always uses Graphiti's BM25 + cosine + RRF recipes; Zep's ``cross_encoder`` reranker is mapped to RRF.
* The ontology is not stored server-side: it is kept in ``GRAPHITI_STATE_DIR`` and passed on every episode.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid as uuidlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from zep_cloud import NotFoundError

from .logger import get_logger

logger = get_logger("mirofish.graphiti")

NEXT_CURSOR_HEADER = "zep-next-cursor"
BATCH_TERMINAL = {"succeeded", "partial", "failed", "invalid", "canceled"}


# --------------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------------

@dataclass
class GraphitiSettings:
    db_host: str = "localhost"
    db_port: int = 6379
    db_username: Optional[str] = None
    db_password: Optional[str] = None
    llm_api_key: str = ""
    llm_base_url: Optional[str] = None
    llm_model: str = ""
    llm_small_model: Optional[str] = None
    # 'json_object' works with far more providers than OpenAI's strict 'json_schema'
    structured_output_mode: str = "json_object"
    llm_max_tokens: int = 8192
    embedding_api_key: str = ""
    embedding_base_url: Optional[str] = None
    embedding_model: str = ""
    embedding_dim: int = 0
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    concurrency: int = 4
    episode_retries: int = 1
    state_dir: str = ""

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "GraphitiSettings":
        env = os.environ if env is None else env
        from ..config import Config

        # Reading the seed needs the most reliable JSON model, so by default Graphiti follows the MAIN model (the one
        # that prepares the simulation and writes the report; MAIN_LLM_* / LLM_PREP_*), which is the main LLM_* model
        # when no separate one is set.
        prep = Config.prep_llm()

        def text(name: str, default: Optional[str] = None) -> Optional[str]:
            value = (env.get(name) or "").strip()
            return value or default

        def number(name: str, default: int) -> int:
            try:
                value = int(float(env.get(name) or 0))
            except ValueError:
                value = 0
            return value if value > 0 else default

        retries = (env.get("GRAPHITI_EPISODE_RETRIES") or "").strip()      # 0 is a valid choice: no retry

        return cls(
            db_host=text("GRAPHITI_DB_HOST", "localhost"),
            db_port=number("GRAPHITI_DB_PORT", 6379),
            db_username=text("GRAPHITI_DB_USERNAME"),
            db_password=text("GRAPHITI_DB_PASSWORD"),
            llm_api_key=text("GRAPHITI_LLM_API_KEY", prep["api_key"] or "") or "",
            llm_base_url=text("GRAPHITI_LLM_BASE_URL", prep["base_url"]),
            llm_model=text("GRAPHITI_LLM_MODEL_NAME", prep["model_name"] or "") or "",
            llm_small_model=text("GRAPHITI_LLM_SMALL_MODEL_NAME"),
            structured_output_mode=text("GRAPHITI_STRUCTURED_OUTPUT", "json_object") or "json_object",
            llm_max_tokens=number("GRAPHITI_LLM_MAX_TOKENS", 8192),
            embedding_api_key=text("GRAPHITI_EMBEDDING_API_KEY", "") or "",
            embedding_base_url=text("GRAPHITI_EMBEDDING_BASE_URL"),
            embedding_model=text("GRAPHITI_EMBEDDING_MODEL_NAME", "") or "",
            embedding_dim=number("GRAPHITI_EMBEDDING_DIM", 0),
            local_embedding_model=text("GRAPHITI_LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            concurrency=number("GRAPHITI_CONCURRENCY", 4),
            episode_retries=int(retries) if retries.isdigit() else 1,
            state_dir=text("GRAPHITI_STATE_DIR", os.path.join(Config.UPLOAD_FOLDER, "graphiti"))
            or os.path.join(Config.UPLOAD_FOLDER, "graphiti"),
        )


# --------------------------------------------------------------------------------------------------
# one long-lived event loop on a daemon thread
# --------------------------------------------------------------------------------------------------

class _LoopThread:
    """Graphiti and its database connection are async and tied to the loop that opened them. Flask request
    threads and the activity-writer thread all call in; this runs everything on one loop."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="graphiti-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coroutine: Awaitable[Any], timeout: Optional[float] = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def submit(self, coroutine: Awaitable[Any]):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)


# --------------------------------------------------------------------------------------------------
# ontology: Zep's EntityModel / EdgeModel classes -> plain pydantic models Graphiti accepts
# --------------------------------------------------------------------------------------------------

def serialize_ontology(entities: Optional[dict], edges: Optional[dict]) -> Dict[str, Any]:
    """The part of a Zep ontology Graphiti needs, as JSON-able data (so it survives a restart)."""

    def fields_of(model: Any) -> Dict[str, str]:
        return {
            name: (field.description or "")
            for name, field in getattr(model, "model_fields", {}).items()
        }

    spec: Dict[str, Any] = {"entities": {}, "edges": {}}
    for name, model in (entities or {}).items():
        spec["entities"][name] = {"doc": (model.__doc__ or "").strip(), "fields": fields_of(model)}
    for name, definition in (edges or {}).items():
        model, pairs = definition if isinstance(definition, (tuple, list)) else (definition, [])
        spec["edges"][name] = {
            "doc": (model.__doc__ or "").strip(),
            "fields": fields_of(model),
            "pairs": [[str(getattr(p, "source", None) or "Entity"), str(getattr(p, "target", None) or "Entity")] for p in pairs or []],
        }
    return spec


def build_graphiti_types(spec: Dict[str, Any]) -> Tuple[dict, dict, dict]:
    """``(entity_types, edge_types, edge_type_map)`` for ``Graphiti.add_episode``."""
    from pydantic import Field, create_model

    def model(name: str, definition: Dict[str, Any]):
        fields = {
            field: (Optional[str], Field(default=None, description=description))
            for field, description in (definition.get("fields") or {}).items()
        }
        return create_model(name, __module__=__name__, __doc__=definition.get("doc") or name, **fields)

    entity_types = {name: model(name, d) for name, d in (spec.get("entities") or {}).items()}
    edge_types = {name: model(name, d) for name, d in (spec.get("edges") or {}).items()}
    edge_type_map: Dict[Tuple[str, str], List[str]] = {}
    for name, d in (spec.get("edges") or {}).items():
        for source, target in d.get("pairs") or []:
            edge_type_map.setdefault((source, target), []).append(name)
    return entity_types, edge_types, edge_type_map


# --------------------------------------------------------------------------------------------------
# what a real model does to the attribute answers
# --------------------------------------------------------------------------------------------------

_INTERNAL_MODEL_PREFIX = "graphiti_core"


def is_custom_type(model: Any) -> bool:
    """True for a model that comes from the user's ontology (entity/edge attributes), not one of Graphiti's own."""
    return not str(getattr(model, "__module__", "")).startswith(_INTERNAL_MODEL_PREFIX)


def _as_text(value: Any) -> Optional[str]:
    """One attribute value as plain text (every ontology attribute is declared as text)."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        if all(v is None or isinstance(v, (str, int, float, bool)) for v in value):
            parts = [_as_text(v) for v in value if v is not None]
            return ", ".join(parts) if parts else None
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, dict):
        if "type" in value and {"description", "title", "anyOf"} & set(value):
            return None                       # a piece of the schema the model echoed, not an answer
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def sanitize_attributes(response: Any, model: Any) -> Any:
    """The model's answer for an entity's or edge's attributes, reduced to what the ontology declares, as text.

    Graphiti keeps *every* key of this answer (not just the declared fields) and checks only the declared ones, so a
    model that adds a key of its own, answers with a nested object, or echoes the schema layout sends a value to the
    database that FalkorDB refuses ("Property values can only be of primitive types or arrays of primitive types"),
    and a number or list where text is declared fails validation. Here: an answer wrapped in one level
    (``{"properties": {...}}``) is unwrapped, undeclared keys are dropped, and every value becomes text.
    """
    if not isinstance(response, dict):
        return response
    fields = set(getattr(model, "model_fields", {}) or {})
    if not fields:
        return {}
    if not fields & set(response):
        wrapped = [v for v in response.values() if isinstance(v, dict) and fields & set(v)]
        if len(wrapped) == 1:
            response = wrapped[0]
    return {key: _as_text(value) for key, value in response.items() if key in fields}


# --------------------------------------------------------------------------------------------------
# Zep-shaped result objects
# --------------------------------------------------------------------------------------------------

def zep_node(node: Any) -> SimpleNamespace:
    uuid = str(getattr(node, "uuid", "") or "")
    return SimpleNamespace(
        uuid_=uuid,
        uuid=uuid,
        name=getattr(node, "name", "") or "",
        labels=list(getattr(node, "labels", None) or []),
        summary=getattr(node, "summary", "") or "",
        attributes=dict(getattr(node, "attributes", None) or {}),
        created_at=getattr(node, "created_at", None),
        graph_id=getattr(node, "group_id", None),
    )


def zep_edge(edge: Any) -> SimpleNamespace:
    uuid = str(getattr(edge, "uuid", "") or "")
    return SimpleNamespace(
        uuid_=uuid,
        uuid=uuid,
        name=getattr(edge, "name", "") or "",
        fact=getattr(edge, "fact", "") or "",
        source_node_uuid=getattr(edge, "source_node_uuid", ""),
        target_node_uuid=getattr(edge, "target_node_uuid", ""),
        attributes=dict(getattr(edge, "attributes", None) or {}),
        created_at=getattr(edge, "created_at", None),
        valid_at=getattr(edge, "valid_at", None),
        invalid_at=getattr(edge, "invalid_at", None),
        expired_at=getattr(edge, "expired_at", None),
        episodes=list(getattr(edge, "episodes", None) or []),
        graph_id=getattr(edge, "group_id", None),
    )


def _not_found(what: str) -> NotFoundError:
    return NotFoundError(body={"message": f"{what} not found"})


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------------------
# the Zep-imitating client
# --------------------------------------------------------------------------------------------------

@dataclass
class _Episode:
    uuid: str
    graph_id: str
    status: str = "queued"            # queued -> processing -> processed | failed
    error: Optional[str] = None


@dataclass
class _Item:
    sequence_index: int
    episode_uuid: str
    graph_id: str
    data: str
    source_description: str
    metadata: Dict[str, Any]
    status: str = "pending"          # pending -> succeeded | failed
    error: Optional[str] = None


@dataclass
class _Batch:
    batch_id: str
    metadata: Dict[str, Any]
    status: str = "draft"
    items: List[_Item] = field(default_factory=list)


class GraphitiZepClient:
    """What the app sees as ``Zep``: ``client.graph``, ``client.batch``."""

    def __init__(self, engine: Any, state_dir: str, loop: Optional[_LoopThread] = None, episode_retries: int = 1,
                 retry_delay: float = 3.0) -> None:
        self._engine = engine
        self._episode_retries, self._retry_delay = max(0, int(episode_retries)), retry_delay
        self._loop = loop or _LoopThread()
        self._state_dir = Path(state_dir)
        self._lock = threading.RLock()
        self._episodes: Dict[str, _Episode] = {}
        self._batches: Dict[str, _Batch] = {}
        self._queues: Dict[str, asyncio.Queue] = {}
        self._workers: List[Any] = []
        self._ontologies: Dict[str, Dict[str, Any]] = {}
        self._known_graphs: List[str] = []
        self.graph = _GraphApi(self)
        self.batch = _BatchApi(self)

    # -- helpers ---------------------------------------------------------------------------------

    def _call(self, coroutine: Awaitable[Any], timeout: Optional[float] = None) -> Any:
        return self._loop.run(coroutine, timeout)

    def _remember_graph(self, graph_id: str) -> None:
        with self._lock:
            if graph_id in self._known_graphs:
                self._known_graphs.remove(graph_id)
            self._known_graphs.insert(0, graph_id)

    def _forget_graph(self, graph_id: str) -> None:
        with self._lock:
            if graph_id in self._known_graphs:
                self._known_graphs.remove(graph_id)
            self._ontologies.pop(graph_id, None)
            self._queues.pop(graph_id, None)
        try:
            (self._state_dir / f"{graph_id}.ontology.json").unlink()
        except OSError:
            pass

    def _ontology_spec(self, graph_id: str) -> Dict[str, Any]:
        with self._lock:
            spec = self._ontologies.get(graph_id)
        if spec is not None:
            return spec
        try:
            spec = json.loads((self._state_dir / f"{graph_id}.ontology.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            spec = {"entities": {}, "edges": {}}
        with self._lock:
            self._ontologies[graph_id] = spec
        return spec

    def _graphs_to_search(self) -> List[str]:
        with self._lock:
            known = list(self._known_graphs)
        if known:
            return known
        try:
            return list(self._call(self._engine.list_graphs(), 30))
        except Exception:
            return []

    # -- episode queue -------------------------------------------------------------------------------

    def _enqueue(self, graph_id: str, job: Callable[[], Awaitable[None]]) -> None:
        """Run ``job`` after the earlier jobs of the same graph (Graphiti dedupes against what came before)."""

        async def put() -> None:
            queue = self._queues.get(graph_id)
            if queue is None:
                queue = self._queues[graph_id] = asyncio.Queue()
                self._workers.append(asyncio.ensure_future(self._worker(queue)))
            await queue.put(job)

        self._loop.submit(put())

    async def _worker(self, queue: "asyncio.Queue") -> None:
        while True:
            job = await queue.get()
            try:
                await job()
            except Exception:  # a job records its own failure; the worker must keep going
                logger.exception("graphiti episode job crashed")
            finally:
                queue.task_done()

    def _episode_job(
        self,
        episode: _Episode,
        *,
        name: str,
        body: str,
        source_description: str,
        reference_time: datetime,
        on_done: Optional[Callable[[_Episode], None]] = None,
    ) -> Callable[[], Awaitable[None]]:
        async def job() -> None:
            episode.status = "processing"
            # A model's answer differs from one call to the next, so a chunk that failed once often works the second
            # time; one lost chunk should not cost the whole graph build.
            for attempt in range(self._episode_retries + 1):
                try:
                    entity_types, edge_types, edge_type_map = build_graphiti_types(self._ontology_spec(episode.graph_id))
                    await self._engine.add_episode(
                        graph_id=episode.graph_id,
                        episode_uuid=episode.uuid,
                        name=name,
                        body=body,
                        source_description=source_description,
                        reference_time=reference_time,
                        entity_types=entity_types or None,
                        edge_types=edge_types or None,
                        edge_type_map=edge_type_map or None,
                    )
                    episode.status, episode.error = "processed", None
                    break
                except Exception as error:  # noqa: BLE001 - recorded, surfaced by episode.get
                    episode.status = "failed"
                    episode.error = f"{type(error).__name__}: {error}"[:500]
                    if attempt < self._episode_retries:
                        logger.warning(f"Graphiti episode {episode.uuid} failed ({episode.error}); retrying")
                        episode.status = "processing"
                        await asyncio.sleep(self._retry_delay)
                    else:
                        logger.error(f"Graphiti episode {episode.uuid} failed: {episode.error}")
            if on_done is not None:
                on_done(episode)

        return job

    # -- lifecycle -----------------------------------------------------------------------------------

    def close(self) -> None:
        async def stop_workers() -> None:
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)

        try:
            self._call(stop_workers(), 10)
            self._call(self._engine.close(), 30)
        except Exception:
            pass


class _GraphApi:
    def __init__(self, client: GraphitiZepClient) -> None:
        self._c = client
        self.node = _NodeApi(client)
        self.edge = _EdgeApi(client)
        self.episode = _EpisodeApi(client)

    def create(self, graph_id: str, name: Optional[str] = None, description: Optional[str] = None, **_: Any) -> SimpleNamespace:
        self._c._call(self._c._engine.create_graph(graph_id), 120)
        self._c._remember_graph(graph_id)
        return SimpleNamespace(graph_id=graph_id, name=name, description=description)

    def get(self, graph_id: str, **_: Any) -> SimpleNamespace:
        exists = graph_id in self._c._known_graphs or graph_id in self._c._call(self._c._engine.list_graphs(), 30)
        if not exists:
            raise _not_found(f"graph {graph_id}")
        self._c._remember_graph(graph_id)
        return SimpleNamespace(graph_id=graph_id)

    def delete(self, graph_id: str, **_: Any) -> None:
        self._c._call(self._c._engine.delete_graph(graph_id), 120)
        self._c._forget_graph(graph_id)

    def set_ontology(self, graph_ids: Optional[List[str]] = None, entities: Optional[dict] = None, edges: Optional[dict] = None, **_: Any) -> None:
        spec = serialize_ontology(entities, edges)
        self._c._state_dir.mkdir(parents=True, exist_ok=True)
        for graph_id in graph_ids or []:
            (self._c._state_dir / f"{graph_id}.ontology.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            with self._c._lock:
                self._c._ontologies[graph_id] = spec

    def add(
        self,
        graph_id: str,
        type: str = "text",
        data: str = "",
        created_at: Optional[str] = None,
        source_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> SimpleNamespace:
        episode = _Episode(uuid=str(uuidlib.uuid4()), graph_id=graph_id)
        with self._c._lock:
            self._c._episodes[episode.uuid] = episode
        meta = metadata or {}
        name = f"{meta.get('source', 'episode')}-{meta.get('platform', '')}-{episode.uuid[:8]}"
        self._c._enqueue(graph_id, self._c._episode_job(
            episode, name=name, body=data, source_description=source_description or "MiroFish",
            reference_time=_parse_time(created_at),
        ))
        return SimpleNamespace(uuid_=episode.uuid, processed=False)

    def search(
        self,
        graph_id: Optional[str] = None,
        query: str = "",
        limit: int = 10,
        scope: str = "edges",
        reranker: Optional[str] = None,
        **_: Any,
    ) -> SimpleNamespace:
        edges, nodes = self._c._call(self._c._engine.search(graph_id, query, scope or "edges", int(limit)), 180)
        return SimpleNamespace(
            edges=[zep_edge(e) for e in edges] if scope in (None, "edges") else None,
            nodes=[zep_node(n) for n in nodes] if scope == "nodes" else None,
        )


class _EpisodeApi:
    def __init__(self, client: GraphitiZepClient) -> None:
        self._c = client

    def get(self, uuid_: str, **_: Any) -> SimpleNamespace:
        with self._c._lock:
            episode = self._c._episodes.get(uuid_)
        if episode is None:
            raise _not_found(f"episode {uuid_}")
        if episode.status == "failed":
            raise RuntimeError(f"Graphiti could not ingest episode {uuid_}: {episode.error}")
        return SimpleNamespace(uuid_=uuid_, processed=episode.status == "processed")


class _RawPage:
    """What ``with_raw_response`` returns in the Zep SDK: ``.data`` and ``.headers``."""

    def __init__(self, data: list, next_cursor: Optional[str]) -> None:
        self.data = data
        self.headers = {NEXT_CURSOR_HEADER: next_cursor} if next_cursor else {}


class _Paged:
    def __init__(self, fetch: Callable[..., _RawPage]) -> None:
        self._fetch = fetch

    def get_by_graph_id(self, graph_id: str, limit: int = 100, cursor: Optional[str] = None, **_: Any) -> _RawPage:
        return self._fetch(graph_id, limit, cursor)


class _NodeApi:
    def __init__(self, client: GraphitiZepClient) -> None:
        self._c = client
        self.with_raw_response = _Paged(self._page)

    def _page(self, graph_id: str, limit: int, cursor: Optional[str]) -> _RawPage:
        nodes = self._c._call(self._c._engine.list_nodes(graph_id, limit, cursor), 120)
        next_cursor = str(nodes[-1].uuid) if len(nodes) >= limit and nodes else None
        return _RawPage([zep_node(n) for n in nodes], next_cursor)

    def get_by_graph_id(self, graph_id: str, limit: int = 100, cursor: Optional[str] = None, **_: Any) -> List[SimpleNamespace]:
        return self._page(graph_id, limit, cursor).data

    def get(self, uuid_: str, **_: Any) -> SimpleNamespace:
        for graph_id in self._c._graphs_to_search():
            node = self._c._call(self._c._engine.get_node(graph_id, uuid_), 60)
            if node is not None:
                return zep_node(node)
        raise _not_found(f"node {uuid_}")

    def get_edges(self, node_uuid: str, **_: Any) -> List[SimpleNamespace]:
        for graph_id in self._c._graphs_to_search():
            if self._c._call(self._c._engine.get_node(graph_id, node_uuid), 60) is not None:
                return [zep_edge(e) for e in self._c._call(self._c._engine.node_edges(graph_id, node_uuid), 120)]
        raise _not_found(f"node {node_uuid}")


class _EdgeApi:
    def __init__(self, client: GraphitiZepClient) -> None:
        self._c = client
        self.with_raw_response = _Paged(self._page)

    def _page(self, graph_id: str, limit: int, cursor: Optional[str]) -> _RawPage:
        edges = self._c._call(self._c._engine.list_edges(graph_id, limit, cursor), 120)
        next_cursor = str(edges[-1].uuid) if len(edges) >= limit and edges else None
        return _RawPage([zep_edge(e) for e in edges], next_cursor)

    def get_by_graph_id(self, graph_id: str, limit: int = 100, cursor: Optional[str] = None, **_: Any) -> List[SimpleNamespace]:
        return self._page(graph_id, limit, cursor).data


class _BatchApi:
    """Zep's Batch API (create, add, process, get, list, list_items) over the same episode queue."""

    def __init__(self, client: GraphitiZepClient) -> None:
        self._c = client

    def create(self, metadata: Optional[Dict[str, Any]] = None, **_: Any) -> SimpleNamespace:
        batch = _Batch(batch_id=str(uuidlib.uuid4()), metadata=dict(metadata or {}))
        with self._c._lock:
            self._c._batches[batch.batch_id] = batch
        return self._summary(batch)

    def _batch(self, batch_id: str) -> _Batch:
        with self._c._lock:
            batch = self._c._batches.get(batch_id)
        if batch is None:
            raise _not_found(f"batch {batch_id}")
        return batch

    @staticmethod
    def _item(item: _Item) -> SimpleNamespace:
        return SimpleNamespace(
            sequence_index=item.sequence_index,
            episode_uuid=item.episode_uuid,
            source_uuid=item.episode_uuid,
            status=item.status,
            error=item.error,
        )

    def _summary(self, batch: _Batch) -> SimpleNamespace:
        total = len(batch.items)
        done = sum(1 for i in batch.items if i.status == "succeeded")
        failed = sum(1 for i in batch.items if i.status == "failed")
        return SimpleNamespace(
            batch_id=batch.batch_id,
            status=batch.status,
            metadata=batch.metadata,
            progress=SimpleNamespace(
                percent_complete=(100.0 * (done + failed) / total) if total else 0.0,
                succeeded_items=done,
                failed_items=failed,
                total_items=total,
            ),
        )

    def add(self, batch_id: str, items: List[Any], **_: Any) -> List[SimpleNamespace]:
        batch = self._batch(batch_id)
        if batch.status != "draft":
            raise RuntimeError(f"batch {batch_id} is {batch.status}; items can only be added to a draft")
        added: List[_Item] = []
        with self._c._lock:
            for raw in items:
                item = _Item(
                    sequence_index=len(batch.items),
                    episode_uuid=str(uuidlib.uuid4()),
                    graph_id=getattr(raw, "graph_id", None),
                    data=getattr(raw, "data", "") or "",
                    source_description=getattr(raw, "source_description", None) or "MiroFish",
                    metadata=dict(getattr(raw, "metadata", None) or {}),
                )
                batch.items.append(item)
                added.append(item)
        return [self._item(i) for i in added]

    def process(self, batch_id: str, **_: Any) -> SimpleNamespace:
        batch = self._batch(batch_id)
        if batch.status != "draft":
            return self._summary(batch)
        batch.status = "processing"
        remaining = {"n": len(batch.items)}

        def finish(episode: _Episode, item: _Item) -> None:
            item.status = "succeeded" if episode.status == "processed" else "failed"
            item.error = episode.error
            with self._c._lock:
                remaining["n"] -= 1
                if remaining["n"] <= 0:
                    failed = sum(1 for i in batch.items if i.status == "failed")
                    batch.status = "succeeded" if not failed else ("failed" if failed == len(batch.items) else "partial")

        if not batch.items:
            batch.status = "succeeded"
        for item in batch.items:
            episode = _Episode(uuid=item.episode_uuid, graph_id=item.graph_id)
            with self._c._lock:
                self._c._episodes[episode.uuid] = episode
            self._c._enqueue(item.graph_id, self._c._episode_job(
                episode,
                name=f"chunk-{item.metadata.get('chunk_index', item.sequence_index)}-{episode.uuid[:8]}",
                body=item.data,
                source_description=item.source_description,
                reference_time=datetime.now(timezone.utc),
                on_done=lambda ep, it=item: finish(ep, it),
            ))
        return self._summary(batch)

    def get(self, batch_id: str, **_: Any) -> SimpleNamespace:
        return self._summary(self._batch(batch_id))

    def list(self, limit: int = 100, cursor: Optional[int] = None, **_: Any) -> SimpleNamespace:
        with self._c._lock:
            batches = [self._summary(b) for b in self._c._batches.values()]
        return SimpleNamespace(batches=batches, next_cursor=None)

    def list_items(self, batch_id: str, limit: int = 100, cursor: Optional[int] = None, **_: Any) -> SimpleNamespace:
        batch = self._batch(batch_id)
        return SimpleNamespace(items=[self._item(i) for i in batch.items], next_cursor=None)


# --------------------------------------------------------------------------------------------------
# the real engine: graphiti-core + FalkorDB
# --------------------------------------------------------------------------------------------------

class GraphitiEngine:
    """Thin async layer over ``graphiti-core``. Every method runs on the client's event-loop thread."""

    def __init__(self, settings: GraphitiSettings) -> None:
        self.settings = settings
        self.graphiti: Any = None
        self.driver: Any = None
        self._graph_driver_cache: Dict[str, Any] = {}

    def _load_local_model(self) -> Any:
        """The local embedding model, loaded before graphiti-core is imported: Graphiti reads EMBEDDING_DIM from the
        environment once, at import time, so the model's real vector size has to be known first."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.settings.local_embedding_model, device="cpu")
        os.environ["EMBEDDING_DIM"] = str(model.get_sentence_embedding_dimension())
        return model

    async def start(self) -> None:
        s = self.settings
        local_model = None
        if s.embedding_base_url and s.embedding_model:
            os.environ["EMBEDDING_DIM"] = str(s.embedding_dim or 768)
        else:
            local_model = await asyncio.to_thread(self._load_local_model)

        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient as _GenericClient

        class OpenAIGenericClient(_GenericClient):
            """Graphiti's OpenAI-compatible client, with the attribute answers made safe to store (see sanitize_attributes)."""

            async def generate_response(self, messages, response_model=None, *args, **kwargs):
                result = await super().generate_response(messages, response_model, *args, **kwargs)
                if response_model is not None and is_custom_type(response_model):
                    return sanitize_attributes(result, response_model)
                return result

        os.environ.setdefault("SEMAPHORE_LIMIT", str(s.concurrency))

        class OrderPreservingReranker(CrossEncoderClient):
            """Keeps the order the fused (RRF) ranking already gave: no second model call."""

            async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
                n = max(len(passages), 1)
                return [(p, 1.0 - i / (n + 1)) for i, p in enumerate(passages)]

        self.driver = FalkorDriver(host=s.db_host, port=s.db_port, username=s.db_username, password=s.db_password)
        llm = OpenAIGenericClient(
            config=LLMConfig(
                api_key=s.llm_api_key or "none",
                base_url=s.llm_base_url,
                model=s.llm_model,
                small_model=s.llm_small_model or s.llm_model,
                temperature=0.0,
                max_tokens=s.llm_max_tokens,
            ),
            max_tokens=s.llm_max_tokens,
            structured_output_mode=s.structured_output_mode,
        )
        self.graphiti = Graphiti(
            graph_driver=self.driver,
            llm_client=llm,
            embedder=self._build_embedder(local_model),
            cross_encoder=OrderPreservingReranker(),
            max_coroutines=s.concurrency,
        )

    def _build_embedder(self, local_model: Any = None) -> Any:
        s = self.settings
        if s.embedding_base_url and s.embedding_model:
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

            return OpenAIEmbedder(config=OpenAIEmbedderConfig(
                api_key=s.embedding_api_key or "none",
                base_url=s.embedding_base_url,
                embedding_model=s.embedding_model,
                embedding_dim=s.embedding_dim or 768,
            ))
        from graphiti_core.embedder.client import EmbedderClient

        class LocalEmbedder(EmbedderClient):
            """sentence-transformers on this machine: no embedding API, nothing leaves the VM."""

            def __init__(self) -> None:
                self._model = local_model

            def _load(self):
                return self._model

            async def create(self, input_data: Any) -> List[float]:
                texts = [input_data] if isinstance(input_data, str) else list(input_data)
                vectors = await asyncio.to_thread(lambda: self._load().encode(texts, normalize_embeddings=True))
                return [float(x) for x in vectors[0]]

            async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
                vectors = await asyncio.to_thread(lambda: self._load().encode(list(input_data_list), normalize_embeddings=True))
                return [[float(x) for x in v] for v in vectors]

        return LocalEmbedder()

    def _driver_for(self, graph_id: str) -> Any:
        driver = self._graph_driver_cache.get(graph_id)
        if driver is None:
            driver = self._graph_driver_cache[graph_id] = self.driver.clone(database=graph_id)
        return driver

    async def list_graphs(self) -> List[str]:
        return [str(g) for g in await self.driver.client.list_graphs()]

    async def create_graph(self, graph_id: str) -> None:
        await self._driver_for(graph_id).build_indices_and_constraints()

    async def delete_graph(self, graph_id: str) -> None:
        await self.driver.client.select_graph(graph_id).delete()
        self._graph_driver_cache.pop(graph_id, None)

    async def add_episode(
        self, *, graph_id: str, episode_uuid: str, name: str, body: str, source_description: str,
        reference_time: datetime, entity_types: Optional[dict], edge_types: Optional[dict], edge_type_map: Optional[dict],
    ) -> None:
        from graphiti_core.nodes import EpisodeType

        await self.graphiti.add_episode(
            name=name,
            episode_body=body,
            source_description=source_description,
            reference_time=reference_time,
            source=EpisodeType.text,
            group_id=graph_id,
            # no uuid=: Graphiti reads a supplied uuid as "update this existing episode" and fails for a new one.
            # ``episode_uuid`` is the id the app polls; the client tracks it, Graphiti keeps its own.
            entity_types=entity_types,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
        )

    async def search(self, graph_id: str, query: str, scope: str, limit: int) -> Tuple[list, list]:
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF, NODE_HYBRID_SEARCH_RRF

        recipe = NODE_HYBRID_SEARCH_RRF if scope == "nodes" else EDGE_HYBRID_SEARCH_RRF
        results = await self.graphiti.search_(query=query, config=recipe.model_copy(update={"limit": limit}), group_ids=[graph_id])
        return list(results.edges or []), list(results.nodes or [])

    async def list_nodes(self, graph_id: str, limit: int, cursor: Optional[str]) -> list:
        from graphiti_core.errors import GroupsNodesNotFoundError
        from graphiti_core.nodes import EntityNode

        try:
            return list(await EntityNode.get_by_group_ids(self._driver_for(graph_id), [graph_id], limit=limit, uuid_cursor=cursor))
        except GroupsNodesNotFoundError:   # Zep answers an empty page; Graphiti raises
            return []

    async def list_edges(self, graph_id: str, limit: int, cursor: Optional[str]) -> list:
        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError

        try:
            return list(await EntityEdge.get_by_group_ids(self._driver_for(graph_id), [graph_id], limit=limit, uuid_cursor=cursor))
        except GroupsEdgesNotFoundError:
            return []

    async def get_node(self, graph_id: str, node_uuid: str) -> Any:
        from graphiti_core.nodes import EntityNode

        try:
            return await EntityNode.get_by_uuid(self._driver_for(graph_id), node_uuid)
        except Exception:  # graphiti raises NodeNotFoundError for an unknown uuid
            return None

    async def node_edges(self, graph_id: str, node_uuid: str) -> list:
        from graphiti_core.edges import EntityEdge

        try:
            return list(await EntityEdge.get_by_node_uuid(self._driver_for(graph_id), node_uuid))
        except Exception:   # a node without edges: Graphiti may raise EdgesNotFoundError
            return []

    async def close(self) -> None:
        if self.graphiti is not None:
            await self.graphiti.close()


# --------------------------------------------------------------------------------------------------
# the process-wide client
# --------------------------------------------------------------------------------------------------

_client: Optional[GraphitiZepClient] = None
_client_lock = threading.Lock()


def get_graphiti_client() -> GraphitiZepClient:
    """The shared Graphiti-backed client (built on first use: connects to FalkorDB and sets up the LLM/embedder)."""
    global _client
    with _client_lock:
        if _client is None:
            settings = GraphitiSettings.from_env()
            loop = _LoopThread()
            engine = GraphitiEngine(settings)
            loop.run(engine.start(), 300)
            _client = GraphitiZepClient(engine, settings.state_dir, loop, episode_retries=settings.episode_retries)
            logger.info(
                f"Graphiti backend ready: db={settings.db_host}:{settings.db_port} llm={settings.llm_model} "
                f"embedder={'api:' + settings.embedding_model if settings.embedding_base_url else 'local:' + settings.local_embedding_model}"
            )
        return _client


def reset_graphiti_client() -> None:
    """Drop the shared client (tests, controlled reconfiguration)."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None
