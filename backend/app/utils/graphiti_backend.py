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
import re
import threading
import uuid as uuidlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

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
    episode_workers: int = 1
    dedupe: str = "exact"
    dedupe_max_calls: int = 50
    dedupe_auto_score: float = 0.97
    dedupe_ask_score: float = 0.85
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

        def score(name: str, default: float) -> float:
            try:
                value = float(env.get(name) or default)
            except ValueError:
                value = default
            return value if 0.0 < value <= 1.0 else default

        dedupe_mode = (text("GRAPHITI_DEDUPE", "exact") or "exact").lower()
        if dedupe_mode not in DEDUPE_MODES:
            dedupe_mode = "exact"
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
            episode_workers=number("GRAPHITI_EPISODE_WORKERS", 1),
            dedupe=dedupe_mode,
            dedupe_max_calls=number("GRAPHITI_DEDUPE_MAX_CALLS", 50),
            dedupe_auto_score=score("GRAPHITI_DEDUPE_AUTO_SCORE", 0.97),
            dedupe_ask_score=score("GRAPHITI_DEDUPE_ASK_SCORE", 0.85),
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


def first_json_value(text: str) -> str:
    """The first complete JSON value in ``text``, dropping anything after it.

    Some models add a second object or a closing remark after the answer ("Extra data: line 3 column 1"), which makes a
    plain ``json.loads`` fail although the answer itself is fine. Text that does not start with JSON is returned as it
    was, so the caller still reports the original problem.
    """
    stripped = text.lstrip()
    try:
        _, end = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return text
    return stripped[:end]


def unwrap_layout(response: Any, model: Any) -> Any:
    """An answer that echoes the schema's layout (``{"properties": {...}}``) is reduced to the inner object.

    Some models do this for any structured answer, including Graphiti's own (e.g. ``EdgeDuplicate``), which then
    fails with "Field required". Only acts when none of the model's fields are at the top and exactly one nested
    object has some of them; any other answer is returned untouched.
    """
    fields = set(getattr(model, "model_fields", {}) or {})
    if isinstance(response, dict) and fields and not fields & set(response):
        wrapped = [v for v in response.values() if isinstance(v, dict) and fields & set(v)]
        if len(wrapped) == 1:
            return wrapped[0]
    return response


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
    response = unwrap_layout(response, model)
    return {key: _as_text(value) for key, value in response.items() if key in fields}


# --------------------------------------------------------------------------------------------------
# duplicate cleanup: planning (pure functions, no database)
# --------------------------------------------------------------------------------------------------

DEDUPE_MODES = ("off", "exact", "llm")


def normalise_name(name: Any) -> str:
    """Lower case, punctuation gone, spaces collapsed: "Persistent  Systems." and "persistent systems" match."""
    return " ".join(re.sub(r"[^\w\s]", " ", str(name or "").lower()).split())


def _kind(node: Any) -> frozenset:
    return frozenset(label for label in (getattr(node, "labels", None) or []) if label not in ("Entity", "Episodic"))


def _earliest_first(nodes: List[Any]) -> List[Any]:
    return sorted(nodes, key=lambda n: (str(getattr(n, "created_at", "") or ""), str(getattr(n, "uuid", ""))))


def plan_exact_merges(nodes: List[Any]) -> List[Tuple[Any, List[Any]]]:
    """Groups of nodes with the same normalised name and the same entity type: (node to keep, nodes to merge into it).

    Graphiti labels an entity with one of the ontology's types in some chunks and not in others, so a node without a
    type joins the typed one of the same name when there is exactly one such type; two different types stay apart.
    """
    by_name: Dict[str, Dict[frozenset, List[Any]]] = {}
    for node in nodes:
        name = normalise_name(getattr(node, "name", ""))
        if name:
            by_name.setdefault(name, {}).setdefault(_kind(node), []).append(node)
    plan = []
    for kinds in by_name.values():
        typed = [k for k in kinds if k]
        untyped = kinds.get(frozenset(), [])
        if len(typed) == 1:
            groups = [kinds[typed[0]] + untyped]
        else:
            groups = [kinds[k] for k in typed] + ([untyped] if untyped and not typed else [])
        for group in groups:
            if len(group) > 1:
                ordered = _earliest_first([n for n in group if _kind(n)]) + _earliest_first([n for n in group if not _kind(n)])
                plan.append((ordered[0], ordered[1:]))
    return plan


def plan_fuzzy_candidates(nodes: List[Any]) -> List[Tuple[Any, Any]]:
    """Pairs (keep, drop) of the same entity type where one name's words are all inside the other's
    ("Persistent" / "Persistent Systems"). The longer name is kept. These still need a yes from the model."""
    by_kind: Dict[frozenset, List[Any]] = {}
    for node in nodes:
        if normalise_name(getattr(node, "name", "")):
            by_kind.setdefault(_kind(node), []).append(node)
    pairs = []
    for group in by_kind.values():
        names = {}
        for node in group:
            names.setdefault(normalise_name(node.name), node)           # one representative per distinct name
        distinct = sorted(names.items(), key=lambda item: -len(item[0].split()))
        for i, (long_name, long_node) in enumerate(distinct):
            long_words = set(long_name.split())
            for short_name, short_node in distinct[i + 1:]:
                short_words = set(short_name.split())
                if short_words < long_words and len(short_name) >= 3:
                    pairs.append((long_node, short_node))
    return pairs


def rank_candidates(nodes: List[Any], vectors: Optional[Dict[str, List[float]]], ask_score: float) -> List[Dict[str, Any]]:
    """Pairs of different names that may be one entity, closest first: ``{"keep", "drop", "score"}``.

    A pair is a candidate when one name's words are inside the other's, or when the name embeddings are at least
    ``ask_score`` close (cosine). Pairs are only formed within one entity type. ``vectors`` maps a normalised name to its
    embedding; without it only the word rule applies and ``score`` is None. The longer name is the one kept.
    """
    by_kind: Dict[frozenset, Dict[str, Any]] = {}
    for node in nodes:
        name = normalise_name(getattr(node, "name", ""))
        if name:
            by_kind.setdefault(_kind(node), {}).setdefault(name, node)       # one representative per distinct name
    found: Dict[Tuple[str, str, frozenset], Dict[str, Any]] = {}

    def add(kind: frozenset, a: str, b: str, node_a: Any, node_b: Any, score: Optional[float]) -> None:
        if (len(a.split()), len(a)) < (len(b.split()), len(b)):
            a, b, node_a, node_b = b, a, node_b, node_a
        found[(a, b, kind)] = {"keep": node_a, "drop": node_b, "score": score}

    for kind, names in by_kind.items():
        ordered = sorted(names)
        if vectors:
            import numpy as np

            known = [n for n in ordered if n in vectors]
            if len(known) > 1:
                matrix = np.array([vectors[n] for n in known], dtype=float)
                matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
                similarity = matrix @ matrix.T
                above = (similarity >= ask_score) & np.triu(np.ones(similarity.shape, dtype=bool), k=1)
                for i, j in zip(*np.where(above)):
                    add(kind, known[i], known[j], names[known[i]], names[known[j]], float(similarity[i, j]))
        for keep, drop in plan_fuzzy_candidates(list(names.values())):
            a, b = normalise_name(keep.name), normalise_name(drop.name)
            if (a, b, kind) in found or (b, a, kind) in found:
                continue
            score = None
            if vectors and a in vectors and b in vectors:
                va, vb = np.array(vectors[a], dtype=float), np.array(vectors[b], dtype=float)
                score = float(va @ vb / max(float(np.linalg.norm(va) * np.linalg.norm(vb)), 1e-12))
            add(kind, a, b, names[a], names[b], score)
    return sorted(found.values(), key=lambda c: -(c["score"] if c["score"] is not None else 0.0))


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
                 retry_delay: float = 3.0, episode_workers: int = 1, dedupe: str = "off",
                 dedupe_max_calls: int = 50, dedupe_auto_score: float = 0.97, dedupe_ask_score: float = 0.85) -> None:
        self._engine = engine
        self._episode_retries, self._retry_delay = max(0, int(episode_retries)), retry_delay
        self._episode_workers = max(1, int(episode_workers))
        self._dedupe, self._dedupe_max_calls = dedupe, max(0, int(dedupe_max_calls))
        self._dedupe_auto_score, self._dedupe_ask_score = dedupe_auto_score, dedupe_ask_score
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
        """Run ``job`` after the earlier jobs of the same graph (Graphiti dedupes against what came before).

        With ``episode_workers`` above 1, that many chunks of a graph run at once: faster, but chunks that run together
        cannot see each other's entities yet, so the same entity can end up twice.
        """

        async def put() -> None:
            queue = self._queues.get(graph_id)
            if queue is None:
                queue = self._queues[graph_id] = asyncio.Queue()
                for _ in range(self._episode_workers):
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

    # -- duplicate cleanup -----------------------------------------------------------------------------

    async def dedupe_graph(self, graph_id: str) -> Dict[str, int]:
        """Merge entities that are the same thing under one name. Run once the chunks of a build are all in: chunks read
        at the same time cannot see each other's entities, so each makes its own. Returns what it did."""
        report = {"nodes": 0, "exact_merged": 0, "fuzzy_merged": 0, "model_calls": 0}
        if self._dedupe == "off":
            return report
        nodes = await self._all_nodes(graph_id)
        report["nodes"] = len(nodes)
        for keep, drops in plan_exact_merges(nodes):
            await self._engine.merge_nodes(graph_id, keep.uuid, [d.uuid for d in drops])
            report["exact_merged"] += len(drops)
            logger.info("graphiti dedupe %s: merged %s into %r", graph_id, [d.name for d in drops], keep.name)
        if self._dedupe == "llm":
            nodes = await self._all_nodes(graph_id)
            vectors = await self._name_vectors(nodes)
            merged_away: set = set()
            for pair in rank_candidates(nodes, vectors, self._dedupe_ask_score):
                keep, drop, score = pair["keep"], pair["drop"], pair["score"]
                if drop.uuid in merged_away or keep.uuid in merged_away:
                    continue
                if score is not None and score >= self._dedupe_auto_score:
                    same = True                    # so close that the model is not asked
                else:
                    if report["model_calls"] >= self._dedupe_max_calls:
                        logger.warning("graphiti dedupe %s: stopped at %d model calls (GRAPHITI_DEDUPE_MAX_CALLS)", graph_id,
                                       self._dedupe_max_calls)
                        break
                    report["model_calls"] += 1
                    try:
                        same = await self._engine.same_entity(keep, drop)
                    except Exception as error:     # one failed judgement leaves that pair unmerged
                        logger.warning("graphiti dedupe %s: could not judge %r / %r: %s", graph_id, keep.name, drop.name, error)
                        continue
                if same:
                    same_name = [n for n in nodes if normalise_name(n.name) == normalise_name(drop.name) and _kind(n) == _kind(drop)]
                    await self._engine.merge_nodes(graph_id, keep.uuid, [n.uuid for n in same_name])
                    merged_away.update(n.uuid for n in same_name)
                    report["fuzzy_merged"] += len(same_name)
                    logger.info("graphiti dedupe %s: merged %r into %r (%s)", graph_id, drop.name, keep.name,
                                "similarity %.3f, no model call" % score if score is not None and score >= self._dedupe_auto_score
                                else "confirmed by the model" + (", similarity %.3f" % score if score is not None else ""))
        return report

    async def _name_vectors(self, nodes: List[Any]) -> Optional[Dict[str, List[float]]]:
        """Embeddings of the distinct node names, by normalised name; None if the embedder fails (word overlap still works)."""
        names = sorted({normalise_name(n.name) for n in nodes if normalise_name(n.name)})
        if len(names) < 2:
            return None
        try:
            return dict(zip(names, await self._engine.embed_names(names)))
        except Exception as error:
            logger.warning("graphiti dedupe: could not embed the names (%s); using name overlap only", error)
            return None

    async def _all_nodes(self, graph_id: str) -> List[Any]:
        nodes: List[Any] = []
        cursor = None
        while True:
            page = await self._engine.list_nodes(graph_id, 200, cursor)
            nodes.extend(page)
            if len(page) < 200:
                return nodes
            cursor = page[-1].uuid

    def _after_batch(self, graph_ids: Iterable[str], settle: Callable[[], None]) -> None:
        """Clean up duplicates in the graphs a batch wrote to, then let the batch finish (the app waits for that)."""
        graphs = sorted({g for g in graph_ids if g})
        if self._dedupe == "off" or not graphs:
            settle()
            return

        async def clean() -> None:
            try:
                for graph_id in graphs:
                    try:
                        logger.info("graphiti dedupe %s: %s", graph_id, await self.dedupe_graph(graph_id))
                    except Exception:      # a failed cleanup must never fail the build: the graph is still usable
                        logger.exception("graphiti dedupe of %s failed; the graph is kept as built", graph_id)
            finally:
                settle()

        self._loop.submit(clean())

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
                    message = f"{type(error).__name__}: {error}"[:500]
                    if attempt < self._episode_retries:      # still trying: other threads polling must not see "failed"
                        logger.warning(f"Graphiti episode {episode.uuid} failed ({message}); retrying")
                        await asyncio.sleep(self._retry_delay)
                    else:
                        episode.error, episode.status = message, "failed"
                        logger.error(f"Graphiti episode {episode.uuid} failed: {message}")
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
                done = remaining["n"] <= 0
            if done:
                def settle() -> None:
                    failed = sum(1 for i in batch.items if i.status == "failed")
                    batch.status = "succeeded" if not failed else ("failed" if failed == len(batch.items) else "partial")

                self._c._after_batch((i.graph_id for i in batch.items), settle)

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
            """Graphiti's OpenAI-compatible client, with the attribute answers made safe to store (see sanitize_attributes) and echoed schema layouts unwrapped."""

            def _strip_code_fences(self, text):
                return first_json_value(super()._strip_code_fences(text))

            async def generate_response(self, messages, response_model=None, *args, **kwargs):
                result = await super().generate_response(messages, response_model, *args, **kwargs)
                if response_model is None:
                    return result
                if is_custom_type(response_model):
                    return sanitize_attributes(result, response_model)
                return unwrap_layout(result, response_model)

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

    async def _fact_embedding(self, edge: Any, driver: Any) -> None:
        """Saving an edge writes its embedding, and a listed edge does not carry it: read it back (or make it again)."""
        try:
            await edge.load_fact_embedding(driver)
        except Exception:
            edge.fact_embedding = None
        if not edge.fact_embedding:
            await edge.generate_embedding(self.graphiti.embedder)

    async def merge_nodes(self, graph_id: str, keep_uuid: str, drop_uuids: List[str]) -> None:
        """Make every node in ``drop_uuids`` part of ``keep_uuid``: its facts and episode links move over, then it is deleted."""
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EntityNode

        driver = self._driver_for(graph_id)
        keep = await EntityNode.get_by_uuid(driver, keep_uuid)
        for drop_uuid in drop_uuids:
            if drop_uuid == keep_uuid:
                continue
            try:
                edges = list(await EntityEdge.get_by_node_uuid(driver, drop_uuid))
            except Exception:       # a node without facts: Graphiti may raise instead of answering empty
                edges = []
            for edge in edges:
                source = keep_uuid if edge.source_node_uuid == drop_uuid else edge.source_node_uuid
                target = keep_uuid if edge.target_node_uuid == drop_uuid else edge.target_node_uuid
                await self._fact_embedding(edge, driver)
                await edge.delete(driver)
                if source == target:        # a fact between two copies of one entity says nothing once they are one
                    continue
                edge.source_node_uuid, edge.target_node_uuid = source, target
                await edge.save(driver)
            await driver.execute_query(
                "MATCH (e:Episodic)-[m:MENTIONS]->(d:Entity {uuid: $drop}) MATCH (k:Entity {uuid: $keep}) "
                "MERGE (e)-[n:MENTIONS {uuid: m.uuid}]->(k) SET n.group_id = m.group_id, n.created_at = m.created_at",
                drop=drop_uuid, keep=keep_uuid,
            )
            dropped = await EntityNode.get_by_uuid(driver, drop_uuid)
            if not (keep.summary or "").strip() and (dropped.summary or "").strip():
                keep.summary = dropped.summary
                await driver.execute_query("MATCH (k:Entity {uuid: $keep}) SET k.summary = $summary", keep=keep_uuid, summary=keep.summary)
            await dropped.delete(driver)
        await self._merge_repeated_facts(driver, keep_uuid)

    async def _merge_repeated_facts(self, driver: Any, node_uuid: str) -> None:
        """After a merge a node can hold the same fact twice (once from each copy): keep one, with both sets of episodes."""
        from graphiti_core.edges import EntityEdge

        try:
            edges = list(await EntityEdge.get_by_node_uuid(driver, node_uuid))
        except Exception:
            return
        seen: Dict[Tuple[str, str, str, str], Any] = {}
        for edge in sorted(edges, key=lambda e: str(e.created_at or "")):
            key = (edge.source_node_uuid, edge.target_node_uuid, edge.name, normalise_name(edge.fact))
            first = seen.setdefault(key, edge)
            if first is edge:
                continue
            first.episodes = list(dict.fromkeys(list(first.episodes or []) + list(edge.episodes or [])))
            await self._fact_embedding(first, driver)
            await first.save(driver)
            await edge.delete(driver)

    async def embed_names(self, names: List[str]) -> List[List[float]]:
        """Embeddings with the graph's own embedder (the local model by default: no model-provider calls)."""
        return [list(v) for v in await self.graphiti.embedder.create_batch(list(names))]

    async def same_entity(self, a: Any, b: Any) -> bool:
        """Ask the graph model whether two nodes are one real-world thing. Conservative: only a clear yes counts."""
        from graphiti_core.prompts.models import Message
        from pydantic import BaseModel

        class SameEntity(BaseModel):
            same: bool

        def describe(node: Any) -> str:
            kind = ", ".join(sorted(_kind(node))) or "unknown"
            return f"name: {node.name}" + chr(10) + f"type: {kind}" + chr(10) + f"summary: {(node.summary or '')[:400]}"

        messages = [
            Message(role="system", content="You decide whether two records in a knowledge graph describe the same real-world "
                                           "entity. Answer {\"same\": true} only if they are clearly the same thing (one is a "
                                           "shorter or alternative name for the other). If they could be different "
                                           "things, answer {\"same\": false}."),
            Message(role="user", content="Record A:" + chr(10) + describe(a) + chr(10) + chr(10) + "Record B:" + chr(10)
                    + describe(b) + chr(10) + chr(10) + "Are A and B the same entity?"),
        ]
        answer = await self.graphiti.llm_client.generate_response(messages, response_model=SameEntity)
        value = answer.get("same") if isinstance(answer, dict) else None
        return value is True or str(value).strip().lower() == "true"

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
            _client = GraphitiZepClient(engine, settings.state_dir, loop, episode_retries=settings.episode_retries,
                                     episode_workers=settings.episode_workers, dedupe=settings.dedupe,
                                     dedupe_max_calls=settings.dedupe_max_calls, dedupe_auto_score=settings.dedupe_auto_score,
                                     dedupe_ask_score=settings.dedupe_ask_score)
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
