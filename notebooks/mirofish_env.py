"""Turns the notebook's Settings and Secrets into the backend's ``.env``, with one model per job.

The jobs and the names (the backend also still reads the old names, see backend/env_aliases.py):

    TWITTER_LLM_*       Twitter agents and the end-of-run poll                                       (was LLM_*)
    REDDIT_LLM_*        Reddit agents                                                                (was LLM_BOOST_*)
    MAIN_LLM_*          the ontology, agent personas, the simulation config, the outcome questions,
                        the report and its tools                                                     (was LLM_PREP_*)
    GRAPHITI_LLM_*      reading the seed into the knowledge graph (Graphiti)                         (new)
    TWITTER_RERANKER_*  the hosted twhin-bert behind the Twitter feed; blank = run it on this VM     (was LLM_EMBEDDING_*)
    GRAPHITI_EMBEDDING_*  embeddings for graph search; blank = a small local model                   (new)

A job left blank follows another: Reddit and Main follow Twitter, Graphiti follows Main. The backend only starts the
Reddit or Main model when it has a key *and* a model, so this fills in the key, the base URL and the model explicitly
instead of leaving the fallback to chance.

Only the standard library, so it is easy to test.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Mapping


def _pick(*values: str) -> str:
    for value in values:
        if (value or "").strip():
            return value.strip()
    return ""


def build_env(
    settings: Mapping[str, object],
    keys: Mapping[str, str],
    *,
    work: str,
) -> Dict[str, str]:
    """``settings``: the Settings cell's variables. ``keys``: the secrets, by the names used in the notebook."""

    def s(name: str) -> str:
        return str(settings.get(name) or "").strip()

    twitter_key = _pick(keys.get("TWITTER_LLM_API_KEY", ""), keys.get("LLM_API_KEY", ""))
    if not twitter_key:
        raise ValueError("no API key: add the secret TWITTER_LLM_API_KEY (or LLM_API_KEY)")
    twitter = {"base_url": s("TWITTER_LLM_BASE_URL"), "model": s("TWITTER_LLM_MODEL_NAME"), "key": twitter_key}
    if not (twitter["base_url"] and twitter["model"]):
        raise ValueError("TWITTER_LLM_BASE_URL and TWITTER_LLM_MODEL_NAME are required")

    def role(prefix: str, fallback: Dict[str, str]) -> Dict[str, str]:
        """This job's endpoint: what the user set, else the job it follows."""
        return {
            "base_url": _pick(s(f"{prefix}_BASE_URL"), fallback["base_url"]),
            "model": _pick(s(f"{prefix}_MODEL_NAME"), fallback["model"]),
            "key": _pick(keys.get(f"{prefix}_API_KEY", ""), fallback["key"]),
        }

    def chosen(prefix: str) -> bool:
        return bool(s(f"{prefix}_MODEL_NAME") or s(f"{prefix}_BASE_URL") or keys.get(f"{prefix}_API_KEY"))

    env: Dict[str, str] = {
        "GRAPH_BACKEND": "graphiti",
        "ZEP_GRAPH_MEMORY_UPDATE": "true" if settings.get("LIVE_GRAPH_UPDATES") else "false",
        "TWITTER_LLM_API_KEY": twitter["key"],
        "TWITTER_LLM_BASE_URL": twitter["base_url"],
        "TWITTER_LLM_MODEL_NAME": twitter["model"],
        "GRAPHITI_DB_PORT": "6390",
        "GRAPHITI_CONCURRENCY": str(settings.get("GRAPHITI_CONCURRENCY") or 4),
        "GRAPHITI_EPISODE_WORKERS": str(settings.get("GRAPHITI_EPISODE_WORKERS") or 1),
        "GRAPHITI_DEDUPE": s("GRAPHITI_DEDUPE") or "exact",
        "GRAPHITI_STATE_DIR": f"{work}/graphiti_state",
        "FLASK_PORT": "5001",
        "FLASK_HOST": "127.0.0.1",
        "LLM_MODEL_TIMEOUT": "180",
        "LLM_MODEL_MAX_RETRIES": "1",
    }

    # Reddit and Main exist only when chosen: the backend treats "a key is set" as "use a separate model".
    if chosen("REDDIT_LLM"):
        reddit = role("REDDIT_LLM", twitter)
        env.update(REDDIT_LLM_API_KEY=reddit["key"], REDDIT_LLM_BASE_URL=reddit["base_url"], REDDIT_LLM_MODEL_NAME=reddit["model"])

    main = twitter
    if chosen("MAIN_LLM"):
        main = role("MAIN_LLM", twitter)
        env.update(MAIN_LLM_API_KEY=main["key"], MAIN_LLM_BASE_URL=main["base_url"], MAIN_LLM_MODEL_NAME=main["model"])

    graphiti = role("GRAPHITI_LLM", main)       # follows Main, which follows Twitter
    env.update(GRAPHITI_LLM_API_KEY=graphiti["key"], GRAPHITI_LLM_BASE_URL=graphiti["base_url"], GRAPHITI_LLM_MODEL_NAME=graphiti["model"])

    if s("TWITTER_RERANKER_BASE_URL"):
        env.update(
            TWITTER_RERANKER_BASE_URL=s("TWITTER_RERANKER_BASE_URL"),
            TWITTER_RERANKER_API_KEY=keys.get("TWITTER_RERANKER_API_KEY", ""),
        )
        if s("TWITTER_RERANKER_MODEL_NAME"):
            env["TWITTER_RERANKER_MODEL_NAME"] = s("TWITTER_RERANKER_MODEL_NAME")

    if s("EMBEDDING") == "api":
        if not (s("EMBEDDING_BASE_URL") and s("EMBEDDING_MODEL_NAME")):
            raise ValueError('EMBEDDING = "api" needs EMBEDDING_BASE_URL and EMBEDDING_MODEL_NAME')
        env.update(
            GRAPHITI_EMBEDDING_BASE_URL=s("EMBEDDING_BASE_URL"),
            GRAPHITI_EMBEDDING_MODEL_NAME=s("EMBEDDING_MODEL_NAME"),
            GRAPHITI_EMBEDDING_API_KEY=keys.get("EMBEDDING_API_KEY", ""),
            GRAPHITI_EMBEDDING_DIM=str(settings.get("EMBEDDING_DIM") or 768),
        )
    return {k: v for k, v in env.items() if v != ""}


def parse_extra_env(text: str, secret: Callable[[str], str] = lambda name: "") -> Dict[str, str]:
    """``KEY=VALUE`` lines from the notebook's EXTRA_ENV box. Blank lines and ``# comments`` are skipped.

    A value ``secret:NAME`` is replaced by the Secret ``NAME``, so a key can be used without pasting it into the
    notebook (which is saved and shared with the text in it).
    """
    extra: Dict[str, str] = {}
    for number, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"EXTRA_ENV line {number}: expected KEY=VALUE, got {raw.strip()[:60]!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value.startswith("secret:"):
            name = value[len("secret:"):].strip()
            value = secret(name)
            if not value:
                raise ValueError(f"EXTRA_ENV line {number}: the Secret {name!r} was not found")
        extra[key] = value
    return extra


def merge_extra_env(env: Dict[str, str], extra: Mapping[str, str]) -> List[str]:
    """Add ``extra`` to ``env`` (the user's explicit lines win). Returns warnings for lines that will not take effect.

    A line using an OLD name (``LLM_MODEL_NAME``) is ignored by the backend when the new name
    (``TWITTER_LLM_MODEL_NAME``) is also set, because a new name wins; this says so instead of failing silently.
    """
    try:
        from env_aliases import ALIASES    # backend/env_aliases.py: new name -> old name
    except ImportError:
        ALIASES = {}
    new_for_old = {old: new for new, old in ALIASES.items()}
    warnings: List[str] = []
    for key, value in extra.items():
        shadow = new_for_old.get(key)
        if shadow and (env.get(shadow) or "").strip() and shadow not in extra:
            warnings.append(f"{key} is ignored: {shadow} is set (from Settings) and a new name wins; set {shadow} instead")
        env[key] = value
    return warnings


def to_dotenv(env: Mapping[str, str]) -> str:
    return "\n".join(f"{k}={v}" for k, v in env.items()) + "\n"
