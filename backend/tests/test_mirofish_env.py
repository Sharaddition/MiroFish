"""One model per job: the notebook's settings -> the backend's .env, through the real alias mapping."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "notebooks"))
import env_aliases  # noqa: E402
from mirofish_env import build_env, to_dotenv  # noqa: E402

BASE = {"TWITTER_LLM_BASE_URL": "https://t/v1", "TWITTER_LLM_MODEL_NAME": "twitter-m", "GRAPHITI_CONCURRENCY": 4}
KEYS = {"TWITTER_LLM_API_KEY": "tk"}


def old_names(settings, keys=KEYS):
    """What the backend actually reads: the .env after the aliases are applied."""
    return env_aliases.with_aliases(build_env({**BASE, **settings}, keys, work="/w"))


def test_with_only_the_twitter_model_nothing_else_is_split_off():
    env = build_env(BASE, KEYS, work="/w")
    assert env["TWITTER_LLM_MODEL_NAME"] == "twitter-m"
    assert not any(k.startswith(("REDDIT_LLM", "MAIN_LLM", "TWITTER_RERANKER")) for k in env)    # the backend treats a key as "use a separate model"
    got = old_names({})
    assert (got["LLM_API_KEY"], got["LLM_MODEL_NAME"], got["LLM_BASE_URL"]) == ("tk", "twitter-m", "https://t/v1")
    assert "LLM_BOOST_API_KEY" not in got and "LLM_PREP_API_KEY" not in got
    # Graphiti follows Twitter when there is no Main model
    assert (env["GRAPHITI_LLM_MODEL_NAME"], env["GRAPHITI_LLM_API_KEY"]) == ("twitter-m", "tk")


def test_a_reddit_model_alone_gets_the_twitter_key_and_base_so_the_backend_really_uses_it():
    got = old_names({"REDDIT_LLM_MODEL_NAME": "reddit-m"})
    assert (got["LLM_BOOST_MODEL_NAME"], got["LLM_BOOST_API_KEY"], got["LLM_BOOST_BASE_URL"]) == ("reddit-m", "tk", "https://t/v1")
    assert got["LLM_MODEL_NAME"] == "twitter-m"                                                       # Twitter unchanged


def test_a_main_model_becomes_the_preparation_and_report_model_and_graphiti_follows_it():
    env = build_env({**BASE, "MAIN_LLM_MODEL_NAME": "main-m", "MAIN_LLM_BASE_URL": "https://m/v1"}, {**KEYS, "MAIN_LLM_API_KEY": "mk"}, work="/w")
    got = env_aliases.with_aliases(env)
    assert (got["LLM_PREP_MODEL_NAME"], got["LLM_PREP_BASE_URL"], got["LLM_PREP_API_KEY"]) == ("main-m", "https://m/v1", "mk")
    assert (env["GRAPHITI_LLM_MODEL_NAME"], env["GRAPHITI_LLM_BASE_URL"], env["GRAPHITI_LLM_API_KEY"]) == ("main-m", "https://m/v1", "mk")


def test_a_main_model_without_its_own_key_uses_the_twitter_key():
    got = old_names({"MAIN_LLM_MODEL_NAME": "main-m"})
    assert (got["LLM_PREP_API_KEY"], got["LLM_PREP_BASE_URL"]) == ("tk", "https://t/v1")              # prep needs a key AND a model to engage


def test_a_graph_model_of_its_own_wins_over_the_main_model():
    env = build_env({**BASE, "MAIN_LLM_MODEL_NAME": "main-m", "GRAPHITI_LLM_MODEL_NAME": "extractor"},
                    {**KEYS, "GRAPHITI_LLM_API_KEY": "gk"}, work="/w")
    assert (env["GRAPHITI_LLM_MODEL_NAME"], env["GRAPHITI_LLM_API_KEY"]) == ("extractor", "gk")
    assert env["MAIN_LLM_MODEL_NAME"] == "main-m"                                                      # and the report still uses Main


def test_the_hosted_feed_ranker_is_only_set_when_asked_for():
    assert "TWITTER_RERANKER_BASE_URL" not in build_env(BASE, KEYS, work="/w")
    got = old_names({"TWITTER_RERANKER_BASE_URL": "https://r.modal.run", "TWITTER_RERANKER_MODEL_NAME": "twhin-bert-base"},
                    {**KEYS, "TWITTER_RERANKER_API_KEY": "rk"})
    assert (got["LLM_EMBEDDING_BASE_URL"], got["LLM_EMBEDDING_API_KEY"], got["LLM_EMBEDDING_MODEL_NAME"]) == ("https://r.modal.run", "rk", "twhin-bert-base")


def test_graph_embeddings_are_local_unless_an_api_is_given():
    assert not any(k.startswith("GRAPHITI_EMBEDDING") for k in build_env(BASE, KEYS, work="/w"))
    env = build_env({**BASE, "EMBEDDING": "api", "EMBEDDING_BASE_URL": "https://e/v1", "EMBEDDING_MODEL_NAME": "bge", "EMBEDDING_DIM": 768},
                    {**KEYS, "EMBEDDING_API_KEY": "ek"}, work="/w")
    assert (env["GRAPHITI_EMBEDDING_BASE_URL"], env["GRAPHITI_EMBEDDING_API_KEY"], env["GRAPHITI_EMBEDDING_DIM"]) == ("https://e/v1", "ek", "768")
    with pytest.raises(ValueError, match="EMBEDDING"):
        build_env({**BASE, "EMBEDDING": "api"}, KEYS, work="/w")


def test_the_old_secret_name_still_works_and_missing_pieces_are_reported():
    assert build_env(BASE, {"LLM_API_KEY": "old"}, work="/w")["TWITTER_LLM_API_KEY"] == "old"
    assert build_env(BASE, {"LLM_API_KEY": "old", "TWITTER_LLM_API_KEY": "new"}, work="/w")["TWITTER_LLM_API_KEY"] == "new"
    with pytest.raises(ValueError, match="API key"):
        build_env(BASE, {}, work="/w")
    with pytest.raises(ValueError, match="required"):
        build_env({"TWITTER_LLM_BASE_URL": "https://t/v1"}, KEYS, work="/w")


def test_run_settings_and_the_dotenv_text():
    env = build_env({**BASE, "LIVE_GRAPH_UPDATES": True, "GRAPHITI_CONCURRENCY": 2}, KEYS, work="/kaggle/working")
    assert env["ZEP_GRAPH_MEMORY_UPDATE"] == "true" and env["GRAPHITI_CONCURRENCY"] == "2" and env["GRAPH_BACKEND"] == "graphiti"
    assert env["GRAPHITI_EPISODE_WORKERS"] == "1" and env["GRAPHITI_DEDUPE"] == "exact"
    assert build_env({**BASE, "GRAPHITI_DEDUPE": "llm"}, KEYS, work="/content")["GRAPHITI_DEDUPE"] == "llm"
    assert build_env({**BASE, "GRAPHITI_EPISODE_WORKERS": 4}, KEYS, work="/content")["GRAPHITI_EPISODE_WORKERS"] == "4"
    assert env["GRAPHITI_STATE_DIR"] == "/kaggle/working/graphiti_state"
    assert build_env(BASE, KEYS, work="/w")["ZEP_GRAPH_MEMORY_UPDATE"] == "false"
    text = to_dotenv({"A": "1", "B": "x y"})
    assert text == "A=1\nB=x y\n"


# --- EXTRA_ENV: any other setting, one KEY=VALUE per line --------------------------------------------------------

from mirofish_env import merge_extra_env, parse_extra_env  # noqa: E402


def test_extra_env_lines_comments_quotes_and_blank_lines():
    got = parse_extra_env('''
        # a comment
        LLM_MAX_CONCURRENCY=8

        SIM_STOP_GRACE_SECONDS = 600
        LLM_FALLBACK_MODEL_NAME="some-model"
        EMPTY=
    ''')
    assert got == {"LLM_MAX_CONCURRENCY": "8", "SIM_STOP_GRACE_SECONDS": "600", "LLM_FALLBACK_MODEL_NAME": "some-model", "EMPTY": ""}
    assert parse_extra_env("") == {} and parse_extra_env(None) == {}


def test_a_secret_reference_is_resolved_and_a_missing_one_is_reported():
    assert parse_extra_env("LLM_FALLBACK_API_KEY=secret:MY_KEY", lambda name: {"MY_KEY": "sk-1"}.get(name, "")) == {"LLM_FALLBACK_API_KEY": "sk-1"}
    with pytest.raises(ValueError, match="MY_KEY"):
        parse_extra_env("LLM_FALLBACK_API_KEY=secret:MY_KEY")


@pytest.mark.parametrize("bad", ["no equals sign", "1BAD=1", "BAD KEY=1", "=1"])
def test_a_malformed_line_names_its_line_number(bad):
    with pytest.raises(ValueError, match="line 2"):
        parse_extra_env(f"OK=1\n{bad}")


def test_extra_lines_win_and_a_shadowed_old_name_is_called_out():
    env = build_env(BASE, KEYS, work="/w")
    warnings = merge_extra_env(env, {"LLM_MAX_CONCURRENCY": "8", "GRAPHITI_CONCURRENCY": "2"})
    assert env["LLM_MAX_CONCURRENCY"] == "8" and env["GRAPHITI_CONCURRENCY"] == "2" and warnings == []     # the user's line wins
    # LLM_MODEL_NAME is the old name of TWITTER_LLM_MODEL_NAME, which Settings has already set: the backend would ignore it
    warnings = merge_extra_env(env, {"LLM_MODEL_NAME": "other"})
    assert len(warnings) == 1 and "TWITTER_LLM_MODEL_NAME" in warnings[0] and "ignored" in warnings[0]
    # but setting the new name itself is the way to override a Settings value
    assert merge_extra_env(env, {"TWITTER_LLM_MODEL_NAME": "other"}) == [] and env["TWITTER_LLM_MODEL_NAME"] == "other"
    assert env_aliases.with_aliases(env)["LLM_MODEL_NAME"] == "other"
