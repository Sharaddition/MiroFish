import httpx
import openai
import pytest

from app.config import Config
from app.utils import llm_client as lc

BACKUP = {"api_key": "bk", "base_url": "http://backup/v1", "model_name": "backup-model"}


def status_error(code):
    request = httpx.Request("POST", "http://x/v1/chat/completions")
    return openai.APIStatusError("failed", response=httpx.Response(code, request=request), body=None)


class Boom:
    """Stands in for a chat-completions client: answers from a list of outcomes (exceptions are raised)."""

    def __init__(self, outcomes):
        self.outcomes, self.models = list(outcomes), []
        completions = type("C", (), {"create": self._create})
        self.chat = type("Chat", (), {"completions": completions()})()

    def _create(self, **kwargs):
        self.models.append(kwargs["model"])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def client(primary, backup, fallback=BACKUP, monkeypatch=None):
    c = lc.LLMClient(api_key="k", base_url="http://main/v1", model="main-model", fallback=fallback)
    c.client = primary
    c._fallback_client = backup
    return c


def ask(c):
    return c._create_completion(messages=[{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=5, response_format=None)


@pytest.mark.parametrize("code", [402, 429, 401, 403, 408, 413, 500, 503])
def test_these_failures_go_to_the_backup(code):
    primary, backup = Boom([status_error(code)]), Boom(["answer"])
    assert ask(client(primary, backup)) == "answer"
    assert primary.models == ["main-model"] and backup.models == ["backup-model"]


def test_a_dropped_connection_and_a_timeout_go_to_the_backup():
    for error in (openai.APIConnectionError(request=httpx.Request("POST", "http://x")),
                  openai.APITimeoutError(request=httpx.Request("POST", "http://x"))):
        assert ask(client(Boom([error]), Boom(["answer"]))) == "answer"


def test_a_bad_request_is_not_retried_elsewhere():
    primary, backup = Boom([status_error(400)]), Boom(["never"])
    with pytest.raises(openai.APIStatusError):
        ask(client(primary, backup))
    assert backup.models == []


def test_with_no_backup_the_error_is_raised():
    with pytest.raises(openai.APIStatusError):
        ask(client(Boom([status_error(429)]), Boom(["never"]), fallback=None))


def test_a_failing_backup_raises_its_own_error_and_is_tried_once():
    backup = Boom([status_error(500)])
    with pytest.raises(openai.APIStatusError):
        ask(client(Boom([status_error(429)]), backup))
    assert backup.models == ["backup-model"]


def test_the_next_call_tries_the_main_model_again():
    primary, backup = Boom([status_error(429), "fine"]), Boom(["answer"])
    c = client(primary, backup)
    assert ask(c) == "answer" and ask(c) == "fine"
    assert primary.models == ["main-model", "main-model"]


def test_chat_json_uses_the_backup_too():
    reply = type("R", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": '{"a": 1}'})(), "finish_reason": "stop"})()]})()
    assert client(Boom([status_error(429)]), Boom([reply])).chat_json([{"role": "user", "content": "x"}]) == {"a": 1}


# --- which backup is configured ---

PRIMARY = {"api_key": "pk", "base_url": "http://prep/v1", "model_name": "prep-model"}


@pytest.fixture
def env(monkeypatch):
    for name in ("LLM_FALLBACK", "LLM_FALLBACK_API_KEY", "LLM_FALLBACK_BASE_URL", "LLM_FALLBACK_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "tk")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "http://twitter/v1")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "twitter-model")
    return monkeypatch


def test_the_dedicated_backup_comes_first_and_its_base_url_defaults_to_the_main_one(env):
    env.setenv("LLM_FALLBACK_API_KEY", "fk")
    env.setenv("LLM_FALLBACK_MODEL_NAME", "cf-model")
    assert Config.fallback_llm(PRIMARY) == {"api_key": "fk", "base_url": "http://twitter/v1", "model_name": "cf-model"}
    env.setenv("LLM_FALLBACK_BASE_URL", "http://cf/v1")
    assert Config.fallback_llm(PRIMARY)["base_url"] == "http://cf/v1"


def test_without_a_dedicated_backup_the_twitter_model_is_the_backup(env):
    assert Config.fallback_llm(PRIMARY) == {"api_key": "tk", "base_url": "http://twitter/v1", "model_name": "twitter-model"}


def test_no_backup_when_it_would_be_the_same_model_or_is_switched_off(env):
    same = {"api_key": "tk", "base_url": "http://twitter/v1", "model_name": "twitter-model"}
    assert Config.fallback_llm(same) is None
    env.setenv("LLM_FALLBACK", "false")
    assert Config.fallback_llm(PRIMARY) is None


def test_for_preparation_carries_the_backup(env):
    env.setattr(Config, "prep_llm", classmethod(lambda cls: PRIMARY))
    assert lc.LLMClient.for_preparation().fallback["model_name"] == "twitter-model"
