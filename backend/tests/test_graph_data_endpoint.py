import pytest
from zep_cloud.core.api_error import ApiError as ZepApiError

from app import create_app
from app.api import graph as graph_api
from app.config import Config


class FakeBuilder:
    calls = 0
    error = None

    def __init__(self, api_key=None):
        pass

    def get_graph_data(self, graph_id, max_retries=None):
        FakeBuilder.calls += 1
        FakeBuilder.last_retries = max_retries
        if FakeBuilder.error:
            raise FakeBuilder.error
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(Config, "ZEP_API_KEY", "k")
    monkeypatch.setattr(graph_api, "GraphBuilderService", FakeBuilder)
    graph_api._graph_data_cache.clear()
    FakeBuilder.calls, FakeBuilder.error = 0, None
    return create_app().test_client()


def test_concurrent_pages_share_one_zep_read_and_it_fails_fast(client):
    assert client.get("/api/graph/data/g1").status_code == 200
    assert client.get("/api/graph/data/g1").status_code == 200
    assert FakeBuilder.calls == 1 and FakeBuilder.last_retries == 1
    client.get("/api/graph/data/g2")
    assert FakeBuilder.calls == 2


def test_the_cache_expires(client, monkeypatch):
    client.get("/api/graph/data/g1")
    monkeypatch.setattr(graph_api, "GRAPH_DATA_CACHE_SECONDS", 0.0)
    client.get("/api/graph/data/g1")
    assert FakeBuilder.calls == 2


def test_a_zep_rate_limit_is_a_429_the_page_can_wait_out(client):
    FakeBuilder.error = ZepApiError(status_code=429, headers={"x-ratelimit-reset": "1", "retry-after": "0"}, body="Rate limit")
    response = client.get("/api/graph/data/g1")
    body = response.get_json()
    assert response.status_code == 429
    assert body["error_code"] == "zep_rate_limited" and body["retry_after_seconds"] >= 1
    assert "rate" in body["error"].lower() and body["success"] is False


def test_other_zep_errors_stay_500(client):
    FakeBuilder.error = ZepApiError(status_code=503, body="down")
    assert client.get("/api/graph/data/g1").status_code == 500
