"""The notebook's pipeline driver, against a scripted backend (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "notebooks"))
import mirofish_driver as md  # noqa: E402


class Response:
    def __init__(self, body=None, status=200, text=None):
        self.body, self.status_code, self._text = body, status, text
        self.ok = status < 400

    def json(self):
        if self.body is None:
            raise ValueError("not json")
        return self.body

    @property
    def text(self):
        return self._text if self._text is not None else str(self.body)

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


class Session:
    """Replies to (METHOD, path) from queues; the last reply of a queue repeats."""

    def __init__(self, routes):
        self.routes, self.calls = {k: list(v) for k, v in routes.items()}, []

    def request(self, method, url, timeout=None, **kwargs):
        path = url.split("127.0.0.1:5001", 1)[-1]
        self.calls.append((method, path, kwargs))
        queue = self.routes[(method, path)]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def get(self, url, timeout=None, **kwargs):
        return self.request("GET", url, timeout, **kwargs)


def driver(routes):
    mf = md.MiroFish("http://127.0.0.1:5001", log=lambda _m: None)
    mf.http = Session(routes)
    return mf


def ok(data):
    return Response({"success": True, "data": data})


def test_a_backend_error_is_raised_with_the_backends_own_text():
    mf = driver({("POST", "/x"): [Response({"success": False, "error": "no graph yet"}, 409)]})
    with pytest.raises(RuntimeError, match="no graph yet"):
        mf._call("POST", "/x")
    mf = driver({("GET", "/y"): [Response(None, 502, text="<html>bad gateway</html>")]})
    with pytest.raises(RuntimeError, match="not JSON"):
        mf._call("GET", "/y")


def test_ontology_needs_a_seed_and_sends_the_prompt(tmp_path):
    mf = driver({("POST", "/api/graph/ontology/generate"): [ok({"project_id": "proj_1", "ontology": {"entity_types": [1, 2], "edge_types": [3]}, "total_text_length": 99})]})
    with pytest.raises(RuntimeError, match="no seed"):
        mf.generate_ontology([], "predict")
    seed = tmp_path / "seed.txt"
    seed.write_text("hello", encoding="utf-8")
    assert mf.generate_ontology([str(seed)], "predict the reaction") == "proj_1"
    _m, _p, kwargs = mf.http.calls[-1]
    assert kwargs["data"]["simulation_requirement"] == "predict the reaction" and kwargs["files"][0][0] == "files"


def test_graph_build_polls_the_task_and_returns_the_graph_id(monkeypatch):
    monkeypatch.setattr(md.time, "sleep", lambda _s: None)
    mf = driver({
        ("POST", "/api/graph/build"): [ok({"task_id": "t1"})],
        ("GET", "/api/graph/task/t1"): [ok({"status": "processing", "progress": 40, "message": "extracting"}),
                                         ok({"status": "completed", "progress": 100, "result": {"graph_id": "g1"}})],
        ("GET", "/api/graph/data/g1"): [ok({"nodes": [{}] * 9, "edges": [{}] * 12, "node_count": 9, "edge_count": 12})],
    })
    logged = []
    mf.log = logged.append
    assert mf.build_graph("proj_1") == "g1"
    assert logged[-1] == "graph g1: 9 nodes, 12 edges"
    body = mf.http.calls[0][2]["json"]
    assert body == {"project_id": "proj_1", "chunk_size": 500, "chunk_overlap": 50}


def test_a_failed_graph_task_raises_with_its_error(monkeypatch):
    monkeypatch.setattr(md.time, "sleep", lambda _s: None)
    mf = driver({("POST", "/api/graph/build"): [ok({"task_id": "t1"})],
                 ("GET", "/api/graph/task/t1"): [ok({"status": "failed", "error": "LLM returned garbage"})]})
    with pytest.raises(RuntimeError, match="garbage"):
        mf.build_graph("proj_1")


def test_prepare_waits_for_the_task_and_reports_the_agent_count(monkeypatch):
    monkeypatch.setattr(md.time, "sleep", lambda _s: None)
    mf = driver({
        ("POST", "/api/simulation/create"): [ok({"simulation_id": "sim_1"})],
        ("POST", "/api/simulation/prepare"): [ok({"task_id": "t2", "status": "preparing"})],
        ("POST", "/api/simulation/prepare/status"): [ok({"status": "processing", "progress": 30}), ok({"status": "completed", "progress": 100})],
        ("GET", "/api/simulation/sim_1/config"): [ok({"agent_configs": [{}, {}, {}]})],
    })
    assert mf.create_and_prepare("proj_1", "g1") == "sim_1"


def test_an_already_prepared_simulation_is_not_waited_for():
    mf = driver({
        ("POST", "/api/simulation/create"): [ok({"simulation_id": "sim_1"})],
        ("POST", "/api/simulation/prepare"): [ok({"already_prepared": True})],
        ("GET", "/api/simulation/sim_1/config"): [ok({"agent_configs": []})],
    })
    assert mf.create_and_prepare("proj_1", "g1") == "sim_1"


def test_run_returns_when_finished_and_raises_when_it_really_failed(monkeypatch):
    monkeypatch.setattr(md.time, "sleep", lambda _s: None)
    running = ok({"runner_status": "running", "current_round": 3, "total_rounds": 24})
    mf = driver({("POST", "/api/simulation/start"): [ok({})],
                 ("GET", "/api/simulation/sim_1/run-status"): [running, ok({"runner_status": "completed", "current_round": 24, "total_rounds": 24})]})
    assert mf.run("sim_1", max_rounds=24, seed=7)["runner_status"] == "completed"
    sent = mf.http.calls[0][2]["json"]
    assert sent["max_rounds"] == 24 and sent["seed"] == 7 and sent["enable_graph_memory_update"] is False and sent["force"] is True

    failed = driver({("POST", "/api/simulation/start"): [ok({})],
                     ("GET", "/api/simulation/sim_1/run-status"): [ok({"runner_status": "failed", "error": "process exit 3"})]})
    with pytest.raises(RuntimeError, match="exit 3"):
        failed.run("sim_1")


def test_a_run_that_only_waits_on_the_graph_sync_counts_as_finished(monkeypatch):
    mf = driver({("POST", "/api/simulation/start"): [ok({})],
                 ("GET", "/api/simulation/sim_1/run-status"): [ok({"runner_status": "failed", "graph_sync_pending": True, "current_round": 24})]})
    assert mf.run("sim_1")["graph_sync_pending"] is True


def test_report_waits_then_downloads_the_markdown(monkeypatch):
    monkeypatch.setattr(md.time, "sleep", lambda _s: None)
    mf = driver({
        ("POST", "/api/report/generate"): [ok({"report_id": "report_1", "task_id": "t3"})],
        ("POST", "/api/report/generate/status"): [ok({"status": "processing", "progress": 50}), ok({"status": "completed", "progress": 100})],
        ("GET", "/api/report/report_1/download"): [Response(text="# Report\n\nBody")],
    })
    assert mf.report("sim_1", allow_partial_graph=True) == "# Report\n\nBody"
    assert mf.http.calls[0][2]["json"] == {"simulation_id": "sim_1", "force_regenerate": True, "allow_partial_graph": True}
