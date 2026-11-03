"""Drives a MiroFish backend over its HTTP API, step by step, with short progress lines.

This is what the web UI does, minus the browser: upload a seed and a prompt, get an ontology, build the knowledge
graph, create and prepare a simulation, run it, and write the report. The notebook calls one method per cell, so a
failure points at one stage and the next cell can be re-run after fixing the cause.

Only needs ``requests``. Every method returns plain ids/strings and raises ``RuntimeError`` with the backend's own
error text when a stage fails.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from graph_view import draw_graph, graph_html, graph_summary

TERMINAL_RUN = {"completed", "stopped", "failed"}


class MiroFish:
    def __init__(self, base_url: str = "http://127.0.0.1:5001", log: Callable[[str], None] = print) -> None:
        self.base = base_url.rstrip("/")
        self.log = log
        self.http = requests.Session()

    # -- plumbing --------------------------------------------------------------------------------

    def _call(self, method: str, path: str, *, timeout: float = 120, **kwargs: Any) -> Dict[str, Any]:
        try:
            response = self.http.request(method, self.base + path, timeout=timeout, **kwargs)
        except requests.RequestException as error:
            raise RuntimeError(f"{method} {path}: could not reach the backend ({error})") from error
        try:
            body = response.json()
        except ValueError:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code}, not JSON: {response.text[:300]}")
        if not body.get("success", response.ok):
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code}: {body.get('error') or body}")
        return body.get("data") if isinstance(body.get("data"), dict) or body.get("data") is not None else body

    def health(self) -> bool:
        try:
            return self.http.get(self.base + "/health", timeout=5).ok
        except requests.RequestException:
            return False

    def wait_for_backend(self, timeout: float = 180) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.health():
                return
            time.sleep(2)
        raise RuntimeError("the backend did not come up; read /content/logs/backend.log")

    # -- 1. ontology ---------------------------------------------------------------------------------

    def generate_ontology(self, seed_files: List[str], requirement: str, project_name: str = "MiroFish run") -> str:
        """Upload the seed document(s) and the prompt; the model designs the entity and relationship types."""
        if not seed_files:
            raise RuntimeError("no seed file: upload at least one .txt / .md / .pdf")
        handles = [open(p, "rb") for p in seed_files]
        try:
            files = [("files", (os.path.basename(p), h)) for p, h in zip(seed_files, handles)]
            data = self._call(
                "POST", "/api/graph/ontology/generate", timeout=900, files=files,
                data={"simulation_requirement": requirement, "project_name": project_name},
            )
        finally:
            for h in handles:
                h.close()
        ontology = data.get("ontology") or {}
        self.log(f"ontology: {len(ontology.get('entity_types', []))} entity types, "
                 f"{len(ontology.get('edge_types', []))} relationship types, {data.get('total_text_length')} characters of seed")
        return data["project_id"]

    # -- 2. knowledge graph -----------------------------------------------------------------------------

    def build_graph(self, project_id: str, chunk_size: int = 500, chunk_overlap: int = 50, timeout: float = 7200) -> str:
        """Chunk the seed and have Graphiti extract entities and facts from every chunk (the slow, LLM-heavy stage)."""
        task = self._call("POST", "/api/graph/build", json={"project_id": project_id, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
        result = self._wait_task(f"/api/graph/task/{task['task_id']}", "graph", timeout)
        graph_id = (result.get("result") or {}).get("graph_id")
        if not graph_id:
            graph_id = self._call("GET", f"/api/graph/project/{project_id}").get("graph_id")
        try:
            data = self._call("GET", f"/api/graph/data/{graph_id}", timeout=300)
            counts = f"{data.get('node_count', len(data.get('nodes', [])))} nodes, {data.get('edge_count', len(data.get('edges', [])))} edges"
        except RuntimeError:
            counts = "node and edge counts unavailable"
        self.log(f"graph {graph_id}: {counts}")
        return graph_id

    def graph_data(self, graph_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/api/graph/data/{graph_id}", timeout=300)

    def show_graph(self, graph_id: str, out_dir: Optional[str] = None, facts: int = 25, max_drawn: int = 150) -> Optional[str]:
        """Look at the knowledge graph: a summary, the facts and a picture. With ``out_dir`` it also saves ``graph.html``
        (interactive, open it in a browser) and ``graph.json`` there."""
        data = self.graph_data(graph_id)
        print(graph_summary(data, facts))
        try:
            draw_graph(data, max_drawn)
        except ImportError:
            print("(no picture: matplotlib / networkx are not installed here)")
        if not out_dir:
            return None
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "graph.html")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(graph_html(data))
        with open(os.path.join(out_dir, "graph.json"), "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1, default=str)
        print("saved", path, "(open it in a browser for the interactive graph)")
        return path

    def _wait_task(self, path: str, label: str, timeout: float, poll: float = 5) -> Dict[str, Any]:
        deadline, last = time.time() + timeout, None
        while time.time() < deadline:
            task = self._call("GET", path)
            line = f"{label}: {task.get('status')} {task.get('progress', 0)}% {task.get('message', '')}"
            if line != last:
                self.log(line)
                last = line
            if task.get("status") == "completed":
                return task
            if task.get("status") == "failed":
                raise RuntimeError(f"{label} failed: {str(task.get('error') or task.get('message'))[:800]}")
            time.sleep(poll)
        raise RuntimeError(f"{label} did not finish within {int(timeout)} s")

    # -- 3. simulation ------------------------------------------------------------------------------------

    def create_and_prepare(self, project_id: str, graph_id: str, timeout: float = 3600) -> str:
        """Create the simulation, then generate every agent's persona and the simulation config."""
        simulation_id = self._call("POST", "/api/simulation/create", json={"project_id": project_id, "graph_id": graph_id})["simulation_id"]
        prepared = self._call("POST", "/api/simulation/prepare", timeout=300,
                              json={"simulation_id": simulation_id, "use_llm_for_profiles": True, "parallel_profile_count": 3})
        if not prepared.get("already_prepared") and prepared.get("task_id"):
            deadline, last = time.time() + timeout, None
            while time.time() < deadline:
                status = self._call("POST", "/api/simulation/prepare/status", json={"task_id": prepared["task_id"], "simulation_id": simulation_id})
                line = f"prepare: {status.get('status')} {status.get('progress', 0)}% {status.get('message', '')}"
                if line != last:
                    self.log(line)
                    last = line
                if status.get("status") in ("completed", "ready") or status.get("already_prepared"):
                    break
                if status.get("status") == "failed":
                    raise RuntimeError(f"prepare failed: {str(status.get('error') or status.get('message'))[:800]}")
                time.sleep(5)
            else:
                raise RuntimeError(f"prepare did not finish within {int(timeout)} s")
        config = self._call("GET", f"/api/simulation/{simulation_id}/config")
        agents = len((config.get("config") or config).get("agent_configs", [])) if isinstance(config, dict) else "?"
        self.log(f"simulation {simulation_id} ready: {agents} agents")
        return simulation_id

    def run(self, simulation_id: str, max_rounds: Optional[int] = None, live_graph_updates: bool = False,
            seed: Optional[int] = None, timeout: float = 4 * 3600, poll: float = 15) -> Dict[str, Any]:
        """Run the social simulation (Twitter-like and Reddit-like at once) and wait for it to end."""
        body: Dict[str, Any] = {"simulation_id": simulation_id, "platform": "parallel",
                                "enable_graph_memory_update": bool(live_graph_updates), "force": True}
        if max_rounds:
            body["max_rounds"] = int(max_rounds)
        if seed is not None:
            body["seed"] = int(seed)
        self._call("POST", "/api/simulation/start", timeout=300, json=body)
        deadline, last = time.time() + timeout, None
        while time.time() < deadline:
            state = self._call("GET", f"/api/simulation/{simulation_id}/run-status")
            line = (f"run: {state.get('runner_status')} round {state.get('current_round')}/{state.get('total_rounds')} "
                    f"(twitter {state.get('twitter_actions_count', 0)} / reddit {state.get('reddit_actions_count', 0)} actions)")
            if line != last:
                self.log(line)
                last = line
            if state.get("runner_status") in TERMINAL_RUN or state.get("graph_sync_pending"):
                if state.get("runner_status") == "failed" and not state.get("graph_sync_pending"):
                    raise RuntimeError(f"the simulation failed: {str(state.get('error'))[:800]}")
                return state
            time.sleep(poll)
        raise RuntimeError(f"the simulation did not finish within {int(timeout)} s (it is still running; call run_status())")

    def run_status(self, simulation_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/api/simulation/{simulation_id}/run-status")

    # -- 4. report ---------------------------------------------------------------------------------------------

    def report(self, simulation_id: str, allow_partial_graph: bool = False, timeout: float = 5400) -> str:
        """Have the report agent write the report; returns its Markdown."""
        body = {"simulation_id": simulation_id, "force_regenerate": True}
        if allow_partial_graph:
            body["allow_partial_graph"] = True
        started = self._call("POST", "/api/report/generate", timeout=300, json=body)
        report_id = started.get("report_id")
        deadline, last = time.time() + timeout, None
        while time.time() < deadline:
            status = self._call("POST", "/api/report/generate/status", json={"task_id": started.get("task_id"), "simulation_id": simulation_id})
            line = f"report: {status.get('status')} {status.get('progress', 0)}% {status.get('message', '')}"
            if line != last:
                self.log(line)
                last = line
            if status.get("status") == "completed":
                break
            if status.get("status") == "failed":
                raise RuntimeError(f"report failed: {str(status.get('error') or status.get('message'))[:800]}")
            time.sleep(10)
        else:
            raise RuntimeError(f"the report did not finish within {int(timeout)} s")
        response = self.http.get(f"{self.base}/api/report/{report_id}/download", timeout=120)
        response.raise_for_status()
        return response.text
