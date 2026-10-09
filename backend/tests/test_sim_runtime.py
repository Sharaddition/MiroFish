import asyncio
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict

import pytest

import sim_runtime


class FakeActionType(str, Enum):
    FOLLOW = "follow"
    CREATE_POST = "create_post"
    INTERVIEW = "interview"


@dataclass
class FakeManualAction:
    action_type: FakeActionType
    action_args: Dict[str, Any]


@pytest.fixture(autouse=True)
def fake_oasis(monkeypatch):
    monkeypatch.setattr(sim_runtime, "_action_classes", lambda: (FakeActionType, FakeManualAction))


class FakeAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def __repr__(self):
        return f"Agent({self.agent_id})"


class FakeGraph:
    def __init__(self, ids):
        self.agents = {i: FakeAgent(i) for i in ids}

    def get_agent(self, agent_id):
        return self.agents[agent_id]


def make_db(path, rows=()):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE trace (user_id INTEGER, created_at TEXT, action TEXT, info TEXT)")
    for user_id, action, info in rows:
        con.execute("INSERT INTO trace VALUES (?, '0', ?, ?)", (user_id, action, info))
    con.commit()
    con.close()


def add_trace(path, user_id, action, info):
    con = sqlite3.connect(path)
    con.execute("INSERT INTO trace VALUES (?, '0', ?, ?)", (user_id, action, info))
    con.commit()
    con.close()


class FakeEnv:
    """Records every step; optionally writes interview rows like OASIS does."""

    def __init__(self, ids, db_path=None, answers=None, fail_on_step=()):
        self.agent_graph = FakeGraph(ids)
        self.db_path = db_path
        self.answers = answers or {}
        self.fail_on_step = set(fail_on_step)
        self.steps = []

    async def step(self, actions):
        index = len(self.steps)
        self.steps.append(actions)
        if index in self.fail_on_step:
            raise RuntimeError("provider exploded")
        for agent, action in actions.items():
            for single in (action if isinstance(action, list) else [action]):
                if single.action_type == FakeActionType.INTERVIEW and self.db_path:
                    answer = self.answers.get(agent.agent_id, f"answer {agent.agent_id}")
                    add_trace(self.db_path, agent.agent_id, "interview",
                              json.dumps({"prompt": single.action_args["prompt"], "response": answer}))


def run(coro):
    return asyncio.run(coro)


# --- database helpers ----------------------------------------------------------

def test_max_trace_rowid(tmp_path):
    db = str(tmp_path / "p.db")
    assert sim_runtime.max_trace_rowid(db) == 0  # missing file
    make_db(db)
    assert sim_runtime.max_trace_rowid(db) == 0  # empty table
    add_trace(db, 1, "sign_up", "{}")
    add_trace(db, 2, "follow", "{}")
    assert sim_runtime.max_trace_rowid(db) == 2
    (tmp_path / "broken.db").write_bytes(b"not a database")
    assert sim_runtime.max_trace_rowid(str(tmp_path / "broken.db")) == 0


def test_read_interview_result_returns_the_latest_row_after_a_cursor(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db, [
        (1, "interview", json.dumps({"prompt": "q", "response": "old"})),
        (2, "interview", json.dumps({"prompt": "q", "response": "other agent"})),
        (1, "like_post", "{}"),
        (1, "interview", json.dumps({"prompt": "q", "response": "newest"})),
    ])
    assert sim_runtime.read_interview_result(db, 1)["response"] == "newest"
    assert sim_runtime.read_interview_result(db, 2)["response"] == "other agent"
    assert sim_runtime.read_interview_result(db, 3) == {"agent_id": 3, "response": None, "timestamp": None}
    # rows at or before the cursor are ignored
    assert sim_runtime.read_interview_result(db, 1, after_rowid=4)["response"] is None
    assert sim_runtime.read_interview_result(db, 1, after_rowid=3)["response"] == "newest"
    assert sim_runtime.read_interview_result(str(tmp_path / "missing.db"), 1)["response"] is None


def test_read_interview_result_tolerates_non_json_info(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db, [(1, "interview", "plain text, not json")])
    assert sim_runtime.read_interview_result(db, 1)["response"] == "plain text, not json"


# --- batch_interview -----------------------------------------------------------

def test_batch_interview_sends_every_prompt_in_one_step_by_default(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv([0, 1, 2], db)
    results = run(sim_runtime.batch_interview(env, env.agent_graph, db, {0: "q0", 2: "q2"}))

    assert len(env.steps) == 1
    sent = {agent.agent_id: action.action_args["prompt"] for agent, action in env.steps[0].items()}
    assert sent == {0: "q0", 2: "q2"}
    assert {i: r["response"] for i, r in results.items()} == {0: "answer 0", 2: "answer 2"}
    assert "error" not in results[0]


def test_batch_interview_ignores_older_interviews_of_the_same_agent(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db, [(0, "interview", json.dumps({"response": "from an earlier interview"}))])
    env = FakeEnv([0], db, fail_on_step={0})  # this step writes nothing
    results = run(sim_runtime.batch_interview(env, env.agent_graph, db, {0: "q"}, log=lambda _m: None))
    assert results[0]["response"] is None
    assert "provider exploded" in results[0]["error"]


def test_batch_interview_chunks_isolate_failures(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv(range(5), db, fail_on_step={1})
    messages = []
    results = run(sim_runtime.batch_interview(
        env, env.agent_graph, db, {i: f"q{i}" for i in range(5)}, chunk_size=2, log=messages.append,
    ))
    assert len(env.steps) == 3
    assert [r["response"] for i, r in sorted(results.items())] == [
        "answer 0", "answer 1", None, None, "answer 4",
    ]
    assert all("provider exploded" in results[i]["error"] for i in (2, 3))
    assert any("Interview步骤失败" in m for m in messages)


def test_batch_interview_flags_unknown_agents_without_sending_them(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv([0], db)
    results = run(sim_runtime.batch_interview(env, env.agent_graph, db, {0: "q", 9: "q"}, log=lambda _m: None))
    assert results[9]["unknown_agent"] is True and results[9]["response"] is None
    assert results[0]["response"] == "answer 0" and "unknown_agent" not in results[0]
    assert [a.agent_id for a in env.steps[0]] == [0]


def test_batch_interview_with_only_unknown_agents_never_steps(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv([0], db)
    results = run(sim_runtime.batch_interview(env, env.agent_graph, db, {7: "q"}, log=lambda _m: None))
    assert env.steps == [] and results[7]["unknown_agent"] is True


def test_batch_interview_of_nobody_does_nothing(tmp_path):
    env = FakeEnv([0])
    assert run(sim_runtime.batch_interview(env, env.agent_graph, str(tmp_path / "p.db"), {})) == {}
    assert env.steps == []


# --- seeded follows ------------------------------------------------------------

def test_apply_seed_follows_groups_actions_per_follower():
    env = FakeEnv([0, 1, 2])
    submitted = run(sim_runtime.apply_seed_follows(env, [(0, 1), (0, 2), (1, 2)]))

    assert submitted == [(0, 1), (0, 2), (1, 2)]
    assert len(env.steps) == 1
    by_agent = {agent.agent_id: action for agent, action in env.steps[0].items()}
    assert [a.action_args["followee_id"] for a in by_agent[0]] == [1, 2]  # several -> a list
    assert by_agent[1].action_args == {"followee_id": 2}  # one -> a single action
    assert all(a.action_type == FakeActionType.FOLLOW for a in by_agent[0])


def test_apply_seed_follows_skips_unknown_followers_and_empty_plans():
    env = FakeEnv([0, 1])
    assert run(sim_runtime.apply_seed_follows(env, [(9, 0), (0, 1)])) == [(0, 1)]
    assert run(sim_runtime.apply_seed_follows(env, [])) == []
    assert run(sim_runtime.apply_seed_follows(env, [(9, 0)])) == []
    assert len(env.steps) == 1  # only the first call stepped


# --- scheduled events ----------------------------------------------------------

class RecordingLogger:
    def __init__(self):
        self.entries = []

    def log_action(self, **kwargs):
        self.entries.append(kwargs)


def test_events_are_posted_by_their_poster_and_logged_as_injected():
    env = FakeEnv([0, 1, 2])
    logger = RecordingLogger()
    events = [
        {"id": "evt_a", "poster_agent_id": 1, "content": "first"},
        {"id": "evt_b", "poster_agent_id": 2, "content": "second"},
    ]
    posted = run(sim_runtime.inject_scheduled_events(
        env, events, log_round=4, agent_names={1: "Media", 2: "Trader"}, action_logger=logger,
    ))

    assert posted == events
    by_agent = {agent.agent_id: action for agent, action in env.steps[0].items()}
    assert by_agent[1].action_type == FakeActionType.CREATE_POST
    assert by_agent[1].action_args == {"content": "first"}
    assert logger.entries == [
        {"round_num": 4, "agent_id": 1, "agent_name": "Media", "action_type": "CREATE_POST",
         "action_args": {"content": "first"}, "phase": "injected", "extra": {"event_id": "evt_a"}},
        {"round_num": 4, "agent_id": 2, "agent_name": "Trader", "action_type": "CREATE_POST",
         "action_args": {"content": "second"}, "phase": "injected", "extra": {"event_id": "evt_b"}},
    ]


def test_two_events_by_the_same_poster_are_both_posted():
    env = FakeEnv([0, 1])
    events = [
        {"id": "a", "poster_agent_id": 1, "content": "one"},
        {"id": "b", "poster_agent_id": 1, "content": "two"},
    ]
    run(sim_runtime.inject_scheduled_events(env, events, log_round=1, agent_names={}))
    (action,) = env.steps[0].values()
    assert [a.action_args["content"] for a in action] == ["one", "two"]


def test_events_with_a_missing_poster_are_skipped_and_reported():
    env = FakeEnv([0])
    logger = RecordingLogger()
    messages = []
    posted = run(sim_runtime.inject_scheduled_events(
        env, [{"id": "ghost", "poster_agent_id": 99, "content": "x"}],
        log_round=2, agent_names={}, action_logger=logger, log=messages.append,
    ))
    assert posted == [] and env.steps == [] and logger.entries == []
    assert any("ghost" in m and "99" in m for m in messages)


def test_events_can_run_without_a_logger():
    env = FakeEnv([0])
    posted = run(sim_runtime.inject_scheduled_events(
        env, [{"id": "a", "poster_agent_id": 0, "content": "x"}], log_round=1, agent_names={},
    ))
    assert len(posted) == 1


# --- the end-of-run outcome poll -------------------------------------------------

POLL_QUESTIONS = [
    {"id": "q1", "text": "Will it trade higher?", "type": "probability"},
    {"id": "q2", "text": "Your stance?", "type": "choice", "options": ["bullish", "neutral", "bearish"]},
]


def poll_config(n=3):
    return {
        "agent_configs": [
            {"agent_id": i, "entity_name": f"Agent {i}", "stance": ["supportive", "opposing", "neutral"][i % 3]}
            for i in range(n)
        ]
    }


def json_answers(ids):
    return {i: json.dumps({"q1": 10 * (i + 1), "q2": ["bullish", "neutral", "bearish"][i % 3], "reason": f"r{i}"}) for i in ids}


def test_final_poll_interviews_every_agent_and_parses_their_answers(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv([0, 1, 2], db, answers=json_answers(range(3)))
    rows = run(sim_runtime.run_final_poll(env, env.agent_graph, db, poll_config(), POLL_QUESTIONS, log=lambda _m: None))

    assert [r["agent_id"] for r in rows] == [0, 1, 2]
    assert all(r["parse_ok"] for r in rows)
    assert rows[1] == {
        "agent_id": 1, "agent_name": "Agent 1", "stance_initial": "opposing",
        "answers": {"q1": 20.0, "q2": "neutral"}, "reason": "r1",
        "raw": json_answers([1])[1], "parse_ok": True,
    }
    prompts = {action.action_args["prompt"] for step in env.steps for action in step.values()}
    assert len(prompts) == 1  # everybody gets the same prompt
    (prompt,) = prompts
    assert "Reply with ONLY a JSON object" in prompt and "q1" in prompt and "q2" in prompt


def test_final_poll_keeps_unparseable_replies_and_flags_them(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    answers = json_answers(range(3))
    answers[1] = "I honestly cannot say."
    env = FakeEnv([0, 1, 2], db, answers=answers)
    rows = run(sim_runtime.run_final_poll(env, env.agent_graph, db, poll_config(), POLL_QUESTIONS, log=lambda _m: None))

    assert [r["parse_ok"] for r in rows] == [True, False, True]
    assert rows[1]["raw"] == "I honestly cannot say." and rows[1]["answers"] == {}
    assert len(env.steps) == 1  # a reply that does not parse is not retried


def test_final_poll_retries_agents_that_never_replied(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv([0, 1, 2], db, answers=json_answers(range(3)), fail_on_step={0})
    rows = run(sim_runtime.run_final_poll(env, env.agent_graph, db, poll_config(), POLL_QUESTIONS, log=lambda _m: None))

    assert len(env.steps) == 2  # the failed step, then one retry
    assert all(r["parse_ok"] for r in rows)
    assert all("error" not in r for r in rows)


def test_final_poll_records_the_error_for_agents_that_stay_silent(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv([0, 1], db, answers=json_answers(range(2)), fail_on_step={0, 1})
    rows = run(sim_runtime.run_final_poll(env, env.agent_graph, db, poll_config(2), POLL_QUESTIONS, log=lambda _m: None))

    assert [r["parse_ok"] for r in rows] == [False, False]
    assert all(r["raw"] is None and "provider exploded" in r["error"] for r in rows)
    assert len(env.steps) == 2  # exactly one retry, no loop


def test_final_poll_chunks_agents(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv(range(5), db, answers=json_answers(range(5)))
    rows = run(sim_runtime.run_final_poll(
        env, env.agent_graph, db, poll_config(5), POLL_QUESTIONS, chunk_size=2, log=lambda _m: None,
    ))
    assert len(env.steps) == 3 and len(rows) == 5 and all(r["parse_ok"] for r in rows)


def test_final_poll_skips_agents_missing_from_the_graph(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)
    env = FakeEnv([0, 1], db, answers=json_answers(range(3)))
    rows = run(sim_runtime.run_final_poll(env, env.agent_graph, db, poll_config(3), POLL_QUESTIONS, log=lambda _m: None))

    assert [r["parse_ok"] for r in rows] == [True, True, False]
    assert "unknown agent" in rows[2]["error"]
    assert len(env.steps) == 1  # nothing was retried for the unknown agent


def test_final_poll_serialises_non_string_responses(tmp_path):
    db = str(tmp_path / "p.db")
    make_db(db)

    class StructuredEnv(FakeEnv):
        async def step(self, actions):
            self.steps.append(actions)
            for agent in actions:
                add_trace(self.db_path, agent.agent_id, "interview",
                          json.dumps({"prompt": "p", "response": {"q1": 55, "q2": "neutral"}}))

    env = StructuredEnv([0], db)
    rows = run(sim_runtime.run_final_poll(env, env.agent_graph, db, poll_config(1), POLL_QUESTIONS, log=lambda _m: None))
    assert rows[0]["parse_ok"] is True and rows[0]["answers"] == {"q1": 55.0, "q2": "neutral"}


def test_write_final_poll_writes_a_list_atomically(tmp_path):
    rows = [{"agent_id": 0, "answers": {"q1": 1.0}, "parse_ok": True}]
    path = sim_runtime.write_final_poll(str(tmp_path), rows)
    assert path == str(tmp_path / "final_poll.json")
    assert json.loads((tmp_path / "final_poll.json").read_text(encoding="utf-8")) == rows
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
    sim_runtime.write_final_poll(str(tmp_path), [])  # overwrites
    assert json.loads((tmp_path / "final_poll.json").read_text(encoding="utf-8")) == []
