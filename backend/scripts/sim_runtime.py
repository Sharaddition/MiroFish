"""
OASIS-facing helpers shared by the simulation run scripts.

``sim_behavior`` holds the pure decisions (who acts, who follows whom, which
events fire). This module executes them against an OASIS environment: seeding
the follow graph, injecting scheduled events and interviewing agents (used by
both the IPC command loop and the end-of-run poll).

``oasis`` is imported lazily, inside the functions that need its action classes,
so the module loads (and can be unit tested with fakes) without the simulation
stack installed.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


def _action_classes():
    """``(ActionType, ManualAction)`` from OASIS (patched in tests)."""
    from oasis import ActionType, ManualAction

    return ActionType, ManualAction


# --- database helpers --------------------------------------------------------

def max_trace_rowid(db_path: str) -> int:
    """Highest rowid in the platform's ``trace`` table (0 when empty/missing).

    The run scripts log agent actions by reading new ``trace`` rows after each
    round. Advancing their cursor to this value after a manual step (seeded
    follows, injected events, initial posts) keeps those rows from being logged
    a second time as ordinary round actions.
    """
    if not os.path.exists(db_path):
        return 0
    try:
        connection = sqlite3.connect(db_path)
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM trace"
            ).fetchone()
        finally:
            connection.close()
        return int(row[0])
    except sqlite3.Error:
        return 0


def read_interview_result(
    db_path: str, agent_id: int, after_rowid: int = 0
) -> Dict[str, Any]:
    """The agent's latest interview recorded in ``trace`` after ``after_rowid``.

    Returns ``{"agent_id", "response", "timestamp"}``; ``response`` is None when
    no interview row exists yet.
    """
    ActionType, _ = _action_classes()
    result: Dict[str, Any] = {"agent_id": agent_id, "response": None, "timestamp": None}
    if not os.path.exists(db_path):
        return result

    try:
        connection = sqlite3.connect(db_path)
        try:
            row = connection.execute(
                """
                SELECT info, created_at FROM trace
                WHERE action = ? AND user_id = ? AND rowid > ?
                ORDER BY rowid DESC LIMIT 1
                """,
                (ActionType.INTERVIEW.value, agent_id, after_rowid),
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as error:
        print(f"  读取Interview结果失败: {error}")
        return result

    if row:
        info_json, created_at = row
        try:
            info = json.loads(info_json) if info_json else {}
            result["response"] = info.get("response", info)
        except (json.JSONDecodeError, AttributeError):
            result["response"] = info_json
        result["timestamp"] = created_at
    return result


# --- interviews --------------------------------------------------------------

async def batch_interview(
    env,
    agent_graph,
    db_path: str,
    prompts: Mapping[int, str],
    *,
    chunk_size: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> Dict[int, Dict[str, Any]]:
    """Interview several agents, returning ``{agent_id: result}``.

    Every prompt is sent as an ``INTERVIEW`` ``ManualAction``; agents in one
    chunk are interviewed concurrently by a single ``env.step`` (a chunk
    defaults to all agents). With a smaller ``chunk_size`` one failing step only
    costs that chunk. Each result is read back from the ``trace`` rows written
    *after* the step started, so an older interview of the same agent is never
    returned by mistake. An agent that could not be interviewed gets a result
    with ``response`` None and an ``error`` message.
    """
    ActionType, ManualAction = _action_classes()
    agent_ids = list(prompts)
    size = chunk_size or max(1, len(agent_ids))
    results: Dict[int, Dict[str, Any]] = {}

    for start in range(0, len(agent_ids), size):
        chunk = agent_ids[start:start + size]
        before = max_trace_rowid(db_path)
        actions = {}
        for agent_id in chunk:
            try:
                agent = agent_graph.get_agent(agent_id)
            except Exception as error:
                log(f"  警告: 无法获取Agent {agent_id}: {error}")
                results[agent_id] = {
                    "agent_id": agent_id,
                    "response": None,
                    "timestamp": None,
                    "error": f"unknown agent: {error}",
                    "unknown_agent": True,
                }
                continue
            actions[agent] = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompts[agent_id]},
            )
        if not actions:
            continue

        step_error: Optional[str] = None
        try:
            await env.step(actions)
        except Exception as error:
            step_error = str(error)
            log(f"  Interview步骤失败: {error}")

        for agent_id in chunk:
            if agent_id in results:
                continue
            result = read_interview_result(db_path, agent_id, after_rowid=before)
            if result["response"] is None and step_error:
                result["error"] = step_error
            results[agent_id] = result
    return results


# --- manual steps: seeded follows and scheduled events -----------------------

async def apply_seed_follows(
    env, follows: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Execute the planned ``(follower, followee)`` pairs in a single step.

    Returns the pairs that were submitted (pairs whose follower is not in the
    agent graph are skipped).
    """
    ActionType, ManualAction = _action_classes()
    per_agent: Dict[Any, List[Any]] = {}
    submitted: List[Tuple[int, int]] = []
    for follower_id, followee_id in follows:
        try:
            follower = env.agent_graph.get_agent(follower_id)
        except Exception:
            continue
        per_agent.setdefault(follower, []).append(
            ManualAction(
                action_type=ActionType.FOLLOW,
                action_args={"followee_id": followee_id},
            )
        )
        submitted.append((follower_id, followee_id))

    if per_agent:
        await env.step(
            {
                agent: (actions[0] if len(actions) == 1 else actions)
                for agent, actions in per_agent.items()
            }
        )
    return submitted


async def inject_scheduled_events(
    env,
    events: Sequence[Mapping[str, Any]],
    *,
    log_round: int,
    agent_names: Mapping[int, str],
    action_logger=None,
    log: Callable[[str], None] = print,
) -> List[Mapping[str, Any]]:
    """Post each due scheduled event as its ``poster_agent_id``.

    Events whose poster is not in the agent graph are skipped (and reported
    through ``log``). Each posted event is written to ``actions.jsonl`` as a
    ``CREATE_POST`` with ``phase: "injected"`` and its ``event_id``. Returns the
    events that were posted.
    """
    ActionType, ManualAction = _action_classes()
    per_agent: Dict[Any, List[Any]] = {}
    posted: List[Mapping[str, Any]] = []
    for event in events:
        poster_id = event.get("poster_agent_id")
        content = event.get("content", "")
        try:
            poster = env.agent_graph.get_agent(poster_id)
        except Exception:
            log(f"  警告: 定时事件 {event.get('id')} 的发布者 Agent {poster_id} 不存在，已跳过")
            continue
        per_agent.setdefault(poster, []).append(
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": content},
            )
        )
        posted.append(event)

    if per_agent:
        await env.step(
            {
                agent: (actions[0] if len(actions) == 1 else actions)
                for agent, actions in per_agent.items()
            }
        )

    if action_logger is not None:
        for event in posted:
            poster_id = event.get("poster_agent_id")
            action_logger.log_action(
                round_num=log_round,
                agent_id=poster_id,
                agent_name=agent_names.get(poster_id, f"Agent_{poster_id}"),
                action_type="CREATE_POST",
                action_args={"content": event.get("content", "")},
                phase="injected",
                extra={"event_id": event.get("id")},
            )
    return posted


# --- the end-of-run outcome poll ---------------------------------------------

async def run_final_poll(
    env,
    agent_graph,
    db_path: str,
    config: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int = 10,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Interview every agent once on the outcome questions.

    All agents get the same JSON-only prompt (``sim_poll.build_poll_prompt``).
    They are interviewed in chunks so one failing request only costs a chunk,
    and agents that produced no reply at all get a single retry. Replies that
    cannot be parsed are kept (``parse_ok: false``) rather than dropped, so the
    ensemble can report a parse rate. Returns the ``final_poll.json`` rows,
    ordered by agent id.
    """
    import sim_poll

    prompt = sim_poll.build_poll_prompt(questions)
    agents = sorted(
        (cfg for cfg in config.get("agent_configs", []) if cfg.get("agent_id") is not None),
        key=lambda cfg: int(cfg["agent_id"]),
    )
    prompts = {int(cfg["agent_id"]): prompt for cfg in agents}

    results = await batch_interview(
        env, agent_graph, db_path, prompts, chunk_size=chunk_size, log=log
    )

    silent = [
        agent_id for agent_id, result in results.items()
        if result.get("response") is None and not result.get("unknown_agent")
    ]
    if silent:
        log(f"  最终问卷: {len(silent)} 个Agent没有回复，重试一次")
        retried = await batch_interview(
            env,
            agent_graph,
            db_path,
            {agent_id: prompt for agent_id in silent},
            chunk_size=max(1, chunk_size // 2),
            log=log,
        )
        for agent_id, result in retried.items():
            if result.get("response") is not None:
                results[agent_id] = result

    rows: List[Dict[str, Any]] = []
    for cfg in agents:
        agent_id = int(cfg["agent_id"])
        result = results.get(agent_id, {})
        response = result.get("response")
        if response is not None and not isinstance(response, str):
            response = json.dumps(response, ensure_ascii=False)
        rows.append(
            sim_poll.poll_row(
                agent_id,
                cfg.get("entity_name", f"Agent_{agent_id}"),
                str(cfg.get("stance", "neutral")),
                response,
                questions,
                error=result.get("error"),
            )
        )
    return rows


def write_final_poll(sim_dir: str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write ``final_poll.json`` (a list of rows) atomically; returns its path."""
    path = os.path.join(sim_dir, "final_poll.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())  # a crash after the rename must not leave a zero-filled file
    os.replace(tmp_path, path)
    return path
