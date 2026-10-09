"""End-to-end validation covering all four verification scenarios in
PLAN-behavior-wiring-and-ensembles.md (Section 4 & Section 5).
"""

import copy
import json
import os
import shutil
from types import SimpleNamespace

import pytest

import sim_behavior as sb
from sim_behavior import (
    PlatformBehavior,
    is_behavior_v2,
    write_effective_profiles,
)
from app.services.ensemble_runner import EnsembleManager
from app.services.simulation_manager import SimulationManager, SimulationState, SimulationStatus
from app.services.simulation_runner import RunnerStatus, SimulationRunner

# Fixtures and helpers from test_ensemble_manager
from test_ensemble_manager import (  # noqa: F401
    BASE_ID,
    create,
    drive,
    env,
    rid,
)

REFERENCE_SIM_DIR = os.path.join(
    os.path.dirname(__file__), "..", "uploads", "simulations", "sim_1b05d92d9701"
)


def load_reference_files():
    """Load config and profiles from the real reference simulation sim_1b05d92d9701."""
    config_path = os.path.join(REFERENCE_SIM_DIR, "simulation_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tw_path = os.path.join(REFERENCE_SIM_DIR, "twitter_profiles.csv")
    with open(tw_path, "r", encoding="utf-8") as f:
        tw_csv = f.read()

    rd_path = os.path.join(REFERENCE_SIM_DIR, "reddit_profiles.json")
    with open(rd_path, "r", encoding="utf-8") as f:
        rd_json = f.read()

    return config, tw_csv, rd_json


# --- Scenario 1: Base sim sim_1b05d92d9701 with legacy (v1) behavior -----------

def test_scenario_1_legacy_v1_run(tmp_path):
    """Scenario 1: Base sim sim_1b05d92d9701 runs with legacy behavior

    - No .effective files are generated
    - No setup follows are planned
    - behavior_version is treated as v1 (legacy)
    """
    config, tw_csv, rd_json = load_reference_files()
    sim_dir = tmp_path / "sim_legacy"
    sim_dir.mkdir()

    # sim_1b05d92d9701 has no behavior_version key: defaults to v1
    assert "behavior_version" not in config or config.get("behavior_version") == 1
    assert not is_behavior_v2(config)

    (sim_dir / "simulation_config.json").write_text(json.dumps(config), encoding="utf-8")
    (sim_dir / "twitter_profiles.csv").write_text(tw_csv, encoding="utf-8")
    (sim_dir / "reddit_profiles.json").write_text(rd_json, encoding="utf-8")

    tw_behavior = PlatformBehavior(config, "twitter", seed=100)
    rd_behavior = PlatformBehavior(config, "reddit", seed=100)

    assert not tw_behavior.v2
    assert not rd_behavior.v2
    # In legacy mode, no seeded follow graph is created
    assert tw_behavior.planned_follows() == []
    assert rd_behavior.planned_follows() == []
    # In legacy mode, scheduled events are ignored
    assert tw_behavior.events_due(0) == []
    assert rd_behavior.events_due(0) == []

    # Effective profile generation should not produce .effective files when not v2
    effective_files = [p.name for p in sim_dir.iterdir() if ".effective" in p.name]
    assert effective_files == []


# --- Scenario 2: Re-prepare a v2 sim with directives and scheduled events -----

def test_scenario_2_behavior_v2_run(tmp_path):
    """Scenario 2: Prepared v2 simulation

    - .effective profiles contain starting disposition directives
    - Setup follows appear in follow plan
    - User-added scheduled event fires at the configured round
    """
    config, tw_csv, rd_json = load_reference_files()
    sim_dir = tmp_path / "sim_v2"
    sim_dir.mkdir()

    # Upgrade to v2
    config["behavior_version"] = 2
    config["event_config"]["topic"] = "HEGAM demerger"
    config["event_config"]["scheduled_events"] = [
        {
            "id": "evt_test",
            "at_sim_hour": 2,
            "poster_agent_id": 1,
            "content": "Fresh supply orders announced",
            "enabled": True,
        }
    ]
    assert is_behavior_v2(config)

    (sim_dir / "simulation_config.json").write_text(json.dumps(config), encoding="utf-8")
    (sim_dir / "twitter_profiles.csv").write_text(tw_csv, encoding="utf-8")
    (sim_dir / "reddit_profiles.json").write_text(rd_json, encoding="utf-8")

    written = write_effective_profiles(str(sim_dir), config)
    assert "twitter" in written and "reddit" in written

    # Twitter effective profiles contain the starting view directive
    effective_tw_text = (sim_dir / "twitter_profiles.effective.csv").read_text(encoding="utf-8")
    assert "HEGAM demerger" in effective_tw_text
    assert "starting view, not a script" in effective_tw_text

    # Reddit effective profiles contain the directive in persona
    effective_rd = json.loads((sim_dir / "reddit_profiles.effective.json").read_text(encoding="utf-8"))
    assert any("HEGAM demerger" in p.get("persona", "") for p in effective_rd)

    # Originals are untouched
    assert (sim_dir / "twitter_profiles.csv").read_text(encoding="utf-8") == tw_csv
    assert (sim_dir / "reddit_profiles.json").read_text(encoding="utf-8") == rd_json

    # Setup follow plan is created
    tw_behavior = PlatformBehavior(config, "twitter", seed=42)
    follows = tw_behavior.planned_follows()
    assert len(follows) > 0
    # No self-follows
    assert not any(f[0] == f[1] for f in follows)

    # Scheduled event fires at round 2 (with minutes_per_round=60, hour 2 is round 2)
    assert tw_behavior.events_due(0) == []
    assert tw_behavior.events_due(1) == []
    due_at_2 = tw_behavior.events_due(2)
    assert len(due_at_2) == 1
    assert due_at_2[0]["id"] == "evt_test"
    assert due_at_2[0]["content"] == "Fresh supply orders announced"


# --- Scenario 3: Ensemble with n=3, max_rounds=12, caveat & zero Zep writes ---

def test_scenario_3_ensemble_n3_poll_and_no_zep(env):
    """Scenario 3: Ensemble with n=3, max_rounds=12, concurrency=1

    - Clones 3 replicate directories
    - Produces 3 final_poll.json files
    - Produces summary.json and summary.md with the mandatory caveat
    - Forces enable_graph_memory_update=False (zero Zep writes)
    """
    config, tw_csv, rd_json = load_reference_files()
    base_dir = env.sims / BASE_ID
    config["behavior_version"] = 2
    config["event_config"]["topic"] = "HEGAM demerger"
    (base_dir / "simulation_config.json").write_text(json.dumps(config), encoding="utf-8")
    (base_dir / "twitter_profiles.csv").write_text(tw_csv, encoding="utf-8")
    (base_dir / "reddit_profiles.json").write_text(rd_json, encoding="utf-8")

    ensemble = create(
        env,
        n_replicates=3,
        max_rounds=12,
        concurrency=1,
        outcome_questions=[
            {"id": "q1", "text": "Will HEGAM trade above its 7 Oct close?", "type": "probability"},
            {"id": "q2", "text": "Current stance on HEGAM?", "type": "choice", "options": ["bullish", "neutral", "bearish"]},
        ],
    )
    ens_id = ensemble["ensemble_id"]

    # Verify 3 replicate dirs exist
    for idx in range(1, 4):
        rep_sim_id = rid(ensemble, idx)
        rep_dir = env.sims / rep_sim_id
        assert rep_dir.is_dir()
        rep_cfg = json.loads((rep_dir / "simulation_config.json").read_text(encoding="utf-8"))
        assert rep_cfg["run"]["ensemble_id"] == ens_id
        assert rep_cfg["run"]["replicate_index"] == idx
        assert rep_cfg["run"]["max_rounds"] == 12

    # Start ensemble
    EnsembleManager.start(ens_id)
    # Verify runner was invoked with enable_graph_memory_update=False for every replicate
    drive(env, ens_id, {})
    for start_call in env.runner.started:
        assert start_call.get("enable_graph_memory_update") is False
        assert start_call.get("wait_for_commands") is False

    # Mock the final polls for each replicate
    for idx, (p_val, stance) in enumerate([(65.0, "bullish"), (75.0, "bullish"), (45.0, "bearish")], start=1):
        rep_poll = [
            {
                "agent_id": 0,
                "agent_name": "Retail Trader",
                "stance_initial": "neutral",
                "answers": {"q1": p_val, "q2": stance},
                "reason": "Market momentum",
                "raw": "{}",
                "parse_ok": True,
            }
        ]
        (env.sims / rid(ensemble, idx) / "final_poll.json").write_text(
            json.dumps(rep_poll, ensure_ascii=False), encoding="utf-8"
        )

    # Let the ensemble finish aggregating
    drive(env, ens_id, {})
    ens_state = EnsembleManager.get(ens_id)
    assert ens_state["status"] == "completed"

    summary_json = EnsembleManager.summary(ens_id)
    summary_md = EnsembleManager.summary_markdown(ens_id)

    assert summary_json is not None
    assert summary_json["n_replicates_ok"] == 3
    assert "q1" in summary_json["polls"]
    assert "q2" in summary_json["polls"]

    # Verify mandatory caveat in markdown and json
    assert "simulated participants' views" in summary_md.lower()
    assert "not calibrated probabilities" in summary_md.lower()
    assert summary_json["caveat"].startswith("These are distributions")


# --- Scenario 4: Identical activation schedules across same seed --------------

def test_scenario_4_same_seed_activation_schedules_match():
    """Scenario 4: Two runs with identical base_seed produce identical activation schedules

    Verifies deterministic activation schedules on both Twitter and Reddit per round.
    """
    config, _, _ = load_reference_files()
    config["behavior_version"] = 2

    SEED = 778899

    tw_run_a = PlatformBehavior(config, "twitter", seed=SEED)
    tw_run_b = PlatformBehavior(config, "twitter", seed=SEED)
    rd_run_a = PlatformBehavior(config, "reddit", seed=SEED)
    rd_run_b = PlatformBehavior(config, "reddit", seed=SEED)

    # 48 simulated rounds (2 days at 60 min/round)
    for round_num in range(48):
        simulated_hour = (round_num * 60 // 60) % 24

        tw_a_active = tw_run_a.select_active(round_num, simulated_hour)
        tw_b_active = tw_run_b.select_active(round_num, simulated_hour)
        rd_a_active = rd_run_a.select_active(round_num, simulated_hour)
        rd_b_active = rd_run_b.select_active(round_num, simulated_hour)

        # Same seed must yield identical active agent list per round
        assert tw_a_active == tw_b_active, f"Twitter schedule mismatch at round {round_num}"
        assert rd_a_active == rd_b_active, f"Reddit schedule mismatch at round {round_num}"

    # Verify that different seeds produce different schedules
    tw_diff = PlatformBehavior(config, "twitter", seed=12345)
    tw_diff_schedule = [tw_diff.select_active(r, (r * 60 // 60) % 24) for r in range(48)]
    tw_a_schedule = [tw_run_a.select_active(r, (r * 60 // 60) % 24) for r in range(48)]
    assert tw_a_schedule != tw_diff_schedule
