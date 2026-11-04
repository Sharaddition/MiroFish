from types import SimpleNamespace

from app.services import simulation_runner as sr
from app.services.simulation_runner import SimulationRunner


def run(monkeypatch, *, updater, live=False, status="failed"):
    state = SimpleNamespace(runner_status=sr.RunnerStatus.FAILED if status == "failed" else sr.RunnerStatus.STOPPED,
                            total_rounds=80, seed=1, platforms=[])
    monkeypatch.setattr(SimulationRunner, "get_run_state", classmethod(lambda cls, _id: state))
    monkeypatch.setattr(SimulationRunner, "has_live_process", classmethod(lambda cls, _id: live))
    monkeypatch.setattr(sr.ZepGraphMemoryManager, "get_updater", classmethod(lambda cls, _id: object() if updater else None))
    return SimulationRunner.resume_info("sim_x")


def test_a_run_whose_zep_writer_is_still_open_is_marked_finishing(monkeypatch):
    info = run(monkeypatch, updater=True)
    assert info["resumable"] is False and info["finishing"] is True
    assert info["reason"] == "the run is still finishing"


def test_other_blocked_or_resumable_runs_are_not_finishing(monkeypatch):
    assert run(monkeypatch, updater=False, live=True)["finishing"] is False
    monkeypatch.setattr(SimulationRunner, "get_run_state", classmethod(
        lambda cls, _id: SimpleNamespace(runner_status=sr.RunnerStatus.FAILED, total_rounds=80, seed=None, platforms=[])))
    assert SimulationRunner.resume_info("sim_x")["finishing"] is False      # blocked for another reason: no seed
