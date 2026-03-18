
from __future__ import annotations

from BottomUpAgent.common.BottomUpAgent import BottomUpAgent
from BottomUpAgent.common.gui_health import run_gui_health_check


class _HealthyObserver:
    def observe(self, step_id: int = 0, phase: str = "healthcheck"):
        return {
            "scene_type": "title_main",
            "screen_image": f"{phase}_{step_id}.png",
        }


class _FailingObserver:
    def observe(self, step_id: int = 0, phase: str = "healthcheck"):
        raise RuntimeError(f"capture failed at {phase}:{step_id}")


class _BrokenEye:
    def observe(self, step_id: int = 0, phase: str = "before"):
        raise RuntimeError("broken eye")


def _config() -> dict:
    return {
        "runtime": {
            "max_steps": 1,
            "stop_on_failures": 3,
            "pause_on_repeated_observe_failures": True,
            "max_observe_failures": 2,
        },
        "model": {"provider": "mock", "name": "demo-model"},
        "environment": {"name": "demo-environment", "window_name": "Demo Window"},
        "visualization": {"enabled": False},
        "_runtime_context": {},
        "_project_root": ".",
    }


def main() -> int:
    ok_report = run_gui_health_check(_HealthyObserver(), capture_count=2)
    assert ok_report.ok is True
    assert ok_report.successful_captures == 2
    assert ok_report.failed_captures == 0

    fail_report = run_gui_health_check(_FailingObserver(), capture_count=2)
    assert fail_report.ok is False
    assert fail_report.failed_captures == 2

    agent = BottomUpAgent(_config())
    agent.eye = _BrokenEye()

    state1 = agent._collect_state(step_id=1, phase="before")
    assert state1["observe_ok"] is False
    assert state1["observe_failure_streak"] == 1

    state2 = agent._collect_state(step_id=2, phase="before")
    assert state2["observe_ok"] is False
    assert state2["observe_failure_streak"] == 2

    step_record = {
        "action_data": {"action_type": "wait"},
        "feedback_data": {"execute_status": "success", "after_scene": "unknown"},
    }
    assert agent._should_stop(step_record) is True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())