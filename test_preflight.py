
from __future__ import annotations

from BottomUpAgent.common.preflight import run_preflight_checks


def _base_config() -> dict:
    return {
        "runtime": {
            "dry_run": True,
        },
        "environment": {
            "capture_mode": "window",
            "window_name": "Slay the Spire 2",
        },
        "model": {
            "provider": "mock",
        },
    }


def main() -> int:
    dry_run_report = run_preflight_checks(_base_config())
    assert dry_run_report.ok, dry_run_report.to_dict()
    assert any(item["name"] == "window_target" for item in dry_run_report.checks)
    assert any(item["status"] == "warning" for item in dry_run_report.checks), dry_run_report.to_dict()

    strict_config = _base_config()
    strict_config["runtime"]["dry_run"] = False
    strict_report = run_preflight_checks(strict_config)
    assert not strict_report.ok, strict_report.to_dict()
    assert any(item["status"] == "error" for item in strict_report.checks), strict_report.to_dict()

    real_model_config = _base_config()
    real_model_config["model"]["provider"] = "openai"
    real_model_report = run_preflight_checks(real_model_config)
    model_checks = [item for item in real_model_report.checks if item["name"] == "model_provider"]
    assert model_checks, real_model_report.to_dict()
    assert model_checks[0]["status"] == "pass", real_model_report.to_dict()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())