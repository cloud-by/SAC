
from __future__ import annotations

from pathlib import Path

from BottomUpAgent.common.config_loader import load_yaml_file
from run import apply_runtime_mode_defaults, normalize_config


def _load_config(path: str) -> dict:
    payload = load_yaml_file(Path(path)) or {}
    assert isinstance(payload, dict)
    payload["_project_root"] = str(Path(__file__).resolve().parent)
    payload.setdefault("_runtime_context", {})
    return payload


def main() -> int:
    project_root = Path(__file__).resolve().parent

    dev = normalize_config(_load_config("config/dev_dryrun.yaml"), project_root)
    assert dev["runtime"]["mode"] == "dev_dryrun"
    assert dev["runtime"]["dry_run"] is True
    assert dev["runtime"]["enable_preflight"] is True
    assert dev["runtime"]["preflight_fail_on_warning"] is False
    assert dev["runtime"]["enable_gui_health_check"] is False

    train = normalize_config(_load_config("config/train_offline.yaml"), project_root)
    assert train["runtime"]["mode"] == "train_offline"
    assert train["runtime"]["dry_run"] is True
    assert train["runtime"]["enable_preflight"] is False
    assert train["runtime"]["startup_delay_seconds"] == 0
    assert train["runtime"]["enable_gui_health_check"] is False

    prod = normalize_config(_load_config("config/prod_windows.yaml"), project_root)
    assert prod["runtime"]["mode"] == "prod_windows"
    assert prod["runtime"]["dry_run"] is False
    assert prod["runtime"]["enable_preflight"] is True
    assert prod["runtime"]["preflight_fail_on_warning"] is True
    assert prod["runtime"]["enable_gui_health_check"] is True

    try:
        apply_runtime_mode_defaults({"runtime": {"mode": "unsupported"}, "environment": {}})
    except ValueError as exc:
        assert "不支持的 runtime.mode" in str(exc)
    else:
        raise AssertionError("unsupported mode should fail")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())