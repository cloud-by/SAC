from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("缺少 PyYAML，请先执行: pip install pyyaml") from exc

    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件不是 dict: {path}")
    return data


def main() -> int:
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parent

    if not (project_root / "BottomUpAgent").exists():
        print(f"[ERROR] 项目根目录看起来不对: {project_root}")
        print(r"用法: python test_scene_capture_flow.py D:\code\sts-agent-course-project")
        return 2

    sys.path.insert(0, str(project_root))

    config_path = project_root / "config" / "game.yaml"
    config = _load_yaml(config_path)
    config["_project_root"] = str(project_root)
    config.setdefault("_runtime_context", {})

    env = config.setdefault("environment", {})
    env.setdefault("capture_mode", "window")
    env.setdefault("window_name", "Slay the Spire 2")

    from BottomUpAgent.group_a.Eye import Eye

    eye = Eye(config)
    state = eye.observe(step_id=999, phase="manual_test")

    scene_scores = state.get("scene_scores", {}) or {}
    sorted_scores = sorted(scene_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_scores = sorted_scores[:6]

    summary = {
        "scene_type": state.get("scene_type"),
        "scene_variant": state.get("scene_variant"),
        "match_confidence": state.get("match_confidence"),
        "matched_template": state.get("matched_template"),
        "flow_prev_scene": state.get("flow_prev_scene"),
        "flow_phase_before": state.get("flow_phase_before"),
        "flow_phase_after": state.get("flow_phase_after"),
        "flow_corrected": state.get("flow_corrected"),
        "flow_reason": state.get("flow_reason"),
        "flow_allowed_scenes": state.get("flow_allowed_scenes", []),
        "flow_ordered_candidates": state.get("flow_ordered_candidates", []),
        "window_bbox": state.get("window_bbox"),
        "screen_image": state.get("screen_image"),
        "selected_character": state.get("selected_character"),
        "top_scene_scores": top_scores,
    }

    print("\n================ Flow 测试结果 ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = project_root / "logs" / "run_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "manual_scene_flow_test.json"
    out_file.write_text(
        json.dumps({"summary": summary, "full_state": state}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[OK] 已保存测试结果:")
    print(f"  - {out_file}")
    print("===============================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())