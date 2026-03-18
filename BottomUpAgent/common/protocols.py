from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


STATE_DEFAULTS: Dict[str, Any] = {
    "scene_type": "unknown",
    "floor": None,
    "hp": None,
    "max_hp": None,
    "energy": None,
    "hand_cards": [],
    "enemies": [],
    "map_options": [],
    "reward_options": [],
    "screen_image": "",
    "detected_regions": {},
}

ACTION_DEFAULTS: Dict[str, Any] = {
    "action_type": "wait",
    "target": {},
    "reason": "未提供动作解释",
    "confidence": 0.0,
    "source": "unknown",
    "params": {},
}

FEEDBACK_DEFAULTS: Dict[str, Any] = {
    "execute_status": "success",
    "click_position": [],
    "error_message": "",
}

SKILL_COUNTER_DEFAULTS: Dict[str, Any] = {
    "success_count": 0,
    "fail_count": 0,
    "reuse_count": 0,
}

TEACHER_DEFAULTS: Dict[str, Any] = {
    "score": 0.0,
    "feedback": "",
    "should_promote_to_skill": False,
    "memory_priority": "low",
    "outcome_tags": [],
}


def ensure_teacher_protocol(
    raw: Any,
    *,
    step_id: int,
    scene_type: str,
    action_type: str,
    execute_status: str,
    episode_id: str | None = None,
) -> Dict[str, Any]:
    result = dict(raw) if isinstance(raw, dict) else {}
    for key, value in TEACHER_DEFAULTS.items():
        result.setdefault(key, value.copy() if isinstance(value, (list, dict)) else value)
    result.setdefault("timestamp", now_str())
    result["step_id"] = step_id
    result["scene_type"] = scene_type
    result["action_type"] = action_type
    result["execute_status"] = execute_status
    if episode_id is not None:
        result["episode_id"] = episode_id
    result.setdefault("skill_key", f"{scene_type}::{action_type}")
    return result


def ensure_state_protocol(raw: Any, *, step_id: int | None = None, phase: str | None = None) -> Dict[str, Any]:
    result = dict(raw) if isinstance(raw, dict) else {}
    for key, value in STATE_DEFAULTS.items():
        result.setdefault(key, value.copy() if isinstance(value, (list, dict)) else value)
    result.setdefault("timestamp", now_str())
    if step_id is not None:
        result["step_id"] = step_id
    if phase is not None:
        result["phase"] = phase
    return result



def ensure_action_protocol(raw: Any, *, scene_type: str | None = None, episode_id: str | None = None, step_id: int | None = None) -> Dict[str, Any]:
    result = dict(raw) if isinstance(raw, dict) else {}
    for key, value in ACTION_DEFAULTS.items():
        result.setdefault(key, value.copy() if isinstance(value, (list, dict)) else value)
    result.setdefault("timestamp", now_str())
    if scene_type is not None:
        result.setdefault("scene_type", scene_type)
    if episode_id is not None:
        result["episode_id"] = episode_id
        result.setdefault("params", {})
        result["params"].setdefault("episode_id", episode_id)
    if step_id is not None:
        result.setdefault("params", {})
        result["params"].setdefault("step_id", step_id)
    return result



def ensure_feedback_protocol(
    raw: Any,
    *,
    action_type: str,
    before_scene: str,
    after_scene: str,
    elapsed_ms: int,
    screenshot_after: str,
    step_id: int,
    screen_diff: str,
) -> Dict[str, Any]:
    result = dict(raw) if isinstance(raw, dict) else {}
    result["action_type"] = result.get("action_type", action_type)
    for key, value in FEEDBACK_DEFAULTS.items():
        result.setdefault(key, value.copy() if isinstance(value, (list, dict)) else value)
    result.setdefault("before_scene", before_scene)
    result.setdefault("after_scene", after_scene)
    result.setdefault("time_cost_ms", elapsed_ms)
    result.setdefault("screen_diff", screen_diff)
    result.setdefault("screenshot_after", screenshot_after)
    result.setdefault("timestamp", now_str())
    result["step_id"] = step_id
    return result



def ensure_skill_protocol(raw: Any, *, scene_type: str, action_type: str, action_pattern: str) -> Dict[str, Any]:
    result = dict(raw) if isinstance(raw, dict) else {}
    result.setdefault("skill_id", f"{scene_type}_{action_type}_001")
    result.setdefault("scene_type", scene_type)
    result.setdefault("trigger_condition", f"scene={scene_type}")
    result.setdefault("action_pattern", action_pattern)
    for key, value in SKILL_COUNTER_DEFAULTS.items():
        result.setdefault(key, value)
    result.setdefault("last_update", now_str())
    return result