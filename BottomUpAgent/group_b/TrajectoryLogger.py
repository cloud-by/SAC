"""
BottomUpAgent/group_b/TrajectoryLogger.py

小修版：
1. 统一 runtime_context 中的 current_episode_id
2. 自动把 episode_id 注入 state/action/feedback 记录
3. 让真实流程里 episode_id 不再左右互搏
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from BottomUpAgent.group_b.StateAdapter import StateAdapter
except Exception:  # pragma: no cover
    try:
        from StateAdapter import StateAdapter  # type: ignore
    except Exception:  # pragma: no cover
        StateAdapter = None  # type: ignore


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TrajectoryLogger:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.runtime_context = config.setdefault("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()
        self.paths = self._init_paths()
        self.current_episode_id: Optional[str] = self.runtime_context.get("current_episode_id")
        self.state_adapter = self._build_state_adapter()
        logging.info("TrajectoryLogger 初始化完成，trajectories_dir=%s", self.paths["trajectories"])

    def _init_paths(self) -> Dict[str, Path]:
        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        defaults = {
            "trajectories": "data/trajectories",
            "trajectory_steps": "data/trajectories/steps",
            "states": "data/states",
            "actions": "data/actions",
            "feedback": "data/feedback",
        }
        result: Dict[str, Path] = {}
        for key, default_value in defaults.items():
            value = raw_paths.get(key, default_value)
            path = Path(value)
            if not path.is_absolute():
                path = self.project_root / path
            path.mkdir(parents=True, exist_ok=True)
            result[key] = path.resolve()
        return result

    def _build_state_adapter(self):
        if StateAdapter is None:
            logging.warning("TrajectoryLogger 未找到 StateAdapter，将只记录原始 state_data。")
            return None
        try:
            return StateAdapter(self.config)
        except Exception as exc:
            logging.warning("StateAdapter 初始化失败，TrajectoryLogger 将只记录原始 state_data: %s", exc)
            return None

    def start_episode(self, task: str = "", meta: Optional[Dict[str, Any]] = None, episode_id: Optional[str] = None) -> str:
        if episode_id:
            eid = episode_id
        else:
            short = uuid.uuid4().hex[:6]
            eid = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{short}"
        self.current_episode_id = eid
        self.runtime_context["current_episode_id"] = eid
        self.runtime_context["episode_id"] = eid
        payload = {"episode_id": eid, "event": "episode_start", "task": task, "meta": meta or {}, "timestamp": now_str()}
        self._append_jsonl(self.paths["trajectories"] / f"{eid}.jsonl", payload)
        self._write_json(self.paths["trajectories"] / "latest_episode_start.json", payload)
        return eid

    def end_episode(self, summary: Optional[Dict[str, Any]] = None, episode_id: Optional[str] = None) -> Dict[str, Any]:
        eid = episode_id or self.current_episode_id or self.runtime_context.get("current_episode_id") or "manual_episode"
        payload = {"episode_id": eid, "event": "episode_end", "summary": summary or {}, "timestamp": now_str()}
        self._append_jsonl(self.paths["trajectories"] / f"{eid}.jsonl", payload)
        self._write_json(self.paths["trajectories"] / "latest_episode_end.json", payload)
        if episode_id is None or episode_id == self.current_episode_id:
            self.current_episode_id = None
            self.runtime_context["current_episode_id"] = None
        return payload

    def log_step(
        self,
        *,
        step_id: int,
        state_data: Dict[str, Any],
        action_data: Optional[Dict[str, Any]] = None,
        feedback_data: Optional[Dict[str, Any]] = None,
        teacher_feedback: Optional[Dict[str, Any]] = None,
        memory_summary: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
        task: str = "",
    ) -> Dict[str, Any]:
        eid = episode_id or self.current_episode_id or self.runtime_context.get("current_episode_id")
        if not eid:
            eid = self.start_episode(task=task)
        else:
            self.current_episode_id = eid
            self.runtime_context["current_episode_id"] = eid
            self.runtime_context["episode_id"] = eid

        state_payload = dict(state_data or {})
        state_payload.setdefault("episode_id", eid)
        action_payload = dict(action_data or {})
        action_payload.setdefault("episode_id", eid)
        action_payload.setdefault("params", {})
        action_payload["params"].setdefault("episode_id", eid)
        feedback_payload = dict(feedback_data or {})
        feedback_payload.setdefault("episode_id", eid)
        teacher_payload = dict(teacher_feedback or {})
        memory_payload = dict(memory_summary or {})
        extra_payload = dict(extra or {})

        adapted = None
        feature_dict = None
        if self.state_adapter is not None:
            try:
                payload = self.state_adapter.adapt_and_encode(
                    state_data=state_payload,
                    step_id=step_id,
                    episode_id=eid,
                    memory_summary=memory_payload,
                )
                adapted = payload.get("state_repr")
                feature_dict = payload.get("features")
            except Exception as exc:
                logging.warning("StateAdapter.adapt_and_encode 失败，改为只记录原始 state_data: %s", exc)

        record = {
            "episode_id": eid,
            "step_id": int(step_id),
            "timestamp": now_str(),
            "state_data": state_payload,
            "state_repr": adapted,
            "feature_dict": feature_dict,
            "action_data": action_payload,
            "feedback_data": feedback_payload,
            "teacher_feedback": teacher_payload,
            "memory_summary": memory_payload,
            "label": self._build_label(action_payload, feedback_payload, teacher_payload),
            "extra": extra_payload,
        }

        self._append_jsonl(self.paths["trajectories"] / f"{eid}.jsonl", record)
        step_name = f"{int(step_id):04d}"
        self._write_json(self.paths["trajectory_steps"] / f"{eid}_step_{step_name}.json", record)
        self._write_json(self.paths["states"] / f"{eid}_state_{step_name}.json", state_payload)
        self._write_json(self.paths["actions"] / f"{eid}_action_{step_name}.json", action_payload)
        self._write_json(self.paths["feedback"] / f"{eid}_feedback_{step_name}.json", feedback_payload)
        self._write_json(self.paths["trajectories"] / "latest_step_record.json", record)
        return record

    def build_training_sample(
        self,
        *,
        step_id: int,
        state_data: Dict[str, Any],
        action_data: Optional[Dict[str, Any]] = None,
        feedback_data: Optional[Dict[str, Any]] = None,
        teacher_feedback: Optional[Dict[str, Any]] = None,
        memory_summary: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        eid = episode_id or self.current_episode_id or self.runtime_context.get("current_episode_id") or "manual_episode"
        state_payload = dict(state_data or {})
        state_payload.setdefault("episode_id", eid)
        action_payload = dict(action_data or {})
        action_payload.setdefault("episode_id", eid)
        action_payload.setdefault("params", {})
        action_payload["params"].setdefault("episode_id", eid)
        feedback_payload = dict(feedback_data or {})
        feedback_payload.setdefault("episode_id", eid)
        teacher_payload = dict(teacher_feedback or {})

        adapted = None
        feature_dict = None
        if self.state_adapter is not None:
            payload = self.state_adapter.adapt_and_encode(
                state_data=state_payload,
                step_id=step_id,
                episode_id=eid,
                memory_summary=memory_summary,
            )
            adapted = payload.get("state_repr")
            feature_dict = payload.get("features")

        return {
            "episode_id": eid,
            "step_id": int(step_id),
            "state_repr": adapted,
            "feature_dict": feature_dict,
            "action_data": action_payload,
            "feedback_data": feedback_payload,
            "teacher_feedback": teacher_payload,
            "label": self._build_label(action_payload, feedback_payload, teacher_payload),
        }

    def _build_label(self, action_data: Dict[str, Any], feedback_data: Dict[str, Any], teacher_feedback: Dict[str, Any]) -> Dict[str, Any]:
        score = self._to_float(teacher_feedback.get("score"), default=0.0)
        execute_status = str(feedback_data.get("execute_status", "unknown") or "unknown").lower()
        before_scene = str(feedback_data.get("before_scene", "unknown") or "unknown")
        after_scene = str(feedback_data.get("after_scene", "unknown") or "unknown")
        return {
            "success": 1 if execute_status == "success" else 0,
            "value": round(score, 4),
            "scene_changed": 1 if before_scene != after_scene else 0,
            "action_type": action_data.get("action_type", "unknown"),
        }

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default
