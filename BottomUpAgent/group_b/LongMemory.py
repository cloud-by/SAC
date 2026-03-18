"""
BottomUpAgent/LongMemory.py

B组长时记忆模块：
1. 保存历史 step 记录
2. 维护经验库 / 技能库
3. 根据执行反馈更新技能统计
4. 提供简单的经验检索能力
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from BottomUpAgent.common.protocols import (
    ensure_action_protocol,
    ensure_feedback_protocol,
    ensure_skill_protocol,
    ensure_state_protocol,
    ensure_teacher_protocol,
)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class LongMemory:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.runtime_context = config.get("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()

        self.max_step_records = int(config.get("runtime", {}).get("max_memory_records", 500))
        self.step_records: List[Dict[str, Any]] = []
        self.skills: Dict[str, Dict[str, Any]] = {}

        self.paths = self._init_paths()
        self._load_existing_skills()

        logging.info("LongMemory 初始化完成，已有技能数=%s", len(self.skills))

    def _init_paths(self) -> Dict[str, Path]:
        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        result: Dict[str, Path] = {}

        defaults = {
            "skills": "data/skills",
            "states": "data/states",
        }

        for key, default_value in defaults.items():
            value = raw_paths.get(key, default_value)
            path = Path(value)
            if not path.is_absolute():
                path = self.project_root / path
            path.mkdir(parents=True, exist_ok=True)
            result[key] = path.resolve()

        return result

    def _load_existing_skills(self) -> None:
        """
        尝试读取之前保存的技能快照，便于演示“记忆延续”。
        当前策略：读取 latest_skills.json（如果存在）。
        """
        latest_file = self.paths["skills"] / "latest_skills.json"
        if not latest_file.exists():
            return

        try:
            with latest_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                raw_skills = payload.get("skills", {})
                if isinstance(raw_skills, dict):
                    self.skills = raw_skills
        except Exception as exc:
            logging.warning("读取已有技能库失败，忽略并继续: %s", exc)

    def add_step_record(self, step_record: Dict[str, Any]) -> None:
        self.step_records.append(step_record)

        if len(self.step_records) > self.max_step_records:
            self.step_records = self.step_records[-self.max_step_records :]

    def add(self, record: Dict[str, Any]) -> None:
        """
        兼容旧接口。
        """
        self.add_step_record(record)

    def summary(self) -> Dict[str, Any]:
        recent_skills = sorted(
            self.skills.values(),
            key=lambda x: (self._priority_rank(x.get("memory_priority", "low")), x.get("last_update", "")),
            reverse=True,
        )[:5]

        return {
            "total_records": len(self.step_records),
            "skill_count": len(self.skills),
            "recent_skills": recent_skills,
        }

    def get_summary(self) -> Dict[str, Any]:
        return self.summary()

    def retrieve(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据 scene_type 检索技能。
        当前先做最朴素的检索：同场景优先，再按 success_count 排序。
        """
        state_data = ensure_state_protocol(state_data, step_id=state_data.get("step_id"),
                                           phase=str(state_data.get("phase", "before") or "before"))
        scene_type = state_data.get("scene_type", "unknown")
        candidates = [
            skill for skill in self.skills.values()
            if skill.get("scene_type") == scene_type
        ]

        candidates.sort(
            key=lambda x: (
                self._priority_rank(x.get("memory_priority", "low")),
                x.get("success_count", 0) - x.get("fail_count", 0),
                x.get("reuse_count", 0),
            ),
            reverse=True,
        )
        return candidates[:10]

    def find_related_skills(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.retrieve(state_data)

    def update_skill(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        teacher_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        根据本次执行情况更新技能。
        """
        state_data = ensure_state_protocol(
            state_data,
            step_id=state_data.get("step_id"),
            phase=str(state_data.get("phase", "before") or "before"),
        )
        action_data = ensure_action_protocol(
            action_data,
            scene_type=str(state_data.get("scene_type", "unknown") or "unknown"),
            episode_id=action_data.get("episode_id") or state_data.get("episode_id"),
            step_id=action_data.get("params", {}).get("step_id") if isinstance(action_data.get("params"),
                                                                               dict) else None,
        )
        feedback_data = ensure_feedback_protocol(
            feedback_data,
            action_type=action_data.get("action_type", "unknown_action"),
            before_scene=feedback_data.get("before_scene", state_data.get("scene_type", "unknown")),
            after_scene=feedback_data.get("after_scene", state_data.get("scene_type", "unknown")),
            elapsed_ms=int(feedback_data.get("time_cost_ms", 0) or 0),
            screenshot_after=str(feedback_data.get("screenshot_after", "") or ""),
            step_id=int(feedback_data.get("step_id", state_data.get("step_id", 0)) or 0),
            screen_diff=str(feedback_data.get("screen_diff", "") or ""),
        )
        teacher_feedback = ensure_teacher_protocol(
            teacher_feedback or {},
            step_id=int((teacher_feedback or {}).get("step_id",
                                                     feedback_data.get("step_id", state_data.get("step_id", 0))) or 0),
            scene_type=str(
                (teacher_feedback or {}).get("scene_type", state_data.get("scene_type", "unknown")) or "unknown"),
            action_type=str((teacher_feedback or {}).get("action_type", action_data.get("action_type",
                                                                                        "unknown_action")) or "unknown_action"),
            execute_status=str((teacher_feedback or {}).get("execute_status", feedback_data.get("execute_status",
                                                                                                "unknown")) or "unknown"),
            episode_id=(teacher_feedback or {}).get("episode_id") or action_data.get("episode_id") or state_data.get(
                "episode_id"),
        )

        scene_type = state_data.get("scene_type", "unknown")
        action_type = action_data.get("action_type", "unknown_action")
        reason = action_data.get("reason", action_type)

        trigger_condition = self._build_trigger_condition(state_data, action_data)
        skill_id = str(teacher_feedback.get("skill_key") or self._build_skill_id(scene_type, action_type, state_data, action_data))

        if skill_id not in self.skills:
            self.skills[skill_id] = {
                "skill_id": skill_id,
                "scene_type": scene_type,
                "trigger_condition": trigger_condition,
                "action_pattern": reason,
                "success_count": 0,
                "fail_count": 0,
                "reuse_count": 0,
                "last_update": now_str(),
            }

        skill = self.skills[skill_id]

        execute_status = feedback_data.get("execute_status", "unknown")
        if execute_status == "success":
            skill["success_count"] = int(skill.get("success_count", 0)) + 1
        else:
            skill["fail_count"] = int(skill.get("fail_count", 0)) + 1

        skill["reuse_count"] = int(skill.get("reuse_count", 0)) + 1
        skill["last_update"] = now_str()
        skill["trigger_condition"] = trigger_condition
        skill["action_pattern"] = reason
        skill["memory_priority"] = teacher_feedback.get("memory_priority", skill.get("memory_priority", "low"))
        skill["outcome_tags"] = sorted(
            set(list(skill.get("outcome_tags", [])) + list(teacher_feedback.get("outcome_tags", []))))
        skill["teacher_feedback"] = teacher_feedback.get("feedback", "")
        skill["last_score"] = teacher_feedback.get("score", skill.get("last_score"))
        skill["last_episode_id"] = teacher_feedback.get("episode_id") or action_data.get(
            "episode_id") or state_data.get("episode_id")
        skill["last_execute_status"] = execute_status

        self._save_latest_skills()
        return dict(skill)

    def update_memory(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        teacher_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        兼容接口。
        """
        return self.update_skill(
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            teacher_feedback=teacher_feedback,
        )

    def learn(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        teacher_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        兼容接口。
        """
        return self.update_skill(
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            teacher_feedback=teacher_feedback,
        )

    def _build_trigger_condition(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
    ) -> str:
        scene_type = state_data.get("scene_type", "unknown")
        hp = state_data.get("hp")
        max_hp = state_data.get("max_hp")
        energy = state_data.get("energy")
        enemy_count = len(state_data.get("enemies", []))
        card_count = len(state_data.get("hand_cards", []))

        return (
            f"scene={scene_type}, hp={hp}/{max_hp}, "
            f"energy={energy}, enemies={enemy_count}, cards={card_count}"
        )

    def _build_skill_id(
        self,
        scene_type: str,
        action_type: str,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
    ) -> str:
        """
        生成比较稳定的 skill_id。
        """
        target = action_data.get("target", {})
        target_name = "generic"

        if isinstance(target, dict):
            if "card_name" in target:
                target_name = str(target["card_name"])
            elif "name" in target:
                target_name = str(target["name"])
            elif "button" in target:
                target_name = str(target["button"])
            elif "card" in target and isinstance(target["card"], dict):
                target_name = str(target["card"].get("name", "card"))
            elif "enemy" in target and isinstance(target["enemy"], dict):
                target_name = str(target["enemy"].get("name", "enemy"))

        safe_target = target_name.lower().replace(" ", "_")
        return f"{scene_type}_{action_type}_{safe_target}"

    def _save_latest_skills(self) -> None:
        latest_file = self.paths["skills"] / "latest_skills.json"
        payload = {
            "updated_at": now_str(),
            "skills": self.skills,
        }

        try:
            with latest_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("保存 latest_skills.json 失败: %s", exc)

    def _priority_rank(self, value: Any) -> int:
        text = str(value or "low").lower()
        if text == "high":
            return 3
        if text == "medium":
            return 2
        return 1