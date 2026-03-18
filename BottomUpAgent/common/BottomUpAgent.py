"""
BottomUpAgent/BottomUpAgent.py

系统总控中枢：
- 组织感知、决策、执行、反馈、经验更新的主循环
- 统一五类数据结构：
  1. 环境状态数据 state_data
  2. 动作决策数据 action_data
  3. 执行反馈数据 feedback_data
  4. 运行日志数据 log_data
  5. 经验/技能数据 skill_data
- 分类保存到 README 规定的目录中
- 为 visualizer 预留更新接口
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from BottomUpAgent.common.protocols import (
    ensure_action_protocol,
    ensure_feedback_protocol,
    ensure_skill_protocol,
    ensure_state_protocol,
)

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class _FallbackEye:
    """A组占位视觉模块。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def observe(self, step_id: int = 0, phase: str = "before") -> Dict[str, Any]:
        return {
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
            "timestamp": now_str(),
            "phase": phase,
            "step_id": step_id,
        }


class _FallbackBrain:
    """B组占位决策模块。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def plan(
        self,
        task: str,
        state_data: Dict[str, Any],
        memory_summary: Dict[str, Any],
        step_id: int,
    ) -> Dict[str, Any]:
        scene = state_data.get("scene_type", "unknown")
        return {
            "action_type": "wait",
            "target": {"scene_type": scene},
            "reason": f"第 {step_id} 步暂无正式 Brain，先执行占位等待动作",
            "confidence": 0.10,
            "source": "FallbackBrain",
            "timestamp": now_str(),
            "params": {"duration": 0.5},
        }


class _FallbackMcts:
    """B组占位搜索模块。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def search(self, action_data: Dict[str, Any], state_data: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        result = dict(action_data)
        source = result.get("source", "FallbackBrain")
        if "Mcts" not in source:
            result["source"] = f"{source}+FallbackMcts"
        return result


class _FallbackUnifiedOperation:
    """A组占位统一动作封装。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def normalize(self, action_data: Dict[str, Any], step_id: int = 0) -> Dict[str, Any]:
        result = dict(action_data)
        result.setdefault("action_type", "wait")
        result.setdefault("target", {})
        result.setdefault("reason", "未提供动作解释")
        result.setdefault("confidence", 0.0)
        result.setdefault("source", "FallbackUnifiedOperation")
        result.setdefault("timestamp", now_str())
        result.setdefault("params", {})
        return result


class _FallbackHand:
    """A组占位执行模块。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def execute(
        self,
        action_data: Dict[str, Any],
        state_data: Dict[str, Any],
        step_id: int,
    ) -> Dict[str, Any]:
        return {
            "action_type": action_data.get("action_type", "wait"),
            "execute_status": "success",
            "before_scene": state_data.get("scene_type", "unknown"),
            "after_scene": state_data.get("scene_type", "unknown"),
            "time_cost_ms": 100,
            "click_position": [],
            "screen_diff": "no_real_execution",
            "error_message": "",
            "screenshot_after": "",
            "timestamp": now_str(),
            "step_id": step_id,
        }


class _FallbackTeacher:
    """B组占位反思模块。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def reflect(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        step_id: int,
    ) -> Dict[str, Any]:
        success = feedback_data.get("execute_status") == "success"
        return {
            "step_id": step_id,
            "score": 1.0 if success else 0.0,
            "feedback": "占位 Teacher：动作已记录，可后续替换为正式反思逻辑。",
            "should_promote_to_skill": success,
            "timestamp": now_str(),
        }


class _FallbackLongMemory:
    """B组占位长时记忆/技能库。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.step_records: List[Dict[str, Any]] = []
        self.skills: Dict[str, Dict[str, Any]] = {}

    def add_step_record(self, step_record: Dict[str, Any]) -> None:
        self.step_records.append(step_record)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_records": len(self.step_records),
            "skill_count": len(self.skills),
            "recent_skills": list(self.skills.values())[-5:],
        }

    def retrieve(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        scene_type = state_data.get("scene_type", "unknown")
        return [v for v in self.skills.values() if v.get("scene_type") == scene_type]

    def update_skill(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        teacher_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scene_type = state_data.get("scene_type", "unknown")
        action_type = action_data.get("action_type", "unknown_action")
        skill_id = f"{scene_type}_{action_type}_001"

        existed = skill_id in self.skills
        record = self.skills.setdefault(
            skill_id,
            {
                "skill_id": skill_id,
                "scene_type": scene_type,
                "trigger_condition": f"scene={scene_type}",
                "action_pattern": action_data.get("reason", action_type),
                "success_count": 0,
                "fail_count": 0,
                "reuse_count": 0,
                "last_update": now_str(),
            },
        )

        if existed:
            record["reuse_count"] += 1

        if feedback_data.get("execute_status") == "success":
            record["success_count"] += 1
        else:
            record["fail_count"] += 1

        if teacher_feedback and teacher_feedback.get("feedback"):
            record["teacher_feedback"] = teacher_feedback["feedback"]

        record["last_update"] = now_str()
        return dict(record)


class _FallbackVisualizer:
    """A组占位可视化模块。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.latest_payload: Dict[str, Any] = {}

    def update(self, **kwargs: Any) -> None:
        self.latest_payload = kwargs


class BottomUpAgent:
    """系统总控类。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.runtime = config.get("runtime", {})
        self.model = config.get("model", {})
        self.environment = config.get("environment", {})
        self.visualization = config.get("visualization", {})
        self.runtime_context = config.get("_runtime_context", {})

        self.run_id = self.runtime_context.get("run_id", f"run_{int(time.time())}")
        self.max_steps = int(self.runtime.get("max_steps", 10))
        self.stop_on_failures = int(self.runtime.get("stop_on_failures", 3))
        self.consecutive_failures = 0

        self.paths = self._init_paths()

        self.eye = self._build_component(
            candidate_modules=[
                "BottomUpAgent.group_a.Eye",
                "BottomUpAgent.Eye",
                "Eye",
            ],
            class_names=["Eye"],
            fallback_cls=_FallbackEye,
        )

        self.brain = self._build_component(
            candidate_modules=[
                "BottomUpAgent.group_b.Brain",
                "BottomUpAgent.Brain",
                "Brain",
            ],
            class_names=["Brain"],
            fallback_cls=_FallbackBrain,
        )

        self.mcts = self._build_component(
            candidate_modules=[
                "BottomUpAgent.group_b.Mcts",
                "BottomUpAgent.Mcts",
                "Mcts",
            ],
            class_names=["Mcts", "MCTS"],
            fallback_cls=_FallbackMcts,
        )

        self.operation = self._build_component(
            candidate_modules=[
                "BottomUpAgent.group_a.UnifiedOperation",
                "BottomUpAgent.UnifiedOperation",
                "UnifiedOperation",
            ],
            class_names=["UnifiedOperation"],
            fallback_cls=_FallbackUnifiedOperation,
        )

        self.hand = self._build_component(
            candidate_modules=[
                "BottomUpAgent.group_a.Hand",
                "BottomUpAgent.Hand",
                "Hand",
            ],
            class_names=["Hand"],
            fallback_cls=_FallbackHand,
        )

        self.teacher = self._build_component(
            candidate_modules=[
                "BottomUpAgent.group_b.Teacher",
                "BottomUpAgent.Teacher",
                "Teacher",
            ],
            class_names=["Teacher"],
            fallback_cls=_FallbackTeacher,
        )

        self.memory = self._build_component(
            candidate_modules=[
                "BottomUpAgent.group_b.LongMemory",
                "BottomUpAgent.LongMemory",
                "LongMemory",
            ],
            class_names=["LongMemory"],
            fallback_cls=_FallbackLongMemory,
        )

        self.visualizer = self._build_component(
            candidate_modules=[
                "BottomUpAgent.group_a.visualizer",
                "BottomUpAgent.visualizer",
                "visualizer",
            ],
            class_names=["Visualizer", "visualizer", "Visualization"],
            fallback_cls=_FallbackVisualizer,
        )

        self.history: List[Dict[str, Any]] = []

        logging.info("BottomUpAgent 初始化完成")
        logging.info("run_id=%s, max_steps=%s", self.run_id, self.max_steps)

    def _init_paths(self) -> Dict[str, Path]:
        defaults = {
            "run_logs": Path("logs/run_logs"),
            "action_logs": Path("logs/action_logs"),
            "states": Path("data/states"),
            "actions": Path("data/actions"),
            "feedback": Path("data/feedback"),
            "skills": Path("data/skills"),
            "screenshots_current": Path("screenshots/current"),
            "screenshots_history": Path("screenshots/history"),
        }

        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        resolved: Dict[str, Path] = {}

        project_root = Path(self.config.get("_project_root", Path.cwd())).resolve()

        for key, default_path in defaults.items():
            value = raw_paths.get(key, str(default_path))
            path = Path(value)
            if not path.is_absolute():
                path = project_root / path
            path.mkdir(parents=True, exist_ok=True)
            resolved[key] = path.resolve()

        return resolved

    def _build_component(self, candidate_modules: List[str], class_names: List[str], fallback_cls):
        last_error = None

        for module_name in candidate_modules:
            try:
                module = importlib.import_module(module_name)

                for class_name in class_names:
                    cls = getattr(module, class_name, None)
                    if cls is None:
                        continue

                    try:
                        instance = cls(self.config)
                    except TypeError:
                        instance = cls()

                    logging.info(
                        "组件加载成功: module=%s, class=%s",
                        module_name,
                        class_name,
                    )
                    return instance

            except Exception as exc:
                last_error = exc
                continue

        logging.warning(
            "组件加载失败，候选模块=%s，使用占位版本。最后错误: %s",
            candidate_modules,
            last_error,
        )
        return fallback_cls(self.config)

    def _call_first_available(
        self,
        obj: Any,
        method_names: List[str],
        default: Any = None,
        **kwargs: Any,
    ) -> Any:
        for name in method_names:
            method = getattr(obj, name, None)
            if not callable(method):
                continue

            try:
                signature = inspect.signature(method)
                accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in signature.parameters.values()
                )
                if accepts_kwargs:
                    return method(**kwargs)

                filtered_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k in signature.parameters
                }
                return method(**filtered_kwargs)
            except TypeError:
                try:
                    return method()
                except Exception as exc:
                    logging.warning("调用 %s.%s 失败: %s", obj.__class__.__name__, name, exc)
            except Exception as exc:
                logging.warning("调用 %s.%s 失败: %s", obj.__class__.__name__, name, exc)

        return default

    def run(self, task: str) -> Dict[str, Any]:
        logging.info("开始执行任务: %s", task)

        for step_id in range(1, self.max_steps + 1):
            step_record = self.step(task=task, step_id=step_id)
            self.history.append(step_record)

            status = step_record["feedback_data"]["execute_status"]
            if status == "success":
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1

            logging.info(
                "Step %s/%s | action=%s | result=%s",
                step_id,
                self.max_steps,
                step_record["action_data"]["action_type"],
                status,
            )

            if self._should_stop(step_record):
                logging.info("满足停止条件，提前结束。")
                break

        run_summary = {
            "run_id": self.run_id,
            "task": task,
            "status": "finished",
            "total_steps": len(self.history),
            "consecutive_failures": self.consecutive_failures,
            "memory_summary": self._get_memory_summary(),
            "finished_at": now_str(),
        }

        summary_file = self.paths["run_logs"] / f"agent_summary_{self.run_id}.json"
        safe_json_dump(summary_file, run_summary)
        return run_summary

    def step(self, task: str, step_id: int) -> Dict[str, Any]:
        start_time = time.perf_counter()

        state_data = self._collect_state(step_id=step_id, phase="before")
        action_data = self._decide_action(task=task, state_data=state_data, step_id=step_id)
        feedback_data, post_state_data = self._execute_action(
            action_data=action_data,
            state_data=state_data,
            step_id=step_id,
        )
        skill_data, teacher_feedback = self._update_experience(
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            step_id=step_id,
        )

        total_time_ms = int((time.perf_counter() - start_time) * 1000)
        log_data = self._build_log_data(
            step_id=step_id,
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            total_time_ms=total_time_ms,
        )

        step_record = {
            "step_id": step_id,
            "task": task,
            "state_data": state_data,
            "post_state_data": post_state_data,
            "action_data": action_data,
            "feedback_data": feedback_data,
            "log_data": log_data,
            "skill_data": skill_data,
            "teacher_feedback": teacher_feedback,
            "timestamp": now_str(),
        }

        self._persist_step_record(step_record)
        self._update_visualizer(step_record)
        self._add_step_to_memory(step_record)

        return step_record

    def _collect_state(self, step_id: int, phase: str) -> Dict[str, Any]:
        fallback = _FallbackEye(self.config).observe(step_id=step_id, phase=phase)
        result = self._call_first_available(
            self.eye,
            ["observe", "get_state", "perceive"],
            default=fallback,
            step_id=step_id,
            phase=phase,
        )
        return self._normalize_state_data(result, step_id=step_id, phase=phase)

    def _decide_action(self, task: str, state_data: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        memory_summary = self._get_memory_summary()
        fallback_action = _FallbackBrain(self.config).plan(
            task=task,
            state_data=state_data,
            memory_summary=memory_summary,
            step_id=step_id,
        )

        proposed_action = self._call_first_available(
            self.brain,
            ["plan", "decide", "think"],
            default=fallback_action,
            task=task,
            state_data=state_data,
            observation=state_data,
            memory_summary=memory_summary,
            step_id=step_id,
        )

        searched_action = self._call_first_available(
            self.mcts,
            ["search", "select", "refine"],
            default=proposed_action,
            action_data=proposed_action,
            proposed_action=proposed_action,
            state_data=state_data,
            step_id=step_id,
        )

        normalized_action = self._call_first_available(
            self.operation,
            ["normalize", "normalize_action", "to_operation"],
            default=searched_action,
            action_data=searched_action,
            action=searched_action,
            step_id=step_id,
        )

        return self._normalize_action_data(normalized_action)

    def _execute_action(
        self,
        action_data: Dict[str, Any],
        state_data: Dict[str, Any],
        step_id: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        start_time = time.perf_counter()

        fallback_feedback = _FallbackHand(self.config).execute(
            action_data=action_data,
            state_data=state_data,
            step_id=step_id,
        )

        raw_feedback = self._call_first_available(
            self.hand,
            ["execute", "act", "perform"],
            default=fallback_feedback,
            action_data=action_data,
            action=action_data,
            state_data=state_data,
            step_id=step_id,
        )

        post_state_data = self._collect_state(step_id=step_id, phase="after")

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        feedback_data = self._normalize_feedback_data(
            raw_feedback=raw_feedback,
            action_data=action_data,
            before_state=state_data,
            after_state=post_state_data,
            elapsed_ms=elapsed_ms,
            step_id=step_id,
        )
        return feedback_data, post_state_data

    def _update_experience(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        step_id: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        fallback_teacher = _FallbackTeacher(self.config).reflect(
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            step_id=step_id,
        )

        teacher_feedback = self._call_first_available(
            self.teacher,
            ["reflect", "review", "evaluate"],
            default=fallback_teacher,
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            step_id=step_id,
        )

        fallback_skill = _FallbackLongMemory(self.config).update_skill(
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            teacher_feedback=teacher_feedback,
        )

        skill_data = self._call_first_available(
            self.memory,
            ["update_skill", "update_memory", "learn"],
            default=fallback_skill,
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            teacher_feedback=teacher_feedback,
        )

        return self._normalize_skill_data(skill_data, state_data, action_data), teacher_feedback

    def _get_memory_summary(self) -> Dict[str, Any]:
        fallback = {"total_records": 0, "skill_count": 0, "recent_skills": []}
        result = self._call_first_available(
            self.memory,
            ["summary", "get_summary"],
            default=fallback,
        )
        if isinstance(result, dict):
            return result
        return fallback

    def _add_step_to_memory(self, step_record: Dict[str, Any]) -> None:
        self._call_first_available(
            self.memory,
            ["add_step_record", "add", "append_record"],
            default=None,
            step_record=step_record,
            record=step_record,
        )

    def _update_visualizer(self, step_record: Dict[str, Any]) -> None:
        if not self.visualization.get("enabled", True):
            return

        self._call_first_available(
            self.visualizer,
            ["update", "render", "push"],
            default=None,
            state_data=step_record["state_data"],
            action_data=step_record["action_data"],
            feedback_data=step_record["feedback_data"],
            log_data=step_record["log_data"],
            skill_data=step_record["skill_data"],
            step_record=step_record,
        )

    def _normalize_state_data(
        self,
        raw: Any,
        step_id: int,
        phase: str,
    ) -> Dict[str, Any]:
        return ensure_state_protocol(raw, step_id=step_id, phase=phase)

    def _normalize_action_data(self, raw: Any) -> Dict[str, Any]:
        return ensure_action_protocol(raw)

    def _normalize_feedback_data(
        self,
        raw_feedback: Any,
        action_data: Dict[str, Any],
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        elapsed_ms: int,
        step_id: int,
    ) -> Dict[str, Any]:
        return ensure_feedback_protocol(
            raw_feedback,
            action_type=action_data.get("action_type", "wait"),
            before_scene=before_state.get("scene_type", "unknown"),
            after_scene=after_state.get("scene_type", before_state.get("scene_type", "unknown")),
            elapsed_ms=elapsed_ms,
            screenshot_after=after_state.get("screen_image", ""),
            step_id=step_id,
            screen_diff=self._infer_screen_diff(before_state, after_state),
        )

    def _normalize_skill_data(
        self,
        raw_skill: Any,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        scene_type = state_data.get("scene_type", "unknown")
        action_type = action_data.get("action_type", "unknown_action")
        return ensure_skill_protocol(
            raw_skill,
            scene_type=scene_type,
            action_type=action_type,
            action_pattern=action_data.get("reason", action_type),
        )

    def _infer_screen_diff(
        self,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
    ) -> str:
        before_scene = before_state.get("scene_type", "unknown")
        after_scene = after_state.get("scene_type", "unknown")

        if before_scene != after_scene:
            return f"scene_changed:{before_scene}->{after_scene}"

        if before_state.get("hp") != after_state.get("hp"):
            return "hp_changed"

        if before_state.get("energy") != after_state.get("energy"):
            return "energy_changed"

        if before_state.get("hand_cards") != after_state.get("hand_cards"):
            return "hand_cards_changed"

        if before_state.get("enemies") != after_state.get("enemies"):
            return "enemy_state_changed"

        return "no_obvious_change"

    def _build_input_summary(self, state_data: Dict[str, Any]) -> str:
        return (
            f"scene={state_data.get('scene_type', 'unknown')}, "
            f"cards={len(state_data.get('hand_cards', []))}, "
            f"enemies={len(state_data.get('enemies', []))}, "
            f"energy={state_data.get('energy', None)}"
        )

    def _build_decision_summary(self, action_data: Dict[str, Any]) -> str:
        action_type = action_data.get("action_type", "unknown")
        reason = action_data.get("reason", "")
        return f"{action_type} | {reason}"

    def _build_log_data(
        self,
        step_id: int,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        total_time_ms: int,
    ) -> Dict[str, Any]:
        return {
            "step_id": step_id,
            "scene_type": state_data.get("scene_type", "unknown"),
            "input_summary": self._build_input_summary(state_data),
            "decision": self._build_decision_summary(action_data),
            "result": feedback_data.get("execute_status", "unknown"),
            "time_cost_ms": total_time_ms,
            "screenshot": feedback_data.get("screenshot_after", ""),
            "operator": "A组执行平台",
            "timestamp": now_str(),
        }

    def _persist_step_record(self, step_record: Dict[str, Any]) -> None:
        step_id = step_record["step_id"]
        step_tag = f"step_{step_id:03d}"

        state_before_file = self.paths["states"] / f"{step_tag}_before.json"
        state_after_file = self.paths["states"] / f"{step_tag}_after.json"
        action_file = self.paths["actions"] / f"{step_tag}.json"
        feedback_file = self.paths["feedback"] / f"{step_tag}.json"
        skill_file = self.paths["skills"] / f"{step_tag}.json"
        action_log_file = self.paths["action_logs"] / f"{step_tag}.json"

        safe_json_dump(state_before_file, step_record["state_data"])
        safe_json_dump(state_after_file, step_record["post_state_data"])
        safe_json_dump(action_file, step_record["action_data"])
        safe_json_dump(feedback_file, step_record["feedback_data"])
        safe_json_dump(skill_file, step_record["skill_data"])
        safe_json_dump(action_log_file, step_record["log_data"])

    def _should_stop(self, step_record: Dict[str, Any]) -> bool:
        action_type = step_record["action_data"].get("action_type")
        execute_status = step_record["feedback_data"].get("execute_status")
        after_scene = step_record["feedback_data"].get("after_scene")

        if action_type == "finish":
            return True

        if after_scene in {"victory", "game_over"}:
            return True

        if execute_status in {"fatal", "terminated"}:
            return True

        if self.consecutive_failures >= self.stop_on_failures:
            logging.warning("连续失败次数达到阈值: %s", self.stop_on_failures)
            return True

        return False