"""
BottomUpAgent/group_a/visualizer.py

A组可视化与展示模块（增强版）：
1. 展示当前状态（state_data）
2. 展示当前动作（action_data）
3. 展示执行反馈与日志（feedback_data / log_data）
4. 展示技能演化信息（skill_data）
5. 将展示快照保存到 logs/run_logs 目录，便于答辩演示

说明：
- 保持轻量文本/JSON 可视化，不引入 GUI 依赖。
- 新增展示 SceneMatcher / SceneFlowGuard 相关调试字段。
- 所有路径均按项目根目录解析，不写死绝对路径。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Visualizer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.runtime_context = config.get("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()
        self.visualization_config = config.get("visualization", {})

        self.enabled = bool(self.visualization_config.get("enabled", True))
        self.refresh_interval_ms = int(self.visualization_config.get("refresh_interval_ms", 300))
        self.max_preview_items = int(self.visualization_config.get("max_preview_items", 5))
        self.max_history = int(self.visualization_config.get("max_history", 300))

        self.paths = self._init_paths()

        self.latest_payload: Dict[str, Any] = {}
        self.step_history: List[Dict[str, Any]] = []

        logging.info(
            "Visualizer 初始化完成，enabled=%s, refresh_interval_ms=%s, max_preview_items=%s",
            self.enabled,
            self.refresh_interval_ms,
            self.max_preview_items,
        )

    def _init_paths(self) -> Dict[str, Path]:
        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        result: Dict[str, Path] = {}

        defaults = {
            "run_logs": "logs/run_logs",
            "action_logs": "logs/action_logs",
            "skills": "data/skills",
        }

        for key, default_value in defaults.items():
            value = raw_paths.get(key, default_value)
            path = Path(value)
            if not path.is_absolute():
                path = self.project_root / path
            path.mkdir(parents=True, exist_ok=True)
            result[key] = path.resolve()

        return result

    def update(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        log_data: Dict[str, Any],
        skill_data: Dict[str, Any],
        step_record: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        接收 BottomUpAgent 每一步的五类核心数据，并更新展示结果。
        """
        if not self.enabled:
            return

        display_payload = {
            "updated_at": now_str(),
            "state_panel": self._build_state_panel(state_data),
            "action_panel": self._build_action_panel(action_data),
            "feedback_panel": self._build_feedback_panel(feedback_data),
            "log_panel": self._build_log_panel(log_data),
            "skill_panel": self._build_skill_panel(skill_data),
            "step_record": step_record or {},
        }

        self.latest_payload = display_payload
        self.step_history.append(
            {
                "state_data": state_data,
                "action_data": action_data,
                "feedback_data": feedback_data,
                "log_data": log_data,
                "skill_data": skill_data,
                "step_record": step_record or {},
                "timestamp": now_str(),
            }
        )
        if len(self.step_history) > self.max_history:
            self.step_history = self.step_history[-self.max_history :]

        self._save_latest_dashboard(display_payload)
        self._save_step_snapshot(log_data, display_payload)
        self._save_text_panel(display_payload)

        # 控制台简要展示，便于直接跑脚本时观察
        self._print_console_summary(display_payload)

    def render(self, **kwargs: Any) -> None:
        """兼容 BottomUpAgent 中可能调用的 render 接口。"""
        self.update(**kwargs)

    def push(self, **kwargs: Any) -> None:
        """兼容 BottomUpAgent 中可能调用的 push 接口。"""
        self.update(**kwargs)

    def get_latest_payload(self) -> Dict[str, Any]:
        return self.latest_payload

    def get_step_history(self) -> List[Dict[str, Any]]:
        return self.step_history

    def _build_state_panel(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        hand_cards = state_data.get("hand_cards", []) or []
        enemies = state_data.get("enemies", []) or []
        menu_options = state_data.get("menu_options", []) or []
        mode_options = state_data.get("mode_options", []) or []
        character_options = state_data.get("character_options", []) or []
        map_options = state_data.get("map_options", []) or []
        reward_options = state_data.get("reward_options", []) or []
        scene_scores = state_data.get("scene_scores", {}) or {}
        top_scene_scores = sorted(scene_scores.items(), key=lambda kv: kv[1], reverse=True)[: self.max_preview_items]
        flow_candidates = state_data.get("flow_ordered_candidates", []) or []

        return {
            "title": "当前状态",
            "scene_type": state_data.get("scene_type", "unknown"),
            "scene_variant": state_data.get("scene_variant"),
            "floor": state_data.get("floor"),
            "hp": f"{state_data.get('hp', '?')}/{state_data.get('max_hp', '?')}",
            "energy": state_data.get("energy"),
            "selected_character": state_data.get("selected_character"),
            "match_confidence": state_data.get("match_confidence"),
            "matched_template": state_data.get("matched_template", ""),
            "flow_prev_scene": state_data.get("flow_prev_scene"),
            "flow_phase_before": state_data.get("flow_phase_before"),
            "flow_phase_after": state_data.get("flow_phase_after"),
            "flow_corrected": state_data.get("flow_corrected", False),
            "flow_reason": state_data.get("flow_reason", ""),
            "flow_ordered_candidates": flow_candidates[: self.max_preview_items],
            "top_scene_scores": top_scene_scores,
            "window_bbox": state_data.get("window_bbox"),
            "available_buttons": state_data.get("available_buttons", []),
            "hand_card_count": len(hand_cards),
            "enemy_count": len(enemies),
            "menu_options_count": len(menu_options),
            "mode_options_count": len(mode_options),
            "character_options_count": len(character_options),
            "map_options_count": len(map_options),
            "reward_options_count": len(reward_options),
            "hand_cards": self._preview_items(hand_cards),
            "enemies": self._preview_items(enemies),
            "menu_options": self._preview_items(menu_options),
            "mode_options": self._preview_items(mode_options),
            "character_options": self._preview_items(character_options),
            "map_options": self._preview_items(map_options),
            "reward_options": self._preview_items(reward_options),
            "screen_image": state_data.get("screen_image", ""),
        }

    def _build_action_panel(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        execution_plan = action_data.get("execution_plan", []) or []
        return {
            "title": "动作决策",
            "action_type": action_data.get("action_type", "unknown"),
            "target": action_data.get("target", {}),
            "reason": action_data.get("reason", ""),
            "confidence": action_data.get("confidence", 0.0),
            "source": action_data.get("source", "unknown"),
            "params": action_data.get("params", {}),
            "execution_plan_count": len(execution_plan),
            "execution_plan": self._preview_items(execution_plan),
        }

    def _build_feedback_panel(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": "执行反馈",
            "execute_status": feedback_data.get("execute_status", "unknown"),
            "before_scene": feedback_data.get("before_scene", "unknown"),
            "after_scene": feedback_data.get("after_scene", "unknown"),
            "time_cost_ms": feedback_data.get("time_cost_ms", 0),
            "click_position": feedback_data.get("click_position", []),
            "screen_diff": feedback_data.get("screen_diff", ""),
            "error_message": feedback_data.get("error_message", ""),
            "screenshot_after": feedback_data.get("screenshot_after", ""),
        }

    def _build_log_panel(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": "运行日志",
            "step_id": log_data.get("step_id"),
            "scene_type": log_data.get("scene_type", "unknown"),
            "input_summary": log_data.get("input_summary", ""),
            "decision": log_data.get("decision", ""),
            "result": log_data.get("result", ""),
            "time_cost_ms": log_data.get("time_cost_ms", 0),
            "operator": log_data.get("operator", "unknown"),
            "timestamp": log_data.get("timestamp", ""),
        }

    def _build_skill_panel(self, skill_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": "技能演化",
            "skill_id": skill_data.get("skill_id", ""),
            "scene_type": skill_data.get("scene_type", "unknown"),
            "trigger_condition": skill_data.get("trigger_condition", ""),
            "action_pattern": skill_data.get("action_pattern", ""),
            "success_count": skill_data.get("success_count", 0),
            "fail_count": skill_data.get("fail_count", 0),
            "reuse_count": skill_data.get("reuse_count", 0),
            "last_update": skill_data.get("last_update", ""),
            "teacher_feedback": skill_data.get("teacher_feedback", ""),
        }

    def _save_latest_dashboard(self, display_payload: Dict[str, Any]) -> None:
        file_path = self.paths["run_logs"] / "latest_dashboard.json"
        try:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(display_payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("保存 latest_dashboard.json 失败: %s", exc)

    def _save_step_snapshot(self, log_data: Dict[str, Any], display_payload: Dict[str, Any]) -> None:
        step_id = int(log_data.get("step_id", 0) or 0)
        file_path = self.paths["run_logs"] / f"dashboard_step_{step_id:03d}.json"
        try:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(display_payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("保存 step dashboard 失败: %s", exc)

    def _save_text_panel(self, display_payload: Dict[str, Any]) -> None:
        """额外生成一个纯文本面板，答辩时直接打开也比较直观。"""
        file_path = self.paths["run_logs"] / "latest_dashboard.txt"
        try:
            lines = self._to_text_lines(display_payload)
            file_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            logging.warning("保存 latest_dashboard.txt 失败: %s", exc)

    def _to_text_lines(self, display_payload: Dict[str, Any]) -> List[str]:
        state_panel = display_payload.get("state_panel", {})
        action_panel = display_payload.get("action_panel", {})
        feedback_panel = display_payload.get("feedback_panel", {})
        log_panel = display_payload.get("log_panel", {})
        skill_panel = display_payload.get("skill_panel", {})

        lines = [
            f"更新时间: {display_payload.get('updated_at', '')}",
            "",
            "===== 当前状态 =====",
            f"场景: {state_panel.get('scene_type', 'unknown')}",
            f"场景变体: {state_panel.get('scene_variant', '')}",
            f"流程修正: {state_panel.get('flow_corrected', False)}",
            f"流程原因: {state_panel.get('flow_reason', '')}",
            f"上一场景: {state_panel.get('flow_prev_scene', '')}",
            f"阶段变化: {state_panel.get('flow_phase_before', '')} -> {state_panel.get('flow_phase_after', '')}",
            f"匹配置信度: {self._fmt_float(state_panel.get('match_confidence'))}",
            f"命中模板: {state_panel.get('matched_template', '')}",
            f"所选角色: {state_panel.get('selected_character', '')}",
            f"层数: {state_panel.get('floor', '')}",
            f"生命: {state_panel.get('hp', '')}",
            f"能量: {state_panel.get('energy', '')}",
            f"手牌数: {state_panel.get('hand_card_count', 0)}",
            f"敌人数: {state_panel.get('enemy_count', 0)}",
            f"菜单选项数: {state_panel.get('menu_options_count', 0)}",
            f"模式选项数: {state_panel.get('mode_options_count', 0)}",
            f"角色选项数: {state_panel.get('character_options_count', 0)}",
            f"地图选项数: {state_panel.get('map_options_count', 0)}",
            f"奖励选项数: {state_panel.get('reward_options_count', 0)}",
            f"可用按钮: {self._format_inline_list(state_panel.get('available_buttons', []))}",
            f"候选场景: {self._format_score_pairs(state_panel.get('top_scene_scores', []))}",
            f"流程候选: {self._format_inline_list(state_panel.get('flow_ordered_candidates', []))}",
            f"预览手牌: {self._format_inline_list(state_panel.get('hand_cards', []))}",
            f"预览敌人: {self._format_inline_list(state_panel.get('enemies', []))}",
            f"截图: {state_panel.get('screen_image', '')}",
            "",
            "===== 动作决策 =====",
            f"动作类型: {action_panel.get('action_type', 'unknown')}",
            f"动作来源: {action_panel.get('source', 'unknown')}",
            f"置信度: {self._fmt_float(action_panel.get('confidence', 0.0))}",
            f"动作解释: {action_panel.get('reason', '')}",
            f"执行计划数: {action_panel.get('execution_plan_count', 0)}",
            f"执行计划预览: {self._format_inline_list(action_panel.get('execution_plan', []))}",
            "",
            "===== 执行反馈 =====",
            f"执行结果: {feedback_panel.get('execute_status', 'unknown')}",
            f"场景变化: {feedback_panel.get('before_scene', 'unknown')} -> {feedback_panel.get('after_scene', 'unknown')}",
            f"耗时(ms): {feedback_panel.get('time_cost_ms', 0)}",
            f"屏幕变化: {feedback_panel.get('screen_diff', '')}",
            f"点击位置: {feedback_panel.get('click_position', [])}",
            f"错误信息: {feedback_panel.get('error_message', '')}",
            "",
            "===== 运行日志 =====",
            f"Step: {log_panel.get('step_id', '')}",
            f"输入摘要: {log_panel.get('input_summary', '')}",
            f"决策摘要: {log_panel.get('decision', '')}",
            f"结果: {log_panel.get('result', '')}",
            f"操作方: {log_panel.get('operator', '')}",
            "",
            "===== 技能演化 =====",
            f"技能ID: {skill_panel.get('skill_id', '')}",
            f"触发条件: {skill_panel.get('trigger_condition', '')}",
            f"动作模式: {skill_panel.get('action_pattern', '')}",
            f"成功次数: {skill_panel.get('success_count', 0)}",
            f"失败次数: {skill_panel.get('fail_count', 0)}",
            f"复用次数: {skill_panel.get('reuse_count', 0)}",
            f"最近更新: {skill_panel.get('last_update', '')}",
            f"教师反馈: {skill_panel.get('teacher_feedback', '')}",
        ]
        return lines

    def _print_console_summary(self, display_payload: Dict[str, Any]) -> None:
        """控制台打印一小段摘要，不刷屏，只给人看一眼当前发生了什么。"""
        try:
            state_panel = display_payload.get("state_panel", {})
            action_panel = display_payload.get("action_panel", {})
            feedback_panel = display_payload.get("feedback_panel", {})
            log_panel = display_payload.get("log_panel", {})

            logging.info(
                "[Visualizer] step=%s | scene=%s/%s | conf=%s | flow_fix=%s | action=%s | result=%s",
                log_panel.get("step_id", ""),
                state_panel.get("scene_type", "unknown"),
                state_panel.get("scene_variant", ""),
                self._fmt_float(state_panel.get("match_confidence")),
                state_panel.get("flow_corrected", False),
                action_panel.get("action_type", "unknown"),
                feedback_panel.get("execute_status", "unknown"),
            )
        except Exception as exc:
            logging.warning("控制台展示摘要失败: %s", exc)

    def _preview_items(self, items: Sequence[Any]) -> List[Any]:
        return list(items[: self.max_preview_items])

    def _fmt_float(self, value: Any) -> str:
        if value is None or value == "":
            return ""
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    def _format_inline_list(self, values: Iterable[Any]) -> str:
        materialized = list(values)
        if not materialized:
            return "[]"
        return json.dumps(materialized, ensure_ascii=False)

    def _format_score_pairs(self, pairs: Sequence[Tuple[Any, Any]]) -> str:
        if not pairs:
            return "[]"
        normalized: List[List[Any]] = []
        for key, score in pairs:
            normalized.append([key, self._fmt_float(score)])
        return json.dumps(normalized, ensure_ascii=False)