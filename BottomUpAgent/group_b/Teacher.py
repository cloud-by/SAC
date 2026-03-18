"""
BottomUpAgent/Teacher.py

B组评估与反思模块：
1. 对动作执行结果进行效果评估
2. 输出反思文字与评分
3. 决定本次动作是否适合纳入经验库
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Teacher:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        logging.info("Teacher 初始化完成")

    def reflect(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        step_id: int,
    ) -> Dict[str, Any]:
        score = self._score_action(
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
        )
        feedback_text = self._build_feedback_text(
            state_data=state_data,
            action_data=action_data,
            feedback_data=feedback_data,
            score=score,
        )

        return {
            "step_id": step_id,
            "score": score,
            "feedback": feedback_text,
            "should_promote_to_skill": score >= 0.5,
            "timestamp": now_str(),
        }

    def review(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        step_id: int,
    ) -> Dict[str, Any]:
        return self.reflect(state_data, action_data, feedback_data, step_id)

    def evaluate(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        step_id: int,
    ) -> Dict[str, Any]:
        return self.reflect(state_data, action_data, feedback_data, step_id)

    def _score_action(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
    ) -> float:
        """
        简单评分逻辑：
        - 执行成功基础分较高
        - scene 有合理变化加分
        - screen_diff 有明显变化加分
        - 失败则降分
        """
        execute_status = feedback_data.get("execute_status", "unknown")
        before_scene = feedback_data.get("before_scene", "unknown")
        after_scene = feedback_data.get("after_scene", "unknown")
        screen_diff = feedback_data.get("screen_diff", "")
        error_message = feedback_data.get("error_message", "")

        if execute_status != "success":
            return 0.1 if error_message else 0.2

        score = 0.6

        if before_scene != after_scene:
            score += 0.15

        if screen_diff and screen_diff not in {"no_obvious_change", "no_real_execution"}:
            score += 0.15

        action_type = action_data.get("action_type", "wait")
        if action_type == "wait":
            score -= 0.15

        if action_type == "finish":
            score += 0.05

        score = max(0.0, min(1.0, score))
        return round(score, 2)

    def _build_feedback_text(
        self,
        state_data: Dict[str, Any],
        action_data: Dict[str, Any],
        feedback_data: Dict[str, Any],
        score: float,
    ) -> str:
        action_type = action_data.get("action_type", "unknown_action")
        execute_status = feedback_data.get("execute_status", "unknown")
        screen_diff = feedback_data.get("screen_diff", "")
        error_message = feedback_data.get("error_message", "")

        if execute_status != "success":
            if error_message:
                return f"动作 {action_type} 执行失败，异常信息：{error_message}。建议后续降低该策略优先级。"
            return f"动作 {action_type} 执行未成功，建议重新检查目标定位与动作参数。"

        if score >= 0.8:
            return f"动作 {action_type} 执行效果较好，界面变化为 {screen_diff}，适合纳入经验库。"

        if score >= 0.5:
            return f"动作 {action_type} 执行成功，但收益一般，建议保留为可选经验。"

        return f"动作 {action_type} 虽已执行，但效果不明显，建议谨慎复用。"