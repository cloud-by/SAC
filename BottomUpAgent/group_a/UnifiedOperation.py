from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class UnifiedOperation:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.environment = config.get("environment", {})
        logging.info("UnifiedOperation 初始化完成")

    def normalize(self, action_data: Dict[str, Any], step_id: int = 0) -> Dict[str, Any]:
        raw = dict(action_data) if isinstance(action_data, dict) else {}
        action_type = raw.get("action_type", "wait")

        normalizer = {
            "enter_single_mode": lambda r: self._normalize_named_button(r, "single_mode"),
            "choose_standard_mode": lambda r: self._normalize_named_button(r, "standard_mode"),
            "select_ironclad": lambda r: self._normalize_named_button(r, "ironclad"),
            "confirm_character": lambda r: self._normalize_named_button(r, "confirm_character"),
            "continue_act": lambda r: self._normalize_named_button(r, "continue"),
            "back": lambda r: self._normalize_named_button(r, "back"),
            "click_button": self._normalize_generic_button,
            "play_card": self._normalize_play_card,
            "click_card": self._normalize_click_card,
            "click_enemy": self._normalize_click_enemy,
            "end_turn": self._normalize_end_turn,
            "choose_map_node": self._normalize_choose_map_node,
            "choose_reward": self._normalize_choose_reward,
            "wait": self._normalize_wait,
            "finish": self._normalize_finish,
        }.get(action_type, self._normalize_unknown)

        normalized = normalizer(raw)
        normalized.setdefault("action_type", action_type)
        normalized.setdefault("reason", raw.get("reason", "未提供动作解释"))
        normalized.setdefault("confidence", raw.get("confidence", 0.0))
        normalized.setdefault("source", raw.get("source", "UnifiedOperation"))
        normalized.setdefault("timestamp", raw.get("timestamp", now_str()))
        normalized.setdefault("step_id", step_id)
        normalized.setdefault("execution_plan", [])
        return normalized

    def _normalize_named_button(self, raw: Dict[str, Any], button_name: str) -> Dict[str, Any]:
        return {
            "action_type": raw.get("action_type", "click_button"),
            "reason": raw.get("reason", ""),
            "target": {"button": button_name},
            "execution_plan": [{"op": "click_button", "button_name": button_name}],
            "params": raw.get("params", {}),
        }

    def _normalize_generic_button(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        target = raw.get("target", {}) or {}
        params = raw.get("params", {}) or {}
        button_name = target.get("button", params.get("button_name", "unknown"))
        return {
            "action_type": "click_button",
            "reason": raw.get("reason", ""),
            "target": {"button": button_name},
            "execution_plan": [{"op": "click_button", "button_name": button_name}],
            "params": params,
        }

    def _normalize_play_card(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        target = raw.get("target", {}) or {}
        params = raw.get("params", {}) or {}

        card_name = target.get("card_name") or (target.get("card", {}) or {}).get("name") or params.get("card_name")
        card_bbox = (target.get("card", {}) or {}).get("bbox") or target.get("card_bbox") or params.get("card_bbox")
        card_id = target.get("card_id", params.get("card_id"))
        card_index = target.get("card_index", params.get("card_index"))

        enemy_name = target.get("enemy_name") or (target.get("enemy", {}) or {}).get("name") or params.get("enemy_name")
        enemy_bbox = (target.get("enemy", {}) or {}).get("bbox") or target.get("enemy_bbox") or params.get("enemy_bbox")
        enemy_id = target.get("enemy_id", params.get("enemy_id"))
        enemy_index = target.get("enemy_index", params.get("enemy_index"))

        plan: List[Dict[str, Any]] = [
            {
                "op": "click_card",
                "card_name": card_name,
                "card_bbox": card_bbox,
                "card_id": card_id,
                "card_index": card_index,
            }
        ]
        if enemy_name is not None or enemy_bbox or enemy_id is not None or enemy_index is not None:
            plan.append(
                {
                    "op": "click_enemy",
                    "enemy_name": enemy_name,
                    "enemy_bbox": enemy_bbox,
                    "enemy_id": enemy_id,
                    "enemy_index": enemy_index,
                }
            )

        return {
            "action_type": "play_card",
            "reason": raw.get("reason", ""),
            "target": target,
            "execution_plan": plan,
            "params": params,
        }

    def _normalize_click_card(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        target = raw.get("target", {}) or {}
        params = raw.get("params", {}) or {}
        return {
            "action_type": "click_card",
            "reason": raw.get("reason", ""),
            "target": target,
            "execution_plan": [
                {
                    "op": "click_card",
                    "card_name": target.get("card_name", params.get("card_name")),
                    "card_id": target.get("card_id", params.get("card_id")),
                    "card_index": target.get("card_index", params.get("card_index")),
                    "card_bbox": target.get("card_bbox", params.get("card_bbox")),
                }
            ],
            "params": params,
        }

    def _normalize_click_enemy(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        target = raw.get("target", {}) or {}
        params = raw.get("params", {}) or {}
        return {
            "action_type": "click_enemy",
            "reason": raw.get("reason", ""),
            "target": target,
            "execution_plan": [
                {
                    "op": "click_enemy",
                    "enemy_name": target.get("enemy_name", params.get("enemy_name")),
                    "enemy_id": target.get("enemy_id", params.get("enemy_id")),
                    "enemy_index": target.get("enemy_index", params.get("enemy_index")),
                    "enemy_bbox": target.get("enemy_bbox", params.get("enemy_bbox")),
                }
            ],
            "params": params,
        }

    def _normalize_end_turn(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action_type": "end_turn",
            "reason": raw.get("reason", ""),
            "target": raw.get("target", {"button": "end_turn"}),
            "execution_plan": [{"op": "click_button", "button_name": "end_turn"}],
            "params": raw.get("params", {}),
        }

    def _normalize_choose_map_node(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        target = raw.get("target", {}) or {}
        params = raw.get("params", {}) or {}
        return {
            "action_type": "choose_map_node",
            "reason": raw.get("reason", ""),
            "target": target,
            "execution_plan": [
                {
                    "op": "click_map_node",
                    "node_id": target.get("id", params.get("node_id")),
                    "node_name": target.get("name", params.get("node_name")),
                    "node_index": target.get("index", params.get("node_index")),
                    "node_bbox": target.get("bbox", params.get("node_bbox")),
                }
            ],
            "params": params,
        }

    def _normalize_choose_reward(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        target = raw.get("target", {}) or {}
        params = raw.get("params", {}) or {}
        return {
            "action_type": "choose_reward",
            "reason": raw.get("reason", ""),
            "target": target,
            "execution_plan": [
                {
                    "op": "click_reward",
                    "reward_id": target.get("id", params.get("reward_id")),
                    "reward_name": target.get("name", params.get("reward_name")),
                    "reward_index": target.get("index", params.get("reward_index")),
                    "reward_bbox": target.get("bbox", params.get("reward_bbox")),
                }
            ],
            "params": params,
        }

    def _normalize_wait(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        params = raw.get("params", {}) or {}
        duration = float(params.get("duration", 0.5))
        return {
            "action_type": "wait",
            "reason": raw.get("reason", ""),
            "target": raw.get("target", {}),
            "execution_plan": [{"op": "wait", "duration": duration}],
            "params": raw.get("params", {"duration": duration}),
        }

    def _normalize_finish(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action_type": "finish",
            "reason": raw.get("reason", ""),
            "target": raw.get("target", {}),
            "execution_plan": [],
            "params": raw.get("params", {}),
        }

    def _normalize_unknown(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        logging.warning("收到未知动作类型，自动降级为 wait: %s", raw.get("action_type"))
        return {
            "action_type": "wait",
            "reason": raw.get("reason", "unknown action fallback to wait"),
            "target": raw.get("target", {}),
            "execution_plan": [{"op": "wait", "duration": 0.3}],
            "params": {"duration": 0.3},
        }