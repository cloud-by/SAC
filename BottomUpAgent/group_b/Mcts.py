"""
BottomUpAgent/group_b/Mcts.py

B组候选策略搜索模块：
1. 接收 Brain 输出的初始动作
2. 基于当前状态生成少量候选动作
3. 对候选动作进行简单评分与选择
4. 输出优化后的 action_data

说明：
当前是“课程演示版轻量搜索器”，不是完整的蒙特卡洛树搜索。
现在先把全场景候选动作和简单排序补齐，让系统不至于在营火和商店前面当机。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple


SCENE_ALIASES = {
    "single_mode_menu": "mode_select",
    "act_intro": "blessing_choice",
    "reward": "blessing_choice",
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Mcts:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.runtime_context = config.setdefault("_runtime_context", {})
        logging.info("Mcts 初始化完成（轻量版）")

    def search(
        self,
        action_data: Dict[str, Any],
        state_data: Dict[str, Any],
        step_id: int,
    ) -> Dict[str, Any]:
        candidates = self._generate_candidates(action_data, state_data)
        if not candidates:
            return self._mark_source(action_data, state_data)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates:
            score = self._score_candidate(candidate, state_data)
            scored.append((score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_action = scored[0]
        best_action = dict(best_action)
        best_action["source"] = self._merge_source(best_action.get("source", "Brain"), "Mcts")
        best_action["confidence"] = round(
            max(float(best_action.get("confidence", 0.0)), min(0.95, best_score)),
            2,
        )
        best_action["timestamp"] = now_str()
        best_action.setdefault("params", {})
        best_action["params"]["mcts_top_score"] = round(best_score, 4)
        best_action["params"]["mcts_candidate_count"] = len(candidates)
        best_action["episode_id"] = self._resolve_episode_id(state_data, best_action)
        best_action["params"].setdefault("episode_id", best_action["episode_id"])
        return best_action

    def select(self, action_data: Dict[str, Any], state_data: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        return self.search(action_data, state_data, step_id)

    def refine(self, action_data: Dict[str, Any], state_data: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        return self.search(action_data, state_data, step_id)

    def _generate_candidates(self, action_data: Dict[str, Any], state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        scene_type = self._normalize_scene(state_data.get("scene_type", "unknown"))
        base_action = dict(action_data)
        episode_id = self._resolve_episode_id(state_data, base_action)
        if episode_id:
            base_action["episode_id"] = episode_id
            base_action.setdefault("params", {})
            base_action["params"].setdefault("episode_id", episode_id)

        candidates: List[Dict[str, Any]] = [base_action]

        if scene_type == "battle":
            candidates.extend(self._battle_candidates(state_data))
        elif scene_type == "map":
            candidates.extend(self._map_candidates(state_data))
        elif scene_type in {"blessing_choice", "card_reward", "event_unknown"}:
            candidates.extend(self._choice_candidates(scene_type, state_data))
        elif scene_type == "merchant_shop":
            candidates.extend(self._merchant_shop_candidates(state_data))
        elif scene_type == "campfire_choice":
            candidates.extend(self._campfire_choice_candidates(state_data))
        elif scene_type == "campfire_upgrade":
            candidates.extend(self._campfire_upgrade_candidates(state_data))
        elif scene_type in {"battle_result", "merchant", "campfire_rest_done"}:
            candidates.extend(self._advance_candidates(state_data))
        elif scene_type in {"title_main", "mode_select", "character_select"}:
            candidates.extend(self._bootstrap_candidates(scene_type))

        unique_candidates: List[Dict[str, Any]] = []
        seen = set()
        for item in candidates:
            key = (
                item.get("action_type"),
                str(item.get("target")),
                item.get("reason"),
            )
            if key in seen:
                continue
            seen.add(key)
            if episode_id:
                item["episode_id"] = episode_id
                item.setdefault("params", {})
                item["params"].setdefault("episode_id", episode_id)
            unique_candidates.append(item)
        return unique_candidates

    def _bootstrap_candidates(self, scene_type: str) -> List[Dict[str, Any]]:
        mapping = {
            "title_main": ("enter_single_mode", "single_mode", "Mcts 候选：进入单人模式。", 0.90),
            "mode_select": ("choose_standard_mode", "standard_mode", "Mcts 候选：选择标准模式。", 0.89),
            "character_select": ("confirm_character", "confirm_character", "Mcts 候选：确认当前角色。", 0.75),
        }
        action_type, button, reason, confidence = mapping[scene_type]
        return [
            {
                "action_type": action_type,
                "target": {"button": button},
                "reason": reason,
                "confidence": confidence,
                "source": "Brain",
                "timestamp": now_str(),
                "params": {},
            }
        ]

    def _battle_candidates(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        hand_cards = list(state_data.get("hand_cards", []) or [])
        enemies = list(state_data.get("enemies", []) or [])
        energy = self._to_int(state_data.get("energy"), default=0)
        target_enemy = enemies[0] if enemies else {}

        for card in hand_cards:
            cost = self._to_int(card.get("cost"), default=99)
            if cost is None or cost > energy:
                continue

            card_type = str(card.get("type", "") or "").lower()
            if card_type == "attack":
                result.append(
                    {
                        "action_type": "play_card",
                        "target": {"card": card, "enemy": target_enemy},
                        "reason": f"Mcts 候选：使用 {card.get('name', 'attack_card')} 进行攻击。",
                        "confidence": 0.70,
                        "source": "Brain",
                        "timestamp": now_str(),
                        "params": {
                            "card_name": card.get("name", "unknown_card"),
                            "enemy_name": target_enemy.get("name", "unknown_enemy"),
                        },
                    }
                )
            elif card_type in {"skill", "power"}:
                result.append(
                    {
                        "action_type": "play_card",
                        "target": {"card": card, "enemy": None},
                        "reason": f"Mcts 候选：使用 {card.get('name', 'skill_card')} 做防御或辅助。",
                        "confidence": 0.66,
                        "source": "Brain",
                        "timestamp": now_str(),
                        "params": {"card_name": card.get("name", "unknown_card")},
                    }
                )

        result.append(
            {
                "action_type": "end_turn",
                "target": {"button": "end_turn"},
                "reason": "Mcts 候选：结束当前回合。",
                "confidence": 0.45,
                "source": "Brain",
                "timestamp": now_str(),
                "params": {},
            }
        )
        return result

    def _map_candidates(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        map_options = list(state_data.get("map_options", []) or [])
        hp_ratio = self._safe_ratio(state_data.get("hp"), state_data.get("max_hp"))
        ordered = sorted(map_options, key=lambda x: self._score_map_kind(x, hp_ratio), reverse=True)

        for option in ordered[:3]:
            result.append(
                {
                    "action_type": "choose_map_node",
                    "target": option,
                    "reason": f"Mcts 候选：选择地图节点 {option.get('id', 'node')}（{option.get('kind', 'unknown')}）。",
                    "confidence": 0.62,
                    "source": "Brain",
                    "timestamp": now_str(),
                    "params": {"node_id": option.get("id", "node_0"), "node_kind": option.get("kind", "unknown")},
                }
            )
        return result

    def _choice_candidates(self, scene_type: str, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        options = self._collect_options(state_data)
        ranked = sorted(options, key=self._choice_sort_key, reverse=True)
        for option in ranked[:3]:
            result.append(
                {
                    "action_type": "choose_reward",
                    "target": option,
                    "reason": f"Mcts 候选：在 {scene_type} 场景选择 {option.get('name', option.get('id', 'option'))}。",
                    "confidence": 0.64,
                    "source": "Brain",
                    "timestamp": now_str(),
                    "params": {
                        "reward_id": option.get("id"),
                        "reward_name": option.get("name"),
                        "reward_bbox": option.get("bbox"),
                    },
                }
            )
        return result

    def _merchant_shop_candidates(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        items = list(state_data.get("shop_items", []) or [])
        for item in items[:3]:
            result.append(
                {
                    "action_type": "choose_reward",
                    "target": item,
                    "reason": f"Mcts 候选：购买商品 {item.get('name', item.get('id', 'shop_item'))}。",
                    "confidence": 0.58,
                    "source": "Brain",
                    "timestamp": now_str(),
                    "params": {
                        "reward_id": item.get("id"),
                        "reward_name": item.get("name"),
                        "reward_bbox": item.get("bbox"),
                    },
                }
            )
        result.extend(self._advance_candidates(state_data))
        return result

    def _campfire_choice_candidates(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        options = list(state_data.get("campfire_options", []) or [])
        hp_ratio = self._safe_ratio(state_data.get("hp"), state_data.get("max_hp"))
        if hp_ratio is not None and hp_ratio < 0.55:
            options = sorted(options, key=lambda x: self._campfire_choice_score(x, low_hp=True), reverse=True)
        else:
            options = sorted(options, key=lambda x: self._campfire_choice_score(x, low_hp=False), reverse=True)

        for option in options[:2]:
            result.append(
                {
                    "action_type": "choose_reward",
                    "target": option,
                    "reason": f"Mcts 候选：营火选择 {option.get('name', option.get('id', 'campfire_option'))}。",
                    "confidence": 0.66,
                    "source": "Brain",
                    "timestamp": now_str(),
                    "params": {
                        "reward_id": option.get("id"),
                        "reward_name": option.get("name"),
                        "reward_bbox": option.get("bbox"),
                    },
                }
            )
        return result

    def _campfire_upgrade_candidates(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        options = list(state_data.get("upgrade_options", []) or [])
        for option in options[:3]:
            result.append(
                {
                    "action_type": "choose_reward",
                    "target": option,
                    "reason": f"Mcts 候选：升级卡牌 {option.get('name', option.get('id', 'upgrade_option'))}。",
                    "confidence": 0.65,
                    "source": "Brain",
                    "timestamp": now_str(),
                    "params": {
                        "reward_id": option.get("id"),
                        "reward_name": option.get("name"),
                        "reward_bbox": option.get("bbox"),
                    },
                }
            )
        return result

    def _advance_candidates(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        if self._has_continue(state_data):
            result.append(
                {
                    "action_type": "continue_act",
                    "target": {"button": "continue"},
                    "reason": "Mcts 候选：点击继续推进流程。",
                    "confidence": 0.70,
                    "source": "Brain",
                    "timestamp": now_str(),
                    "params": {},
                }
            )
        if self._has_back(state_data):
            result.append(
                {
                    "action_type": "back",
                    "target": {"button": "back"},
                    "reason": "Mcts 候选：点击返回离开当前页面。",
                    "confidence": 0.42,
                    "source": "Brain",
                    "timestamp": now_str(),
                    "params": {},
                }
            )
        return result

    def _score_candidate(self, candidate: Dict[str, Any], state_data: Dict[str, Any]) -> float:
        scene_type = self._normalize_scene(state_data.get("scene_type", "unknown"))
        action_type = candidate.get("action_type", "wait")
        hp_ratio = self._safe_ratio(state_data.get("hp"), state_data.get("max_hp"))
        energy = self._to_int(state_data.get("energy"), default=0)
        score = float(candidate.get("confidence", 0.3))

        if scene_type == "battle":
            low_hp = hp_ratio is not None and hp_ratio < 0.4
            if action_type == "play_card":
                target = candidate.get("target", {}) if isinstance(candidate.get("target"), dict) else {}
                card = target.get("card", {}) if isinstance(target, dict) else {}
                card_type = str(card.get("type", "") or "").lower()
                cost = self._to_int(card.get("cost"), default=99)
                card_name = str(card.get("name", "") or "").lower()

                if low_hp and card_type == "skill":
                    score += 0.18
                elif (not low_hp) and card_type == "attack":
                    score += 0.18
                else:
                    score += 0.05

                if card_name == "bash":
                    score += 0.06
                if cost <= energy:
                    score += 0.04
            elif action_type == "end_turn":
                if energy == 0:
                    score += 0.08
                else:
                    score -= 0.12

        elif scene_type == "map":
            if action_type == "choose_map_node":
                option = candidate.get("target", {}) if isinstance(candidate.get("target"), dict) else {}
                score += self._score_map_kind(option, hp_ratio)

        elif scene_type in {"blessing_choice", "card_reward", "event_unknown"}:
            if action_type == "choose_reward":
                option = candidate.get("target", {}) if isinstance(candidate.get("target"), dict) else {}
                score += self._choice_bonus(option)
                if scene_type == "card_reward":
                    score += 0.10
                elif scene_type == "blessing_choice":
                    score += 0.08
                else:
                    score += 0.05

        elif scene_type == "merchant_shop":
            if action_type == "choose_reward":
                score += 0.08
            elif action_type in {"continue_act", "back"}:
                score += 0.03

        elif scene_type == "campfire_choice":
            if action_type == "choose_reward":
                option = candidate.get("target", {}) if isinstance(candidate.get("target"), dict) else {}
                score += self._campfire_choice_score(option, low_hp=(hp_ratio is not None and hp_ratio < 0.55))

        elif scene_type == "campfire_upgrade":
            if action_type == "choose_reward":
                score += 0.12

        elif scene_type in {"battle_result", "merchant", "campfire_rest_done"}:
            if action_type == "continue_act":
                score += 0.14
            elif action_type == "back":
                score += 0.02

        elif scene_type in {"title_main", "mode_select", "character_select"}:
            if action_type in {"enter_single_mode", "choose_standard_mode", "confirm_character", "select_ironclad"}:
                score += 0.18

        if action_type == "wait":
            score -= 0.15

        return round(max(0.0, min(0.99, score)), 2)

    def _score_map_kind(self, option: Dict[str, Any], hp_ratio: float | None) -> float:
        kind = str(option.get("kind", "unknown") or "unknown").lower()
        base = {
            "event_unknown": 0.18,
            "merchant": 0.12,
            "rest": 0.10,
            "battle": 0.08,
            "elite": -0.04,
            "boss": -0.08,
            "unknown": 0.0,
        }.get(kind, 0.0)
        if hp_ratio is not None:
            if hp_ratio < 0.45:
                if kind == "rest":
                    base += 0.24
                if kind in {"elite", "boss"}:
                    base -= 0.18
            elif hp_ratio > 0.75 and kind == "event_unknown":
                base += 0.06
        base += self._to_float(option.get("confidence"), default=0.0) * 0.08
        return round(base, 4)

    def _choice_bonus(self, option: Dict[str, Any]) -> float:
        name = str(option.get("name", "") or "").lower()
        confidence = self._to_float(option.get("confidence"), default=0.0)
        bonus = confidence * 0.10
        if any(word in name for word in ["skip", "leave", "cancel", "ignore"]):
            bonus -= 0.08
        return round(bonus, 4)

    def _campfire_choice_score(self, option: Dict[str, Any], low_hp: bool) -> float:
        name = str(option.get("name", "") or "").lower()
        if low_hp:
            if name in {"rest", "heal", "recover"}:
                return 0.22
            if name in {"upgrade", "smith", "forge"}:
                return 0.02
            return 0.08
        else:
            if name in {"upgrade", "smith", "forge"}:
                return 0.20
            if name in {"rest", "heal", "recover"}:
                return 0.04
            return 0.08

    def _collect_options(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        buckets = [
            state_data.get("reward_options", []),
            state_data.get("menu_options", []),
            state_data.get("mode_options", []),
            state_data.get("character_options", []),
            state_data.get("campfire_options", []),
            state_data.get("upgrade_options", []),
            state_data.get("shop_items", []),
        ]
        result: List[Dict[str, Any]] = []
        for bucket in buckets:
            if isinstance(bucket, list):
                result.extend([x for x in bucket if isinstance(x, dict)])
        return result

    def _choice_sort_key(self, option: Dict[str, Any]) -> float:
        return self._to_float(option.get("confidence"), default=0.5) + self._choice_bonus(option)

    def _mark_source(self, action_data: Dict[str, Any], state_data: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(action_data)
        result["source"] = self._merge_source(result.get("source", "Brain"), "Mcts")
        result["timestamp"] = now_str()
        eid = self._resolve_episode_id(state_data, result)
        if eid:
            result["episode_id"] = eid
            result.setdefault("params", {})
            result["params"].setdefault("episode_id", eid)
        return result

    def _merge_source(self, source: str, suffix: str) -> str:
        if suffix in str(source):
            return str(source)
        return f"{source}+{suffix}"

    def _normalize_scene(self, scene_type: Any) -> str:
        raw = str(scene_type or "unknown")
        return SCENE_ALIASES.get(raw, raw)

    def _resolve_episode_id(self, state_data: Dict[str, Any], action_data: Dict[str, Any]) -> str | None:
        return (
            action_data.get("episode_id")
            or state_data.get("episode_id")
            or self.runtime_context.get("current_episode_id")
            or self.runtime_context.get("episode_id")
        )

    def _has_continue(self, state_data: Dict[str, Any]) -> bool:
        if state_data.get("continue_bbox"):
            return True
        available = [str(x).lower() for x in state_data.get("available_buttons", []) or []]
        return "continue" in available

    def _has_back(self, state_data: Dict[str, Any]) -> bool:
        if state_data.get("back_button_bbox"):
            return True
        available = [str(x).lower() for x in state_data.get("available_buttons", []) or []]
        return "back" in available

    def _to_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _safe_ratio(self, hp: Any, max_hp: Any) -> float | None:
        hp_i = self._to_int(hp, default=-1)
        max_i = self._to_int(max_hp, default=-1)
        if hp_i < 0 or max_i <= 0:
            return None
        return round(hp_i / max_i, 4)
