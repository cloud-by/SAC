"""
BottomUpAgent/group_b/Brain.py

B组决策模块：
1. 接收 A组 提供的 state_data
2. 根据 scene_type / scene_variant / 可交互元素生成结构化动作
3. 输出动作原因、置信度与来源说明
4. 保持与 UnifiedOperation / Hand 当前接口兼容
5. 可选接入 base_model 作为辅助分析层（不强依赖）

说明：
当前版本优先把“全场景流”打通，先让系统别在商店和营火前面犯迷糊。
复杂学习策略后面再叠，不急着把课程项目做成黑箱神谕机。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from BottomUpAgent.common.protocols import ensure_action_protocol, ensure_state_protocol

try:
    from base_model import build_model
except Exception:
    build_model = None


SCENE_ALIASES = {
    "single_mode_menu": "mode_select",
    "act_intro": "blessing_choice",
    "reward": "blessing_choice",
    "victory": "victory",
    "death": "death",
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Brain:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_config = config.get("model", {})
        self.runtime_config = config.get("runtime", {})
        self.runtime_context = config.setdefault("_runtime_context", {})

        self.provider = self.model_config.get("provider", "mock")
        self.model_name = self.model_config.get("name", "demo-model")
        self.enable_llm_assist = bool(self.runtime_config.get("enable_llm_assist", True))
        self.attach_llm_reason = bool(self.runtime_config.get("attach_llm_reason_to_reason", False))

        self.llm = None
        self.llm_enabled = False

        if self.enable_llm_assist and callable(build_model):
            try:
                self.llm = build_model(config)
                self.llm_enabled = True
                logging.info(
                    "Brain LLM 初始化完成，provider=%s, model=%s",
                    self.provider,
                    self.model_name,
                )
            except Exception as exc:
                self.llm = None
                self.llm_enabled = False
                logging.warning("Brain LLM 初始化失败，自动退回规则模式: %s", exc)
        else:
            logging.info(
                "Brain 初始化完成，provider=%s, model=%s, llm_assist=%s",
                self.provider,
                self.model_name,
                self.enable_llm_assist,
            )

    def plan(
        self,
        task: str,
        state_data: Dict[str, Any],
        memory_summary: Dict[str, Any],
        step_id: int,
    ) -> Dict[str, Any]:
        state_data = ensure_state_protocol(state_data, step_id=step_id,
                                           phase=str(state_data.get("phase", "before") or "before"))
        scene_type = self._normalize_scene(state_data.get("scene_type", "unknown"))
        episode_id = self._resolve_episode_id(state_data)

        dispatch = {
            "title_main": self._plan_title_main,
            "mode_select": self._plan_mode_select,
            "character_select": self._plan_character_select,
            "blessing_choice": self._plan_blessing_choice,
            "map": self._plan_map,
            "battle": self._plan_battle,
            "battle_result": self._plan_battle_result,
            "card_reward": self._plan_card_reward,
            "merchant": self._plan_merchant,
            "merchant_shop": self._plan_merchant_shop,
            "campfire_choice": self._plan_campfire_choice,
            "campfire_upgrade": self._plan_campfire_upgrade,
            "campfire_rest_done": self._plan_campfire_rest_done,
            "event_unknown": self._plan_event_unknown,
            "victory": self._plan_terminal,
            "death": self._plan_terminal,
        }

        planner = dispatch.get(scene_type, self._plan_unknown)
        action = planner(task, state_data, memory_summary, step_id)
        action = self._normalize_action(action)
        action = ensure_action_protocol(
            action,
            scene_type=scene_type,
            episode_id=episode_id,
            step_id=step_id,
        )

        llm_analysis = self._try_llm_assist(
            task=task,
            state_data=state_data,
            memory_summary=memory_summary,
            step_id=step_id,
            current_action=action,
        )
        if llm_analysis:
            action.setdefault("params", {})
            action["params"]["llm_analysis"] = llm_analysis
            action["params"]["llm_provider"] = self.provider
            action["params"]["llm_model"] = self.model_name

            if self.attach_llm_reason:
                short_text = llm_analysis.strip().replace("\n", " ")
                if len(short_text) > 120:
                    short_text = short_text[:120] + "..."
                action["reason"] = f"{action['reason']}（LLM辅助分析：{short_text}）"

        return action

    # ---------- scene planners ----------

    def _plan_title_main(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        return {
            "action_type": "enter_single_mode",
            "target": {"button": "single_mode"},
            "reason": "当前位于主页，优先进入单人模式以开始一局游戏。",
            "confidence": 0.96,
            "source": "Brain-rule-title_main",
            "timestamp": now_str(),
            "params": {},
        }

    def _plan_mode_select(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        return {
            "action_type": "choose_standard_mode",
            "target": {"button": "standard_mode"},
            "reason": "当前位于模式选择页，默认进入标准模式。",
            "confidence": 0.94,
            "source": "Brain-rule-mode_select",
            "timestamp": now_str(),
            "params": {},
        }

    def _plan_character_select(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        selected_character = str(state_data.get("selected_character", "") or "").lower()
        if selected_character != "ironclad":
            return {
                "action_type": "select_ironclad",
                "target": {"button": "ironclad"},
                "reason": "当前位于角色选择页，优先选择铁甲战士作为演示角色。",
                "confidence": 0.93,
                "source": "Brain-rule-character_select",
                "timestamp": now_str(),
                "params": {"character": "ironclad"},
            }

        return {
            "action_type": "confirm_character",
            "target": {"button": "confirm_character"},
            "reason": "铁甲战士已经被选中，继续确认角色进入开局流程。",
            "confidence": 0.95,
            "source": "Brain-rule-character_select",
            "timestamp": now_str(),
            "params": {"character": "ironclad"},
        }

    def _plan_blessing_choice(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        options = self._collect_choice_options(state_data)
        if options:
            option = self._pick_best_choice(options)
            return self._make_choose_reward_action(
                option=option,
                reason="当前处于开局三选一/赐福选择场景，先选择一个可执行选项继续推进。",
                confidence=0.76,
                source="Brain-rule-blessing_choice",
            )
        return self._make_continue_or_wait(state_data, scene_type="blessing_choice", source="Brain-rule-blessing_choice")

    def _plan_map(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        map_options = list(state_data.get("map_options", []) or [])
        if not map_options:
            return self._make_continue_or_wait(state_data, scene_type="map", source="Brain-rule-map")

        hp_ratio = self._safe_ratio(state_data.get("hp"), state_data.get("max_hp"))
        ordered = sorted(map_options, key=lambda x: self._score_map_option(x, hp_ratio), reverse=True)
        option = ordered[0]
        return {
            "action_type": "choose_map_node",
            "target": option,
            "reason": f"当前处于地图场景，按演示策略优先选择 {option.get('kind', 'unknown')} 节点继续推进。",
            "confidence": 0.78,
            "source": "Brain-rule-map",
            "timestamp": now_str(),
            "params": {
                "node_id": option.get("id", "node_0"),
                "node_kind": option.get("kind", "unknown"),
            },
        }

    def _plan_battle(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        hp = state_data.get("hp")
        max_hp = state_data.get("max_hp")
        energy = self._to_int(state_data.get("energy"), default=0)
        hand_cards = list(state_data.get("hand_cards", []) or [])
        enemies = list(state_data.get("enemies", []) or [])

        low_hp = False
        if isinstance(hp, int) and isinstance(max_hp, int) and max_hp > 0:
            low_hp = hp / max_hp < 0.4

        playable_cards = [card for card in hand_cards if self._to_int(card.get("cost"), default=99) <= energy]
        defend_card = self._find_card(playable_cards, card_type="skill", card_name="Defend")
        bash_card = self._find_card(playable_cards, card_type="attack", card_name="Bash")
        strike_card = self._find_card(playable_cards, card_type="attack", card_name="Strike")
        any_attack = self._find_card(playable_cards, card_type="attack")
        any_skill = self._find_card(playable_cards, card_type="skill")
        target_enemy = self._pick_enemy(enemies)

        if low_hp and defend_card:
            return self._make_play_card_action(
                card=defend_card,
                enemy=None,
                reason="当前生命值偏低，优先使用防御牌降低风险。",
                confidence=0.84,
            )

        if bash_card:
            return self._make_play_card_action(
                card=bash_card,
                enemy=target_enemy,
                reason="当前能量允许，优先使用更高价值的攻击牌推进战斗。",
                confidence=0.87,
            )

        if strike_card:
            return self._make_play_card_action(
                card=strike_card,
                enemy=target_enemy,
                reason="当前能量足够，优先使用攻击牌推进战斗节奏。",
                confidence=0.85,
            )

        if not low_hp and any_attack:
            return self._make_play_card_action(
                card=any_attack,
                enemy=target_enemy,
                reason="当前有可用攻击牌，优先造成伤害。",
                confidence=0.80,
            )

        if any_skill:
            return self._make_play_card_action(
                card=any_skill,
                enemy=None,
                reason="当前没有更好的攻击牌，先使用技能牌争取优势。",
                confidence=0.72,
            )

        return {
            "action_type": "end_turn",
            "target": {"button": "end_turn"},
            "reason": "当前没有可稳定执行的高收益牌，结束回合。",
            "confidence": 0.60,
            "source": "Brain-rule-battle",
            "timestamp": now_str(),
            "params": {},
        }

    def _plan_battle_result(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        return self._make_continue_or_wait(state_data, scene_type="battle_result", source="Brain-rule-battle_result")

    def _plan_card_reward(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        options = self._collect_choice_options(state_data)
        if options:
            option = self._pick_best_choice(options)
            return self._make_choose_reward_action(
                option=option,
                reason="当前处于战斗后选牌奖励场景，默认选择第一张可用奖励牌。",
                confidence=0.78,
                source="Brain-rule-card_reward",
            )
        return self._make_continue_or_wait(state_data, scene_type="card_reward", source="Brain-rule-card_reward")

    def _plan_merchant(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        options = self._collect_choice_options(state_data)
        if options:
            option = options[0]
            return self._make_choose_reward_action(
                option=option,
                reason="当前处于商人遭遇页，先进入可交互选项继续流程。",
                confidence=0.68,
                source="Brain-rule-merchant",
            )
        return self._make_continue_or_wait(state_data, scene_type="merchant", source="Brain-rule-merchant")

    def _plan_merchant_shop(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        shop_items = list(state_data.get("shop_items", []) or [])
        if shop_items:
            item = shop_items[0]
            return self._make_choose_reward_action(
                option=item,
                reason="当前处于商店购买页，先选择一个可点击商品作为演示策略。",
                confidence=0.66,
                source="Brain-rule-merchant_shop",
            )
        return self._make_continue_or_wait(state_data, scene_type="merchant_shop", source="Brain-rule-merchant_shop")

    def _plan_campfire_choice(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        options = list(state_data.get("campfire_options", []) or [])
        if options:
            hp_ratio = self._safe_ratio(state_data.get("hp"), state_data.get("max_hp"))
            if hp_ratio is not None and hp_ratio < 0.55:
                preferred = self._find_named_option(options, {"rest", "heal", "recover"})
                if preferred is None:
                    preferred = options[0]
                return self._make_choose_reward_action(
                    option=preferred,
                    reason="当前生命值偏低，营火优先休息回血。",
                    confidence=0.82,
                    source="Brain-rule-campfire_choice",
                )

            preferred = self._find_named_option(options, {"upgrade", "smith", "forge"})
            if preferred is None:
                preferred = options[-1]
            return self._make_choose_reward_action(
                option=preferred,
                reason="当前生命值尚可，营火优先选择锻造/升级提升战力。",
                confidence=0.80,
                source="Brain-rule-campfire_choice",
            )
        return self._make_continue_or_wait(state_data, scene_type="campfire_choice", source="Brain-rule-campfire_choice")

    def _plan_campfire_upgrade(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        options = list(state_data.get("upgrade_options", []) or [])
        if options:
            option = options[0]
            return self._make_choose_reward_action(
                option=option,
                reason="当前处于锻造/升级页面，默认选择第一张可升级卡牌。",
                confidence=0.77,
                source="Brain-rule-campfire_upgrade",
            )
        return self._make_continue_or_wait(state_data, scene_type="campfire_upgrade", source="Brain-rule-campfire_upgrade")

    def _plan_campfire_rest_done(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        return self._make_continue_or_wait(state_data, scene_type="campfire_rest_done", source="Brain-rule-campfire_rest_done")

    def _plan_event_unknown(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        options = self._collect_choice_options(state_data)
        if options:
            option = self._pick_best_choice(options)
            return self._make_choose_reward_action(
                option=option,
                reason="当前处于问号事件页面，优先选择一个可执行选项继续推进。",
                confidence=0.70,
                source="Brain-rule-event_unknown",
            )
        return self._make_continue_or_wait(state_data, scene_type="event_unknown", source="Brain-rule-event_unknown")

    def _plan_terminal(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        return {
            "action_type": "finish",
            "target": {"scene_type": self._normalize_scene(state_data.get("scene_type", "unknown"))},
            "reason": "当前已经到达终局场景，结束本次任务。",
            "confidence": 0.98,
            "source": "Brain-rule-terminal",
            "timestamp": now_str(),
            "params": {},
        }

    def _plan_unknown(self, task: str, state_data: Dict[str, Any], memory_summary: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        if self._has_continue(state_data):
            return {
                "action_type": "continue_act",
                "target": {"button": "continue"},
                "reason": "当前场景尚未建立明确策略，但检测到继续按钮，先尝试继续。",
                "confidence": 0.35,
                "source": "Brain-rule-unknown",
                "timestamp": now_str(),
                "params": {},
            }

        return {
            "action_type": "wait",
            "target": {"scene_type": self._normalize_scene(state_data.get("scene_type", "unknown"))},
            "reason": "当前场景尚未建立明确策略，先等待并记录状态。",
            "confidence": 0.15,
            "source": "Brain-rule-unknown",
            "timestamp": now_str(),
            "params": {"duration": 0.5},
        }

    # ---------- helpers ----------

    def _make_play_card_action(self, card: Dict[str, Any], enemy: Optional[Dict[str, Any]], reason: str, confidence: float) -> Dict[str, Any]:
        return {
            "action_type": "play_card",
            "target": {
                "card": card,
                "enemy": enemy,
            },
            "reason": reason,
            "confidence": confidence,
            "source": "Brain-rule-battle",
            "timestamp": now_str(),
            "params": {
                "card_name": card.get("name"),
                "cost": card.get("cost"),
                "enemy_name": (enemy or {}).get("name"),
            },
        }

    def _make_choose_reward_action(self, option: Dict[str, Any], reason: str, confidence: float, source: str) -> Dict[str, Any]:
        return {
            "action_type": "choose_reward",
            "target": option,
            "reason": reason,
            "confidence": confidence,
            "source": source,
            "timestamp": now_str(),
            "params": {
                "reward_id": option.get("id"),
                "reward_name": option.get("name"),
                "reward_bbox": option.get("bbox"),
            },
        }

    def _make_continue_or_wait(self, state_data: Dict[str, Any], scene_type: str, source: str) -> Dict[str, Any]:
        if self._has_continue(state_data):
            return {
                "action_type": "continue_act",
                "target": {"button": "continue"},
                "reason": f"当前处于 {scene_type} 场景，优先点击继续推进流程。",
                "confidence": 0.74,
                "source": source,
                "timestamp": now_str(),
                "params": {},
            }
        if self._has_back(state_data):
            return {
                "action_type": "back",
                "target": {"button": "back"},
                "reason": f"当前处于 {scene_type} 场景，未检测到更优交互，先尝试返回。",
                "confidence": 0.42,
                "source": source,
                "timestamp": now_str(),
                "params": {},
            }
        return {
            "action_type": "wait",
            "target": {"scene_type": scene_type},
            "reason": f"当前处于 {scene_type} 场景，但暂未检测到可稳定执行的控件，先等待。",
            "confidence": 0.18,
            "source": source,
            "timestamp": now_str(),
            "params": {"duration": 0.4},
        }

    def _collect_choice_options(self, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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

    def _pick_best_choice(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        def key_fn(item: Dict[str, Any]) -> float:
            name = str(item.get("name", "") or "").lower()
            confidence = self._to_float(item.get("confidence"), default=0.5)
            penalty = 0.0
            if any(word in name for word in ["skip", "leave", "ignore", "cancel"]):
                penalty -= 0.15
            return confidence + penalty

        return sorted(options, key=key_fn, reverse=True)[0]

    def _pick_enemy(self, enemies: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not enemies:
            return {}
        return enemies[0]

    def _find_card(
        self,
        hand_cards: List[Dict[str, Any]],
        card_type: Optional[str] = None,
        card_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        for card in hand_cards:
            if card_type is not None and str(card.get("type", "")).lower() != str(card_type).lower():
                continue
            if card_name is not None and str(card.get("name", "")).lower() != str(card_name).lower():
                continue
            return card
        return None

    def _find_named_option(self, options: List[Dict[str, Any]], names: set[str]) -> Optional[Dict[str, Any]]:
        lowered = {str(x).lower() for x in names}
        for option in options:
            name = str(option.get("name", "") or "").lower()
            if name in lowered:
                return option
        return None

    def _score_map_option(self, option: Dict[str, Any], hp_ratio: Optional[float]) -> float:
        kind = str(option.get("kind", "unknown") or "unknown").lower()
        score_map = {
            "event_unknown": 0.72,
            "merchant": 0.66,
            "rest": 0.58,
            "battle": 0.62,
            "elite": 0.42,
            "boss": 0.35,
            "unknown": 0.40,
        }
        score = score_map.get(kind, 0.40)
        if hp_ratio is not None:
            if hp_ratio < 0.45:
                if kind == "rest":
                    score += 0.30
                if kind in {"elite", "boss"}:
                    score -= 0.25
            elif hp_ratio > 0.75:
                if kind == "event_unknown":
                    score += 0.08
        score += self._to_float(option.get("confidence"), default=0.0) * 0.10
        return round(score, 4)

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

    def _normalize_scene(self, scene_type: Any) -> str:
        raw = str(scene_type or "unknown")
        return SCENE_ALIASES.get(raw, raw)

    def _resolve_episode_id(self, state_data: Dict[str, Any]) -> Optional[str]:
        return (
            state_data.get("episode_id")
            or self.runtime_context.get("current_episode_id")
            or self.runtime_context.get("episode_id")
        )

    def _safe_ratio(self, hp: Any, max_hp: Any) -> Optional[float]:
        hp_i = self._to_int(hp)
        max_i = self._to_int(max_hp)
        if hp_i is None or max_i is None or max_i <= 0:
            return None
        return round(hp_i / max_i, 4)

    def _to_int(self, value: Any, default: Optional[int] = None) -> Optional[int]:
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(action)
        result.setdefault("action_type", "wait")
        result.setdefault("target", {})
        result.setdefault("reason", "未提供动作解释")
        result.setdefault("confidence", 0.0)
        result.setdefault("source", "Brain")
        result.setdefault("timestamp", now_str())
        result.setdefault("params", {})
        return result

    def _try_llm_assist(
        self,
        task: str,
        state_data: Dict[str, Any],
        memory_summary: Dict[str, Any],
        step_id: int,
        current_action: Dict[str, Any],
    ) -> str:
        if not self.llm_enabled or self.llm is None:
            return ""
        try:
            prompt = self._build_llm_prompt(task, state_data, memory_summary, step_id, current_action)
            if hasattr(self.llm, "generate") and callable(getattr(self.llm, "generate")):
                response = self.llm.generate(prompt)  # type: ignore[attr-defined]
            elif hasattr(self.llm, "chat") and callable(getattr(self.llm, "chat")):
                response = self.llm.chat(  # type: ignore[attr-defined]
                    [{"role": "user", "content": prompt}],
                    temperature=float(self.model_config.get("temperature", 0.2)),
                    max_tokens=int(self.model_config.get("max_tokens", 512)),
                )
            else:
                raise AttributeError(f"{self.llm.__class__.__name__} 未提供 generate/chat 接口")
            return str(response or "").strip()
        except Exception as exc:
            logging.warning("LLM 辅助分析失败，忽略并继续使用规则动作: %s", exc)
            return ""

    def _build_llm_prompt(
        self,
        task: str,
        state_data: Dict[str, Any],
        memory_summary: Dict[str, Any],
        step_id: int,
        current_action: Dict[str, Any],
    ) -> str:
        return (
            "你是一个帮助解释 Slay the Spire 2 当前动作选择的辅助分析器。\n"
            f"任务: {task}\n"
            f"step_id: {step_id}\n"
            f"scene_type: {state_data.get('scene_type')}\n"
            f"scene_variant: {state_data.get('scene_variant')}\n"
            f"当前动作: {current_action}\n"
            f"memory_summary: {memory_summary}\n"
            "请用简短中文说明当前动作的合理性，不要修改动作本身。"
        )
