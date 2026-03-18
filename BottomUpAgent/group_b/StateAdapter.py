"""
BottomUpAgent/group_b/StateAdapter.py

小修版：主要修复 episode_id 在真实流程中的统一问题。
其余逻辑与上一版保持兼容。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class StateAdapter:
    PRE_RUN_SCENES = {"title_main", "mode_select", "character_select", "blessing_choice"}
    IN_RUN_SCENES = {
        "map",
        "battle",
        "battle_result",
        "card_reward",
        "merchant",
        "merchant_shop",
        "campfire_choice",
        "campfire_upgrade",
        "campfire_rest_done",
        "event_unknown",
    }
    TERMINAL_SCENES = {"victory", "death", "game_over", "unknown_terminal"}

    SCENE_ALIASES = {
        "reward": "blessing_choice",
        "act_intro": "blessing_choice",
        "single_mode_menu": "mode_select",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.runtime_context = config.setdefault("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()

        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        adapter_dir = raw_paths.get("adapted_states", "data/adapted_states")
        self.adapter_dir = Path(adapter_dir)
        if not self.adapter_dir.is_absolute():
            self.adapter_dir = self.project_root / self.adapter_dir
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

        self.persist_enabled = bool(config.get("runtime", {}).get("persist_adapted_state", False))
        logging.info("StateAdapter 初始化完成，persist=%s", self.persist_enabled)

    def adapt(
        self,
        state_data: Dict[str, Any],
        *,
        step_id: Optional[int] = None,
        episode_id: Optional[str] = None,
        memory_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_episode_id = episode_id or state_data.get("episode_id") or self.runtime_context.get("current_episode_id")

        scene_type = self._normalize_scene(str(state_data.get("scene_type", "unknown") or "unknown"))
        scene_variant = self._normalize_variant(state_data.get("scene_variant"))

        hp = self._to_int(state_data.get("hp"))
        max_hp = self._to_int(state_data.get("max_hp"))
        energy = self._to_int(state_data.get("energy"), default=0)
        floor = self._to_int(state_data.get("floor"), default=self._to_int(state_data.get("floor_hint")))

        hand_cards = self._safe_list(state_data.get("hand_cards") or state_data.get("detected_cards"))
        enemies = self._safe_list(state_data.get("enemies") or state_data.get("detected_enemies"))
        map_options = self._safe_list(state_data.get("map_options"))
        reward_options = self._safe_list(state_data.get("reward_options"))
        shop_items = self._safe_list(state_data.get("shop_items"))
        campfire_options = self._safe_list(state_data.get("campfire_options"))
        upgrade_options = self._safe_list(state_data.get("upgrade_options"))
        menu_options = self._safe_list(state_data.get("menu_options"))
        mode_options = self._safe_list(state_data.get("mode_options"))
        character_options = self._safe_list(state_data.get("character_options"))
        detected_buttons = self._safe_list(state_data.get("detected_buttons"))
        available_buttons = self._safe_list(state_data.get("available_buttons"))

        hp_ratio = self._safe_ratio(hp, max_hp)
        phase = self._infer_phase(scene_type, state_data)
        flow_corrected = bool(state_data.get("flow_corrected", False))
        match_confidence = self._to_float(state_data.get("match_confidence"), default=0.0)

        hand_summary = self._summarize_hand(hand_cards)
        enemy_summary = self._summarize_enemies(enemies)
        map_summary = self._summarize_map(map_options)
        reward_summary = self._summarize_reward(reward_options)
        button_summary = self._summarize_buttons(detected_buttons, available_buttons)

        state_repr: Dict[str, Any] = {
            "episode_id": resolved_episode_id,
            "step_id": step_id,
            "timestamp": now_str(),
            "scene_type": scene_type,
            "scene_variant": scene_variant,
            "phase": phase,
            "scene_key": self._build_scene_key(scene_type, scene_variant),
            "floor": floor,
            "hp": hp,
            "max_hp": max_hp,
            "hp_ratio": hp_ratio,
            "hp_bucket": self._bucket_hp(hp_ratio),
            "energy": energy,
            "energy_bucket": self._bucket_energy(energy),
            "selected_character": self._normalize_variant(state_data.get("selected_character")),
            "match_confidence": round(match_confidence, 4),
            "matched_template": state_data.get("matched_template"),
            "flow_corrected": flow_corrected,
            "flow_reason": str(state_data.get("flow_reason", "") or ""),
            "flow_prev_scene": self._normalize_scene(str(state_data.get("flow_prev_scene", "unknown") or "unknown")),
            "flow_phase_before": str(state_data.get("flow_phase_before", "") or ""),
            "flow_phase_after": str(state_data.get("flow_phase_after", "") or ""),
            "hand_count": len(hand_cards),
            "enemy_count": len(enemies),
            "map_option_count": len(map_options),
            "reward_option_count": len(reward_options),
            "shop_item_count": len(shop_items),
            "campfire_option_count": len(campfire_options),
            "upgrade_option_count": len(upgrade_options),
            "menu_option_count": len(menu_options),
            "mode_option_count": len(mode_options),
            "character_option_count": len(character_options),
            "hand_summary": hand_summary,
            "enemy_summary": enemy_summary,
            "map_summary": map_summary,
            "reward_summary": reward_summary,
            "button_summary": button_summary,
            "flags": {
                "has_continue": button_summary["has_continue"],
                "has_back": button_summary["has_back"],
                "has_end_turn": button_summary["has_end_turn"],
                "has_map_node": len(map_options) > 0,
                "has_reward_choice": len(reward_options) > 0,
                "has_shop_item": len(shop_items) > 0,
                "has_campfire_choice": len(campfire_options) > 0,
                "is_low_hp": hp_ratio is not None and hp_ratio < 0.40,
                "is_mid_hp": hp_ratio is not None and 0.40 <= hp_ratio < 0.70,
                "is_high_hp": hp_ratio is not None and hp_ratio >= 0.70,
            },
            "state_signature": self._build_state_signature(
                scene_type=scene_type,
                scene_variant=scene_variant,
                hp_bucket=self._bucket_hp(hp_ratio),
                energy_bucket=self._bucket_energy(energy),
                hand_summary=hand_summary,
                enemy_summary=enemy_summary,
                map_summary=map_summary,
                reward_summary=reward_summary,
            ),
        }

        if memory_summary:
            state_repr["memory_summary"] = {
                "skill_count": self._to_int(memory_summary.get("skill_count"), default=0),
                "total_records": self._to_int(memory_summary.get("total_records"), default=0),
                "recent_skills_count": len(self._safe_list(memory_summary.get("recent_skills"))),
            }

        state_repr["raw_refs"] = {
            "screen_image": state_data.get("screen_image"),
            "window_bbox": state_data.get("window_bbox"),
            "screen_image_name": Path(str(state_data.get("screen_image", ""))).name if state_data.get("screen_image") else None,
        }

        if self.persist_enabled:
            self._persist(state_repr)
        return state_repr

    def build_feature_dict(self, state_repr: Dict[str, Any]) -> Dict[str, Any]:
        hand_summary = dict(state_repr.get("hand_summary", {}) or {})
        enemy_summary = dict(state_repr.get("enemy_summary", {}) or {})
        map_summary = dict(state_repr.get("map_summary", {}) or {})
        flags = dict(state_repr.get("flags", {}) or {})
        return {
            "scene_type": state_repr.get("scene_type", "unknown"),
            "scene_variant": state_repr.get("scene_variant"),
            "phase": state_repr.get("phase", "unknown"),
            "floor": state_repr.get("floor"),
            "hp_ratio": state_repr.get("hp_ratio"),
            "hp_bucket": state_repr.get("hp_bucket"),
            "energy": state_repr.get("energy", 0),
            "energy_bucket": state_repr.get("energy_bucket"),
            "hand_count": state_repr.get("hand_count", 0),
            "enemy_count": state_repr.get("enemy_count", 0),
            "map_option_count": state_repr.get("map_option_count", 0),
            "reward_option_count": state_repr.get("reward_option_count", 0),
            "shop_item_count": state_repr.get("shop_item_count", 0),
            "campfire_option_count": state_repr.get("campfire_option_count", 0),
            "attack_card_count": hand_summary.get("attack_count", 0),
            "skill_card_count": hand_summary.get("skill_count", 0),
            "power_card_count": hand_summary.get("power_count", 0),
            "playable_card_count": hand_summary.get("playable_count", 0),
            "has_attack_card": hand_summary.get("attack_count", 0) > 0,
            "has_skill_card": hand_summary.get("skill_count", 0) > 0,
            "enemy_attack_count": enemy_summary.get("attack_intent_count", 0),
            "enemy_unknown_count": enemy_summary.get("unknown_intent_count", 0),
            "map_has_battle": map_summary.get("has_battle", False),
            "map_has_event": map_summary.get("has_event", False),
            "map_has_rest": map_summary.get("has_rest", False),
            "map_has_shop": map_summary.get("has_shop", False),
            "map_has_elite": map_summary.get("has_elite", False),
            "reward_has_card": state_repr.get("reward_option_count", 0) > 0,
            "match_confidence": state_repr.get("match_confidence", 0.0),
            "flow_corrected": state_repr.get("flow_corrected", False),
            "has_continue": flags.get("has_continue", False),
            "has_back": flags.get("has_back", False),
            "has_end_turn": flags.get("has_end_turn", False),
            "is_low_hp": flags.get("is_low_hp", False),
            "episode_id": state_repr.get("episode_id"),
        }

    def adapt_and_encode(self, state_data: Dict[str, Any], *, step_id: Optional[int] = None, episode_id: Optional[str] = None, memory_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state_repr = self.adapt(state_data, step_id=step_id, episode_id=episode_id, memory_summary=memory_summary)
        features = self.build_feature_dict(state_repr)
        return {"state_repr": state_repr, "features": features}

    def _infer_phase(self, scene_type: str, state_data: Dict[str, Any]) -> str:
        if scene_type in self.PRE_RUN_SCENES:
            return "pre_run"
        if scene_type in self.IN_RUN_SCENES:
            return "in_run"
        if scene_type in self.TERMINAL_SCENES:
            return "terminal"
        flow_after = str(state_data.get("flow_phase_after", "") or "")
        if flow_after:
            return flow_after
        return "unknown"

    def _normalize_scene(self, scene_type: str) -> str:
        return self.SCENE_ALIASES.get(scene_type, scene_type)

    def _normalize_variant(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _safe_list(self, value: Any) -> List[Dict[str, Any]]:
        return [x for x in value] if isinstance(value, list) else []

    def _to_int(self, value: Any, default: Optional[int] = None) -> Optional[int]:
        try:
            if value is None:
                return default
            return int(value)
        except Exception:
            return default

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _safe_ratio(self, num: Optional[int], den: Optional[int]) -> Optional[float]:
        if num is None or den is None or den <= 0:
            return None
        return round(num / den, 4)

    def _bucket_hp(self, hp_ratio: Optional[float]) -> str:
        if hp_ratio is None:
            return "empty"
        if hp_ratio < 0.40:
            return "low"
        if hp_ratio < 0.70:
            return "mid"
        return "high"

    def _bucket_energy(self, energy: Optional[int]) -> str:
        if energy is None:
            return "empty"
        if energy <= 0:
            return "low"
        if energy <= 1:
            return "mid"
        return "high"

    def _build_scene_key(self, scene_type: str, scene_variant: Optional[str]) -> str:
        return f"{scene_type}:{scene_variant}" if scene_variant else scene_type

    def _summarize_hand(self, hand_cards: List[Dict[str, Any]]) -> Dict[str, Any]:
        attack_count = sum(1 for c in hand_cards if str(c.get("type", "")).lower() == "attack")
        skill_count = sum(1 for c in hand_cards if str(c.get("type", "")).lower() == "skill")
        power_count = sum(1 for c in hand_cards if str(c.get("type", "")).lower() == "power")
        playable_count = sum(1 for c in hand_cards if self._to_int(c.get("cost"), 99) <= 3)
        sample_names = [str(c.get("name", "unknown")) for c in hand_cards[:5]]
        return {
            "count": len(hand_cards),
            "attack_count": attack_count,
            "skill_count": skill_count,
            "power_count": power_count,
            "playable_count": playable_count,
            "sample_names": sample_names,
        }

    def _summarize_enemies(self, enemies: List[Dict[str, Any]]) -> Dict[str, Any]:
        attack_intent_count = 0
        unknown_intent_count = 0
        for enemy in enemies:
            intent = str(enemy.get("intent", "unknown") or "unknown").lower()
            if "attack" in intent:
                attack_intent_count += 1
            if intent == "unknown":
                unknown_intent_count += 1
        return {
            "count": len(enemies),
            "attack_intent_count": attack_intent_count,
            "unknown_intent_count": unknown_intent_count,
            "sample_names": [str(e.get("name", "unknown")) for e in enemies[:5]],
        }

    def _summarize_map(self, map_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        kind_counts = {"battle": 0, "event_unknown": 0, "rest": 0, "merchant": 0, "elite": 0, "boss": 0, "unknown": 0}
        for option in map_options:
            kind = str(option.get("kind", "unknown") or "unknown").lower()
            if kind not in kind_counts:
                kind = "unknown"
            kind_counts[kind] += 1
        return {
            "count": len(map_options),
            "kind_counts": kind_counts,
            "has_battle": kind_counts["battle"] > 0,
            "has_event": kind_counts["event_unknown"] > 0,
            "has_rest": kind_counts["rest"] > 0,
            "has_shop": kind_counts["merchant"] > 0,
            "has_elite": kind_counts["elite"] > 0,
            "has_boss": kind_counts["boss"] > 0,
        }

    def _summarize_reward(self, reward_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"count": len(reward_options), "sample_names": [str(x.get("name", "unknown")) for x in reward_options[:5]]}

    def _summarize_buttons(self, detected_buttons: List[Dict[str, Any]], available_buttons: List[Any]) -> Dict[str, Any]:
        names = {str(btn.get("name", "")).lower() for btn in detected_buttons if isinstance(btn, dict)}
        names.update(str(x).lower() for x in available_buttons)
        return {
            "names": sorted(x for x in names if x),
            "has_continue": "continue" in names,
            "has_back": "back" in names,
            "has_end_turn": "end_turn" in names,
        }

    def _build_state_signature(self, *, scene_type: str, scene_variant: Optional[str], hp_bucket: str, energy_bucket: str, hand_summary: Dict[str, Any], enemy_summary: Dict[str, Any], map_summary: Dict[str, Any], reward_summary: Dict[str, Any]) -> str:
        parts = [
            scene_type,
            scene_variant or "none",
            hp_bucket,
            energy_bucket,
            f"h{hand_summary.get('count', 0)}",
            f"a{hand_summary.get('attack_count', 0)}",
            f"s{hand_summary.get('skill_count', 0)}",
            f"e{enemy_summary.get('count', 0)}",
            f"ea{enemy_summary.get('attack_intent_count', 0)}",
            f"m{map_summary.get('count', 0)}",
            f"r{reward_summary.get('count', 0)}",
        ]
        return "|".join(str(x) for x in parts)

    def _persist(self, state_repr: Dict[str, Any]) -> None:
        episode_id = state_repr.get("episode_id") or "manual_episode"
        step_id = state_repr.get("step_id") or 0
        path = self.adapter_dir / f"{episode_id}_adapted_{int(step_id):04d}.json"
        path.write_text(json.dumps(state_repr, ensure_ascii=False, indent=2), encoding="utf-8")
