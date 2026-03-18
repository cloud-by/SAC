
"""
BottomUpAgent/group_b/StateEncoder.py

轻量状态编码器：
1. 将 StateAdapter 输出的 state_repr / features 转成更适合训练的扁平特征
2. 当前优先支持：
   - one-hot / multi-hot 风格编码
   - 连续值归一化
3. 不依赖第三方 ML 框架，便于课程项目快速联调
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class StateEncoder:
    SCENE_TYPES = [
        "title_main",
        "mode_select",
        "character_select",
        "blessing_choice",
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
        "unknown",
    ]

    HP_BUCKETS = ["empty", "low", "mid", "high"]
    ENERGY_BUCKETS = ["empty", "low", "mid", "high"]

    def encode(self, feature_dict: Dict[str, Any]) -> Dict[str, float]:
        encoded: Dict[str, float] = {}

        self._encode_one_hot(encoded, "scene", str(feature_dict.get("scene_type", "unknown")), self.SCENE_TYPES)
        self._encode_optional_text(encoded, "scene_variant", feature_dict.get("scene_variant"))
        self._encode_optional_text(encoded, "phase", feature_dict.get("phase"))

        self._encode_one_hot(encoded, "hp_bucket", str(feature_dict.get("hp_bucket", "unknown")), self.HP_BUCKETS)
        self._encode_one_hot(encoded, "energy_bucket", str(feature_dict.get("energy_bucket", "empty")), self.ENERGY_BUCKETS)

        encoded["floor"] = self._safe_float(feature_dict.get("floor"))
        encoded["hp_ratio"] = self._safe_float(feature_dict.get("hp_ratio"))
        encoded["energy"] = self._safe_float(feature_dict.get("energy"))
        encoded["hand_count"] = self._safe_float(feature_dict.get("hand_count"))
        encoded["enemy_count"] = self._safe_float(feature_dict.get("enemy_count"))
        encoded["map_option_count"] = self._safe_float(feature_dict.get("map_option_count"))
        encoded["reward_option_count"] = self._safe_float(feature_dict.get("reward_option_count"))
        encoded["shop_item_count"] = self._safe_float(feature_dict.get("shop_item_count"))
        encoded["campfire_option_count"] = self._safe_float(feature_dict.get("campfire_option_count"))
        encoded["attack_card_count"] = self._safe_float(feature_dict.get("attack_card_count"))
        encoded["skill_card_count"] = self._safe_float(feature_dict.get("skill_card_count"))
        encoded["power_card_count"] = self._safe_float(feature_dict.get("power_card_count"))
        encoded["playable_card_count"] = self._safe_float(feature_dict.get("playable_card_count"))
        encoded["enemy_attack_count"] = self._safe_float(feature_dict.get("enemy_attack_count"))
        encoded["enemy_unknown_count"] = self._safe_float(feature_dict.get("enemy_unknown_count"))
        encoded["match_confidence"] = self._safe_float(feature_dict.get("match_confidence"))

        bool_keys = [
            "has_attack_card",
            "has_skill_card",
            "map_has_battle",
            "map_has_event",
            "map_has_rest",
            "map_has_shop",
            "map_has_elite",
            "reward_has_card",
            "flow_corrected",
            "has_continue",
            "has_back",
            "has_end_turn",
            "is_low_hp",
        ]
        for key in bool_keys:
            encoded[key] = 1.0 if bool(feature_dict.get(key, False)) else 0.0

        return encoded

    def to_vector(self, encoded: Dict[str, float]) -> Tuple[List[str], List[float]]:
        keys = sorted(encoded.keys())
        values = [float(encoded[k]) for k in keys]
        return keys, values

    def encode_to_vector(self, feature_dict: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        return self.to_vector(self.encode(feature_dict))

    def _encode_one_hot(self, out: Dict[str, float], prefix: str, value: str, candidates: List[str]) -> None:
        normalized = value if value in candidates else candidates[-1]
        for item in candidates:
            out[f"{prefix}__{item}"] = 1.0 if item == normalized else 0.0

    def _encode_optional_text(self, out: Dict[str, float], prefix: str, value: Any) -> None:
        text = str(value).strip().lower() if value is not None else ""
        if not text:
            out[f"{prefix}__none"] = 1.0
            return
        out[f"{prefix}__none"] = 0.0
        out[f"{prefix}__{text}"] = 1.0

    def _safe_float(self, value: Any) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except Exception:
            return 0.0
