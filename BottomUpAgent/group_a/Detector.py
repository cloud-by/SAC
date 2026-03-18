from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

try:
    from BottomUpAgent.group_a.SceneMatcher import SceneMatcher
except Exception:  # pragma: no cover
    try:
        from SceneMatcher import SceneMatcher  # type: ignore
    except Exception:  # pragma: no cover
        SceneMatcher = None  # type: ignore

try:
    from BottomUpAgent.group_a.SceneFlowGuard import SceneFlowGuard
except Exception:  # pragma: no cover
    try:
        from SceneFlowGuard import SceneFlowGuard  # type: ignore
    except Exception:  # pragma: no cover
        SceneFlowGuard = None  # type: ignore


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Detector:
    CARD_TEMPLATES = [
        {"name": "Strike", "cost": 1, "type": "attack"},
        {"name": "Defend", "cost": 1, "type": "skill"},
        {"name": "Strike", "cost": 1, "type": "attack"},
        {"name": "Defend", "cost": 1, "type": "skill"},
        {"name": "Bash", "cost": 2, "type": "attack"},
    ]

    SUPPORTED_FORCE_SCENES = {
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
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.environment = config.get("environment", {})
        self.runtime_context = config.setdefault("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()

        self.debug_dir = self.project_root / "data" / "states"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.capture_bbox = self._normalize_bbox(self.environment.get("capture_bbox"))
        force_scene = self.environment.get("force_scene")
        self.force_scene = force_scene if force_scene in self.SUPPORTED_FORCE_SCENES else None
        self.runtime_capture_bbox: Optional[List[int]] = None

        self.scene_match_cfg = dict(config.get("scene_match", {}) or {})
        self.scene_match_enabled = bool(self.scene_match_cfg.get("enabled", True))
        self.scene_matcher = self._build_scene_matcher()
        self.scene_flow_guard = self._build_scene_flow_guard()

        logging.info(
            "Detector 初始化完成，capture_bbox=%s, force_scene=%s, scene_match_enabled=%s",
            self.capture_bbox,
            self.force_scene,
            self.scene_match_enabled,
        )

    def _build_scene_flow_guard(self):
        if SceneFlowGuard is None:
            logging.warning("SceneFlowGuard 导入失败，将不启用流程修正。")
            return None
        try:
            return SceneFlowGuard(self.runtime_context)
        except Exception as exc:
            logging.warning("SceneFlowGuard 初始化失败，将不启用流程修正: %s", exc)
            return None

    def detect(self, screen_image: Union[str, Image.Image]) -> Dict[str, Any]:
        image = self._load_image(screen_image)
        width, height = image.size

        runtime_bbox = self._normalize_bbox(getattr(self, "runtime_capture_bbox", None))
        offset_x = runtime_bbox[0] if runtime_bbox else (self.capture_bbox[0] if self.capture_bbox else 0)
        offset_y = runtime_bbox[1] if runtime_bbox else (self.capture_bbox[1] if self.capture_bbox else 0)

        rois = self._build_base_regions(width=width, height=height, offset_x=offset_x, offset_y=offset_y)

        match_result = self._match_scene(image=image)
        scene_hint = match_result.get("scene_hint", "unknown")
        scene_scores = match_result.get("scene_scores", {})
        scene_variant = match_result.get("scene_variant")
        matched_template = match_result.get("matched_template")
        match_confidence = float(match_result.get("match_confidence", 0.0))

        legacy_scores = self._legacy_scene_scores(image, rois)
        if scene_hint == "unknown":
            legacy_scene = self._classify_legacy_scene(legacy_scores)
            if legacy_scene != "unknown":
                scene_hint = legacy_scene
                if legacy_scene not in scene_scores:
                    scene_scores = dict(scene_scores)
                    scene_scores[legacy_scene] = float(legacy_scores.get(legacy_scene, 0.0))

        flow_result = {
            "scene_hint": scene_hint,
            "scene_variant": scene_variant,
            "flow_phase_before": None,
            "flow_phase_after": None,
            "flow_prev_scene": None,
            "flow_allowed_scenes": [],
            "flow_corrected": False,
            "flow_reason": "flow_guard_disabled",
            "flow_ordered_candidates": [],
        }
        if self.scene_flow_guard is not None:
            flow_result = self.scene_flow_guard.resolve(
                raw_scene=scene_hint,
                scene_scores=scene_scores,
                match_confidence=match_confidence,
                legacy_scores=legacy_scores,
                scene_variant=scene_variant,
            )
            scene_hint = flow_result.get("scene_hint", scene_hint)
            scene_variant = flow_result.get("scene_variant", scene_variant)

        payload = self._detect_scene_specific_payload(image, rois, scene_hint)
        player = self._estimate_player_status(scene_hint=scene_hint, image=image, rois=rois, payload=payload)

        result: Dict[str, Any] = {
            "scene_hint": scene_hint,
            "scene_variant": scene_variant,
            "scene_scores": scene_scores,
            "match_confidence": match_confidence,
            "matched_template": matched_template,
            "scene_match_debug": match_result.get("debug", {}),
            "legacy_scene_scores": legacy_scores,
            "flow_phase_before": flow_result.get("flow_phase_before"),
            "flow_phase_after": flow_result.get("flow_phase_after"),
            "flow_prev_scene": flow_result.get("flow_prev_scene"),
            "flow_allowed_scenes": flow_result.get("flow_allowed_scenes", []),
            "flow_corrected": flow_result.get("flow_corrected", False),
            "flow_reason": flow_result.get("flow_reason", ""),
            "flow_ordered_candidates": flow_result.get("flow_ordered_candidates", []),
            "window_bbox": runtime_bbox or self.capture_bbox or [0, 0, width, height],
            "capture_bbox": runtime_bbox or self.capture_bbox,
            "timestamp": now_str(),
            "title_logo_area": rois["title_logo_area"],
            "title_menu_column": rois["title_menu_column"],
            "title_single_mode": rois["title_single_mode"],
            "single_standard_card": rois["single_standard_card"],
            "single_daily_card": rois["single_daily_card"],
            "single_custom_card": rois["single_custom_card"],
            "back_button": rois["back_button"],
            "character_confirm_button": rois["character_confirm_button"],
            "continue_button": rois["continue_button"],
            "right_arrow_button": rois["right_arrow_button"],
            "center_modal_button": rois["center_modal_button"],
            "hand_area": rois["hand_area"],
            "enemy_area": rois["enemy_area"],
            "player_hp_bar": rois["player_hp_bar"],
            "energy_area": rois["energy_area"],
            "map_area": rois["map_area"],
            "reward_area": rois["reward_area"],
            "end_turn_button": rois["end_turn_button"],
            "detected_cards": payload.get("detected_cards", []),
            "detected_enemies": payload.get("detected_enemies", []),
            "map_options": payload.get("map_options", []),
            "reward_options": payload.get("reward_options", []),
            "player": player,
            "floor_hint": payload.get("floor_hint"),
        }

        result.update(payload)
        self._save_debug_result(screen_image=screen_image, result=result)
        return result

    def _build_scene_matcher(self):
        if not self.scene_match_enabled:
            return None
        if SceneMatcher is None:
            logging.warning("SceneMatcher 导入失败，将只使用旧启发式分类。")
            return None
        try:
            return SceneMatcher(self.config)
        except Exception as exc:
            logging.warning("SceneMatcher 初始化失败，将只使用旧启发式分类: %s", exc)
            return None

    def _match_scene(self, image: Image.Image) -> Dict[str, Any]:
        if isinstance(self.force_scene, str) and self.force_scene in self.SUPPORTED_FORCE_SCENES:
            return {
                "scene_hint": str(self.force_scene),
                "scene_variant": None,
                "scene_scores": {str(self.force_scene): 1.0},
                "match_confidence": 1.0,
                "matched_template": None,
                "debug": {"forced": True},
            }

        if self.scene_matcher is None:
            return {
                "scene_hint": "unknown",
                "scene_variant": None,
                "scene_scores": {},
                "match_confidence": 0.0,
                "matched_template": None,
                "debug": {"reason": "no_scene_matcher"},
            }

        return self.scene_matcher.match_scene(image)

    def _classify_legacy_scene(self, scores: Dict[str, float]) -> str:
        if not scores:
            return "unknown"
        best_scene = max(scores, key=scores.get)
        best_score = scores[best_scene]
        return best_scene if best_score >= 2.0 else "unknown"

    def _legacy_scene_scores(self, image: Image.Image, rois: Dict[str, List[int]]) -> Dict[str, float]:
        scores = {
            "title_main": 0.0,
            "mode_select": 0.0,
            "character_select": 0.0,
            "blessing_choice": 0.0,
            "battle": 0.0,
            "map": 0.0,
        }

        logo_gold = self._gold_ratio(image, rois["title_logo_area"])
        title_menu_contrast = self._contrast(image, rois["title_menu_column"])
        back_red = self._red_ratio(image, rois["back_button"])
        screen_red = self._red_ratio(image, rois["character_main_area"])

        left_b = self._brightness(image, rois["single_standard_card"])
        mid_b = self._brightness(image, rois["single_daily_card"])
        right_b = self._brightness(image, rois["single_custom_card"])
        left_c = self._contrast(image, rois["single_standard_card"])
        mid_c = self._contrast(image, rois["single_daily_card"])
        right_c = self._contrast(image, rois["single_custom_card"])
        mid_blue = self._blue_ratio(image, rois["single_daily_card"])
        right_warm = self._warm_ratio(image, rois["single_custom_card"])

        info_panel_gold = self._gold_ratio(image, rois["character_info_panel"])
        iron_b = self._brightness(image, rois["character_ironclad_card"])
        iron_warm = self._warm_ratio(image, rois["character_ironclad_card"])
        second_b = self._brightness(image, rois["character_second_card"])
        confirm_blue = self._blue_ratio(image, rois["character_confirm_button"])

        map_b = self._brightness(image, rois["map_area"])
        map_gold = self._gold_ratio(image, rois["map_area"])
        legend_blue = self._blue_ratio(image, rois["map_legend"])
        option_blue = self._blue_ratio(image, rois["reward_option_panel"])
        option_contrast = self._contrast(image, rois["reward_option_panel"])
        textbar_blue = self._blue_ratio(image, rois["reward_text_bar"])
        reward_brightness = self._brightness(image, rois["reward_area"])
        end_turn_warm = self._warm_ratio(image, rois["end_turn_button"])
        end_turn_gold = self._gold_ratio(image, rois["end_turn_button"])
        hand_contrast = self._contrast(image, rois["hand_area"])
        enemy_contrast = self._contrast(image, rois["enemy_area"])

        if logo_gold > 0.08:
            scores["title_main"] += 2.0
        if logo_gold > 0.12:
            scores["title_main"] += 1.0
        if title_menu_contrast > 24:
            scores["title_main"] += 1.0
        if back_red < 0.10:
            scores["title_main"] += 1.0
        if screen_red < 0.18:
            scores["title_main"] += 1.0

        avg_card_b = (left_b + mid_b + right_b) / 3.0
        avg_card_c = (left_c + mid_c + right_c) / 3.0
        if avg_card_c > 26:
            scores["mode_select"] += 2.0
        if avg_card_b > 70:
            scores["mode_select"] += 1.0
        if mid_blue > 0.18:
            scores["mode_select"] += 1.5
        if right_warm > 0.18:
            scores["mode_select"] += 1.5
        if back_red > 0.10:
            scores["mode_select"] += 1.0

        if screen_red > 0.18:
            scores["character_select"] += 2.0
        if info_panel_gold > 0.010:
            scores["character_select"] += 1.0
        if iron_warm > 0.16:
            scores["character_select"] += 1.5
        if iron_b > second_b + 6:
            scores["character_select"] += 1.0
        if confirm_blue > 0.06:
            scores["character_select"] += 0.5

        if map_gold > 0.45:
            scores["map"] += 2.0
        if map_b > 105:
            scores["map"] += 1.0
        if legend_blue > 0.20:
            scores["map"] += 1.5
        if back_red > 0.10:
            scores["map"] += 1.0

        if option_blue > 0.20:
            scores["blessing_choice"] += 2.0
        if option_contrast > 26:
            scores["blessing_choice"] += 1.0
        if textbar_blue > 0.15:
            scores["blessing_choice"] += 1.5
        if 45 < reward_brightness < 105:
            scores["blessing_choice"] += 0.5

        if end_turn_warm > 0.10 or end_turn_gold > 0.10:
            scores["battle"] += 2.0
        if hand_contrast > 18:
            scores["battle"] += 1.0
        if enemy_contrast > 12:
            scores["battle"] += 1.0

        if map_gold > 0.45:
            scores["battle"] -= 3.0
        if option_blue > 0.20:
            scores["battle"] -= 3.0

        return scores

    def _detect_scene_specific_payload(self, image: Image.Image, rois: Dict[str, List[int]], scene_hint: str) -> Dict[
        str, Any]:
        title_menu_options = self._detect_menu_options(rois) if scene_hint == "title_main" else []
        payload: Dict[str, Any] = {
            "detected_buttons": self._detect_buttons(rois, scene_hint, title_menu_options=title_menu_options),
            "menu_options": [],
            "mode_options": [],
            "character_options": [],
            "selected_character": None,
            "detected_cards": [],
            "detected_enemies": [],
            "map_options": [],
            "reward_options": [],
        }

        if scene_hint == "title_main":
            payload["menu_options"] = title_menu_options
            if title_menu_options:
                payload["single_mode_bbox"] = title_menu_options[0].get("bbox", rois["title_single_mode"])
            else:
                payload["single_mode_bbox"] = rois["title_single_mode"]

        elif scene_hint == "mode_select":
            payload["mode_options"] = self._detect_mode_options(rois)
            payload["standard_mode_bbox"] = rois["single_standard_card"]
            payload["back_button_bbox"] = rois["back_button"]

        elif scene_hint == "character_select":
            payload["character_options"] = self._detect_character_options(rois)
            payload["selected_character"] = self._detect_selected_character(image, rois)
            payload["confirm_button_bbox"] = rois["character_confirm_button"]
            payload["back_button_bbox"] = rois["back_button"]

        elif scene_hint == "blessing_choice":
            payload["reward_options"] = self._detect_reward_options(rois["reward_area"])
            payload["back_button_bbox"] = rois["back_button"]

        elif scene_hint == "map":
            payload["map_options"] = self._detect_map_options(rois["map_area"])
            payload["legend_visible"] = True
            payload["back_button_bbox"] = rois["back_button"]
            payload["floor_hint"] = 1

        elif scene_hint == "battle":
            payload["detected_cards"] = self._detect_cards(image, rois["hand_area"])
            payload["detected_enemies"] = self._detect_enemies(image, rois["enemy_area"])
            payload["floor_hint"] = 1

        elif scene_hint == "battle_result":
            payload["continue_bbox"] = rois["center_modal_button"]

        elif scene_hint == "card_reward":
            payload["reward_options"] = self._detect_card_reward_options(rois["card_reward_panel"])
            payload["continue_bbox"] = rois["center_modal_button"]
            payload["skip_button_bbox"] = rois["right_arrow_button"]

        elif scene_hint == "merchant":
            payload["merchant_enter_shop_bbox"] = rois["center_modal_button"]
            payload["continue_bbox"] = rois["right_arrow_button"]

        elif scene_hint == "merchant_shop":
            payload["shop_items"] = self._detect_shop_items(rois["merchant_shop_grid"])
            payload["continue_bbox"] = rois["right_arrow_button"]

        elif scene_hint == "campfire_choice":
            payload["campfire_options"] = self._detect_campfire_options(rois["campfire_option_bar"])
            payload["continue_bbox"] = rois["right_arrow_button"]

        elif scene_hint == "campfire_upgrade":
            payload["upgrade_options"] = self._detect_upgrade_options(rois["upgrade_grid"])
            payload["continue_bbox"] = rois["right_arrow_button"]

        elif scene_hint == "campfire_rest_done":
            payload["continue_bbox"] = rois["right_arrow_button"]

        elif scene_hint == "event_unknown":
            payload["continue_bbox"] = rois["right_arrow_button"]

        return payload

    def _detect_buttons(
            self,
            rois: Dict[str, List[int]],
            scene_hint: str,
            title_menu_options: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if scene_hint == "title_main":
            menu_options = list(title_menu_options or [])
            if menu_options and menu_options[0].get("bbox"):
                return [{"name": "single_mode", "bbox": menu_options[0]["bbox"], "confidence": 0.95}]
            return [{"name": "single_mode", "bbox": rois["title_single_mode"], "confidence": 0.95}]
        if scene_hint == "mode_select":
            return [
                {"name": "standard_mode", "bbox": rois["single_standard_card"], "confidence": 0.95},
                {"name": "back", "bbox": rois["back_button"], "confidence": 0.88},
            ]
        if scene_hint == "character_select":
            return [
                {"name": "ironclad", "bbox": rois["character_ironclad_card"], "confidence": 0.90},
                {"name": "confirm_character", "bbox": rois["character_confirm_button"], "confidence": 0.92},
                {"name": "back", "bbox": rois["back_button"], "confidence": 0.88},
            ]
        if scene_hint == "map":
            return [{"name": "back", "bbox": rois["back_button"], "confidence": 0.88}]
        if scene_hint == "battle":
            return [{"name": "end_turn", "bbox": rois["end_turn_button"], "confidence": 0.85}]
        if scene_hint in {"battle_result", "card_reward", "merchant", "merchant_shop", "campfire_choice",
                          "campfire_upgrade", "campfire_rest_done", "event_unknown"}:
            return [{"name": "continue", "bbox": rois["right_arrow_button"], "confidence": 0.78}]
        return []

    def _detect_menu_options(self, rois: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = rois["title_menu_column"]
        h = y2 - y1
        items = [
            ("single_mode", [x1, y1 + int(h * 0.02), x2, y1 + int(h * 0.16)]),
            ("multi_mode", [x1, y1 + int(h * 0.16), x2, y1 + int(h * 0.30)]),
            ("timeline", [x1, y1 + int(h * 0.30), x2, y1 + int(h * 0.44)]),
            ("settings", [x1, y1 + int(h * 0.44), x2, y1 + int(h * 0.58)]),
            ("encyclopedia", [x1, y1 + int(h * 0.58), x2, y1 + int(h * 0.72)]),
            ("quit", [x1, y1 + int(h * 0.72), x2, y2]),
        ]
        return [{"name": name, "bbox": bbox, "confidence": 0.75} for name, bbox in items]

    def _detect_mode_options(self, rois: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        return [
            {"name": "standard_mode", "bbox": rois["single_standard_card"], "confidence": 0.95},
            {"name": "daily_mode", "bbox": rois["single_daily_card"], "confidence": 0.90},
            {"name": "custom_mode", "bbox": rois["single_custom_card"], "confidence": 0.90},
        ]

    def _detect_character_options(self, rois: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = rois["character_portrait_strip"]
        w = x2 - x1
        step = max(1, w // 5)
        names = ["ironclad", "silent", "defect", "watcher", "unknown"]
        result: List[Dict[str, Any]] = []
        for i, name in enumerate(names):
            bx1 = x1 + step * i
            bx2 = x1 + step * (i + 1) if i < 4 else x2
            result.append({"name": name, "bbox": [bx1, y1, bx2, y2], "confidence": 0.70})
        return result

    def _detect_selected_character(self, image: Image.Image, rois: Dict[str, List[int]]) -> Optional[str]:
        iron_b = self._brightness(image, rois["character_ironclad_card"])
        iron_warm = self._warm_ratio(image, rois["character_ironclad_card"])
        second_b = self._brightness(image, rois["character_second_card"])
        if iron_warm > 0.16 or iron_b > second_b + 6:
            return "ironclad"
        return None

    def _detect_cards(self, image: Image.Image, hand_area: List[int]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = hand_area
        hand_width = x2 - x1
        hand_height = y2 - y1
        slot_count = 6
        slot_w = max(1, hand_width // slot_count)
        detected: List[Dict[str, Any]] = []
        for i in range(slot_count):
            sx1 = x1 + i * slot_w + int(slot_w * 0.08)
            sx2 = x1 + (i + 1) * slot_w - int(slot_w * 0.08)
            sy1 = y1 + int(hand_height * 0.05)
            sy2 = y2 - int(hand_height * 0.05)
            bbox = [sx1, sy1, sx2, sy2]
            contrast = self._contrast(image, bbox)
            brightness = self._brightness(image, bbox)
            if contrast > 14 and brightness > 35:
                template = self.CARD_TEMPLATES[len(detected) % len(self.CARD_TEMPLATES)]
                detected.append({"name": template["name"], "bbox": bbox, "cost": template["cost"], "type": template["type"], "confidence": round(min(0.95, 0.45 + contrast / 50.0), 2)})
        return detected

    def _detect_enemies(self, image: Image.Image, enemy_area: List[int]) -> List[Dict[str, Any]]:
        contrast = self._contrast(image, enemy_area)
        brightness = self._brightness(image, enemy_area)
        if contrast < 8 or brightness < 20:
            return []
        return [{"id": "enemy_1", "name": "Unknown Enemy", "bbox": enemy_area, "hp": None, "intent": "unknown", "confidence": round(min(0.90, 0.40 + contrast / 40.0), 2)}]

    def _detect_map_options(self, map_area: List[int]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = map_area
        w = x2 - x1
        h = y2 - y1
        return [
            {"id": "node_center", "name": "node_center", "bbox": [x1 + int(w * 0.47), y1 + int(h * 0.46), x1 + int(w * 0.53), y1 + int(h * 0.54)], "kind": "enemy", "confidence": 0.72},
            {"id": "node_left", "name": "node_left", "bbox": [x1 + int(w * 0.28), y1 + int(h * 0.58), x1 + int(w * 0.34), y1 + int(h * 0.66)], "kind": "unknown", "confidence": 0.60},
            {"id": "node_right", "name": "node_right", "bbox": [x1 + int(w * 0.66), y1 + int(h * 0.58), x1 + int(w * 0.72), y1 + int(h * 0.66)], "kind": "unknown", "confidence": 0.60},
        ]

    def _detect_reward_options(self, reward_area: List[int]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = reward_area
        w = x2 - x1
        h = y2 - y1
        boxes = [
            [x1 + int(w * 0.08), y1 + int(h * 0.40), x1 + int(w * 0.92), y1 + int(h * 0.54)],
            [x1 + int(w * 0.08), y1 + int(h * 0.56), x1 + int(w * 0.92), y1 + int(h * 0.70)],
            [x1 + int(w * 0.08), y1 + int(h * 0.72), x1 + int(w * 0.92), y1 + int(h * 0.86)],
        ]
        return [{"id": f"choice_{idx}", "name": f"choice_{idx}", "bbox": bbox, "confidence": 0.72 if idx == 1 else 0.65} for idx, bbox in enumerate(boxes, start=1)]

    def _detect_card_reward_options(self, panel: List[int]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = panel
        w = x2 - x1
        boxes = [
            [x1 + int(w * 0.05), y1, x1 + int(w * 0.30), y2],
            [x1 + int(w * 0.36), y1, x1 + int(w * 0.61), y2],
            [x1 + int(w * 0.67), y1, x1 + int(w * 0.92), y2],
        ]
        return [{"id": f"card_reward_{idx}", "name": f"card_reward_{idx}", "bbox": bbox, "confidence": 0.78} for idx, bbox in enumerate(boxes, start=1)]

    def _detect_shop_items(self, grid: List[int]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = grid
        w = x2 - x1
        h = y2 - y1
        cols, rows = 5, 2
        cell_w = max(1, w // cols)
        cell_h = max(1, h // rows)
        items: List[Dict[str, Any]] = []
        idx = 1
        for r in range(rows):
            for c in range(cols):
                bx1 = x1 + c * cell_w + int(cell_w * 0.05)
                by1 = y1 + r * cell_h + int(cell_h * 0.08)
                bx2 = x1 + (c + 1) * cell_w - int(cell_w * 0.05)
                by2 = y1 + (r + 1) * cell_h - int(cell_h * 0.08)
                items.append({"id": f"shop_item_{idx}", "name": f"shop_item_{idx}", "bbox": [bx1, by1, bx2, by2], "confidence": 0.70})
                idx += 1
        return items

    def _detect_campfire_options(self, bar: List[int]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = bar
        w = x2 - x1
        return [
            {"id": "rest", "name": "rest", "bbox": [x1, y1, x1 + w // 2, y2], "confidence": 0.70},
            {"id": "upgrade", "name": "upgrade", "bbox": [x1 + w // 2, y1, x2, y2], "confidence": 0.70},
        ]

    def _detect_upgrade_options(self, grid: List[int]) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = grid
        w = x2 - x1
        h = y2 - y1
        cols, rows = 5, 2
        cell_w = max(1, w // cols)
        cell_h = max(1, h // rows)
        items: List[Dict[str, Any]] = []
        idx = 1
        for r in range(rows):
            for c in range(cols):
                bx1 = x1 + c * cell_w + int(cell_w * 0.05)
                by1 = y1 + r * cell_h + int(cell_h * 0.08)
                bx2 = x1 + (c + 1) * cell_w - int(cell_w * 0.05)
                by2 = y1 + (r + 1) * cell_h - int(cell_h * 0.08)
                items.append({"id": f"upgrade_card_{idx}", "name": f"upgrade_card_{idx}", "bbox": [bx1, by1, bx2, by2], "confidence": 0.70})
                idx += 1
        return items

    def _estimate_player_status(self, scene_hint: str, image: Image.Image, rois: Dict[str, List[int]], payload: Dict[str, Any]) -> Dict[str, Any]:
        if scene_hint == "battle":
            energy = 3 if payload.get("detected_cards") else 3
            return {"hp": None, "max_hp": None, "energy": energy}
        if scene_hint in {"map", "blessing_choice", "battle_result", "card_reward", "merchant", "merchant_shop", "campfire_choice", "campfire_upgrade", "campfire_rest_done", "event_unknown"}:
            return {"hp": None, "max_hp": None, "energy": 0}
        if scene_hint == "character_select":
            return {"hp": 80, "max_hp": 80, "energy": 0}
        return {"hp": None, "max_hp": None, "energy": 0}

    def _build_base_regions(self, width: int, height: int, offset_x: int, offset_y: int) -> Dict[str, List[int]]:
        def box(x1r: float, y1r: float, x2r: float, y2r: float) -> List[int]:
            return [offset_x + int(width * x1r), offset_y + int(height * y1r), offset_x + int(width * x2r), offset_y + int(height * y2r)]
        return {
            "hand_area": box(0.12, 0.77, 0.88, 0.98),
            "enemy_area": box(0.50, 0.16, 0.92, 0.68),
            "player_hp_bar": box(0.04, 0.88, 0.22, 0.95),
            "energy_area": box(0.39, 0.86, 0.50, 0.98),
            "end_turn_button": box(0.84, 0.70, 0.97, 0.87),
            "map_area": box(0.10, 0.08, 0.90, 0.82),
            "map_legend": box(0.84, 0.29, 0.99, 0.82),
            "reward_area": box(0.18, 0.18, 0.82, 0.88),
            "reward_option_panel": box(0.22, 0.50, 0.80, 0.90),
            "reward_text_bar": box(0.31, 0.54, 0.72, 0.62),
            "card_reward_panel": box(0.08, 0.22, 0.92, 0.78),
            "title_logo_area": box(0.20, 0.20, 0.70, 0.58),
            "title_menu_column": box(0.38, 0.64, 0.54, 0.96),
            "title_single_mode": box(0.40, 0.67, 0.52, 0.75),
            "single_standard_card": box(0.19, 0.18, 0.39, 0.86),
            "single_daily_card": box(0.41, 0.18, 0.61, 0.86),
            "single_custom_card": box(0.63, 0.18, 0.84, 0.86),
            "back_button": box(0.00, 0.70, 0.10, 0.93),
            "continue_button": box(0.08, 0.92, 0.28, 0.995),
            "right_arrow_button": box(0.90, 0.68, 0.99, 0.92),
            "center_modal_button": box(0.40, 0.78, 0.60, 0.92),
            "character_main_area": box(0.00, 0.00, 0.88, 0.90),
            "character_info_panel": box(0.08, 0.28, 0.46, 0.70),
            "character_portrait_strip": box(0.27, 0.80, 0.68, 0.98),
            "character_ironclad_card": box(0.31, 0.82, 0.38, 0.96),
            "character_second_card": box(0.39, 0.82, 0.46, 0.96),
            "character_confirm_button": box(0.86, 0.78, 0.99, 0.97),
            "merchant_shop_grid": box(0.12, 0.16, 0.88, 0.86),
            "campfire_option_bar": box(0.30, 0.10, 0.70, 0.34),
            "upgrade_grid": box(0.10, 0.12, 0.90, 0.90),
        }

    def _crop_abs(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        runtime_bbox = self._normalize_bbox(getattr(self, "runtime_capture_bbox", None))
        if runtime_bbox is not None:
            offset_x = runtime_bbox[0]
            offset_y = runtime_bbox[1]
        else:
            offset_x = self.capture_bbox[0] if self.capture_bbox else 0
            offset_y = self.capture_bbox[1] if self.capture_bbox else 0
        x1, y1, x2, y2 = bbox
        local_x1 = max(0, x1 - offset_x)
        local_y1 = max(0, y1 - offset_y)
        local_x2 = max(local_x1 + 1, x2 - offset_x)
        local_y2 = max(local_y1 + 1, y2 - offset_y)
        return image.crop((local_x1, local_y1, local_x2, local_y2))

    def _np_rgb(self, image: Image.Image, bbox: List[int]) -> np.ndarray:
        crop = self._crop_abs(image, bbox).convert("RGB")
        return np.asarray(crop, dtype=np.uint8)

    def _gray(self, image: Image.Image, bbox: List[int]) -> np.ndarray:
        arr = self._np_rgb(image, bbox).astype(np.float32)
        return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    def _brightness(self, image: Image.Image, bbox: List[int]) -> float:
        return float(self._gray(image, bbox).mean())

    def _contrast(self, image: Image.Image, bbox: List[int]) -> float:
        return float(self._gray(image, bbox).std())

    def _warm_ratio(self, image: Image.Image, bbox: List[int]) -> float:
        arr = self._np_rgb(image, bbox).astype(np.int16)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mask = (r > 110) & (g > 60) & (r > b + 15)
        return float(mask.mean())

    def _gold_ratio(self, image: Image.Image, bbox: List[int]) -> float:
        arr = self._np_rgb(image, bbox).astype(np.int16)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mask = (r > 140) & (g > 100) & (b < 150) & (r > g - 20) & (r > b + 20)
        return float(mask.mean())

    def _red_ratio(self, image: Image.Image, bbox: List[int]) -> float:
        arr = self._np_rgb(image, bbox).astype(np.int16)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mask = (r > 110) & (r > g + 25) & (r > b + 25)
        return float(mask.mean())

    def _blue_ratio(self, image: Image.Image, bbox: List[int]) -> float:
        arr = self._np_rgb(image, bbox).astype(np.int16)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mask = (b > 110) & (b > r + 20) & (b > g + 10)
        return float(mask.mean())

    def _load_image(self, screen_image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(screen_image, Image.Image):
            return screen_image.convert("RGB")
        return Image.open(screen_image).convert("RGB")

    def _normalize_bbox(self, bbox: Any) -> Optional[List[int]]:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if x2 > x1 and y2 > y1:
                    return [x1, y1, x2, y2]
            except Exception:
                return None
        return None

    def _save_debug_result(self, screen_image: Union[str, Image.Image], result: Dict[str, Any]) -> None:
        try:
            stem = Path(screen_image).stem if isinstance(screen_image, str) else f"frame_{datetime.now().strftime('%H%M%S')}"
            file_path = self.debug_dir / f"detector_{stem}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("保存 Detector 调试结果失败: %s", exc)