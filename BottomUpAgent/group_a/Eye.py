from __future__ import annotations

import importlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Eye:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.environment = config.get("environment", {})
        self.runtime_context = config.setdefault("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()

        self.window_name = self.environment.get("window_name", "Demo Window")
        self.resolution = self.environment.get("resolution", [1280, 720])
        self.capture_bbox = self._normalize_bbox(self.environment.get("capture_bbox"))
        self.capture_mode = self.environment.get("capture_mode", "screen")
        self.bring_window_to_front = bool(self.environment.get("bring_window_to_front", False))

        self.paths = self._init_paths()
        self.detector = self._build_detector()
        self.last_valid_bbox: Optional[List[int]] = None
        self._last_capture_bbox: Optional[List[int]] = None

        logging.info("Eye 初始化完成，window_name=%s, capture_bbox=%s", self.window_name, self.capture_bbox)

    def _init_paths(self) -> Dict[str, Path]:
        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        result: Dict[str, Path] = {}
        defaults = {"screenshots_current": "screenshots/current", "screenshots_history": "screenshots/history"}
        for key, default_value in defaults.items():
            value = raw_paths.get(key, default_value)
            path = Path(value)
            if not path.is_absolute():
                path = self.project_root / path
            path.mkdir(parents=True, exist_ok=True)
            result[key] = path.resolve()
        return result

    def _resolve_capture_bbox(self) -> Optional[List[int]]:
        if self.capture_mode == "window":
            bbox = self._get_window_bbox()
            if bbox is not None:
                self.last_valid_bbox = bbox
                return bbox
            if self.last_valid_bbox is not None:
                logging.warning("本次窗口定位失败，回退到上一次有效客户区 bbox=%s", self.last_valid_bbox)
                return self.last_valid_bbox
        if self.capture_bbox:
            return self.capture_bbox
        return None

    def _get_window_bbox(self) -> Optional[List[int]]:
        try:
            import ctypes
            import ctypes.wintypes
            import time
            import pygetwindow as gw  # type: ignore
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
            user32 = ctypes.windll.user32
            target_name = " ".join(str(self.window_name).split()).strip().lower()
            all_windows = gw.getAllWindows()
            candidates = []
            for w in all_windows:
                title = getattr(w, "title", "") or ""
                norm_title = " ".join(title.split()).strip().lower()
                if not norm_title:
                    continue
                if getattr(w, "isMinimized", False):
                    continue
                if getattr(w, "width", 0) <= 200 or getattr(w, "height", 0) <= 200:
                    continue
                if target_name in norm_title or norm_title in target_name:
                    candidates.append(w)
            if not candidates:
                visible_titles = [(getattr(w, "title", "") or "").strip() for w in all_windows if (getattr(w, "title", "") or "").strip()]
                logging.warning("未找到标题包含 [%s] 的窗口。当前可见窗口标题示例: %s", self.window_name, visible_titles[:15])
                return None
            target = max(candidates, key=lambda w: int(w.width) * int(w.height))
            if self.bring_window_to_front:
                try:
                    target.activate()
                    time.sleep(0.4)
                except Exception as exc:
                    logging.warning("激活目标窗口失败，将继续尝试截图: %s", exc)
            hwnd = int(getattr(target, "_hWnd", 0))
            if hwnd <= 0:
                left = max(0, int(target.left))
                top = max(0, int(target.top))
                right = left + int(target.width)
                bottom = top + int(target.height)
                bbox = [left, top, right, bottom]
                logging.warning("无法获取窗口句柄，回退到窗口外框 bbox=%s", bbox)
                return bbox
            rect = ctypes.wintypes.RECT()
            ok = user32.GetClientRect(hwnd, ctypes.byref(rect))
            if not ok:
                logging.warning("GetClientRect 失败，回退到窗口外框。")
                left = max(0, int(target.left))
                top = max(0, int(target.top))
                right = left + int(target.width)
                bottom = top + int(target.height)
                return [left, top, right, bottom]
            pt1 = ctypes.wintypes.POINT(rect.left, rect.top)
            pt2 = ctypes.wintypes.POINT(rect.right, rect.bottom)
            user32.ClientToScreen(hwnd, ctypes.byref(pt1))
            user32.ClientToScreen(hwnd, ctypes.byref(pt2))
            left = max(0, int(pt1.x))
            top = max(0, int(pt1.y))
            right = max(left + 1, int(pt2.x))
            bottom = max(top + 1, int(pt2.y))
            bbox = [left, top, right, bottom]
            logging.info("已定位目标窗口客户区 title=%s, client_bbox=%s", getattr(target, "title", ""), bbox)
            return bbox
        except Exception as exc:
            logging.warning("按窗口标题定位失败，回退到 capture_bbox/fullscreen: %s", exc)
            return None

    def _normalize_bbox(self, bbox: Any) -> Optional[List[int]]:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if x2 > x1 and y2 > y1:
                    return [x1, y1, x2, y2]
            except Exception:
                return None
        return None

    def _build_detector(self):
        candidate_modules = ["BottomUpAgent.group_a.Detector", "BottomUpAgent.Detector", "Detector"]
        for module_name in candidate_modules:
            try:
                module = importlib.import_module(module_name)
                detector_cls = getattr(module, "Detector", None)
                if detector_cls is not None:
                    return detector_cls(self.config)
            except Exception:
                continue
        logging.warning("Detector 加载失败，Eye 将使用默认检测结果。")
        return None

    def observe(self, step_id: int = 0, phase: str = "before") -> Dict[str, Any]:
        screen_image = self._capture_screen(step_id=step_id, phase=phase)
        detected_regions = self._detect_regions(screen_image=screen_image)
        state_data = {
            "scene_type": self._infer_scene_type(detected_regions),
            "scene_variant": detected_regions.get("scene_variant"),
            "scene_scores": detected_regions.get("scene_scores", {}),
            "match_confidence": detected_regions.get("match_confidence", 0.0),
            "matched_template": detected_regions.get("matched_template"),
            "flow_phase_before": detected_regions.get("flow_phase_before"),
            "flow_phase_after": detected_regions.get("flow_phase_after"),
            "flow_prev_scene": detected_regions.get("flow_prev_scene"),
            "flow_allowed_scenes": detected_regions.get("flow_allowed_scenes", []),
            "flow_corrected": detected_regions.get("flow_corrected", False),
            "flow_reason": detected_regions.get("flow_reason", ""),
            "flow_ordered_candidates": detected_regions.get("flow_ordered_candidates", []),
            "floor": self._infer_floor(detected_regions),
            "hp": self._infer_hp(detected_regions),
            "max_hp": self._infer_max_hp(detected_regions),
            "energy": self._infer_energy(detected_regions),
            "hand_cards": self._infer_hand_cards(detected_regions),
            "enemies": self._infer_enemies(detected_regions),
            "map_options": self._infer_map_options(detected_regions),
            "reward_options": self._infer_reward_options(detected_regions),
            "menu_options": list(detected_regions.get("menu_options", [])),
            "mode_options": list(detected_regions.get("mode_options", [])),
            "character_options": list(detected_regions.get("character_options", [])),
            "selected_character": detected_regions.get("selected_character"),
            "available_buttons": list(detected_regions.get("detected_buttons", [])),
            "screen_image": screen_image,
            "detected_regions": detected_regions,
            "timestamp": now_str(),
            "step_id": step_id,
            "phase": phase,
            "window_name": self.window_name,
            "resolution": self.resolution,
            "window_bbox": detected_regions.get("window_bbox", self._last_capture_bbox),
        }
        self._save_state_snapshot(state_data)
        return state_data

    def _capture_screen(self, step_id: int, phase: str) -> str:
        bbox = self._resolve_capture_bbox()
        self._last_capture_bbox = bbox
        image = self._grab_screen_image(bbox=bbox)
        current_file = self.paths["screenshots_current"] / f"current_{phase}.png"
        history_file = self.paths["screenshots_history"] / f"step_{step_id:03d}_{phase}.png"
        image.save(history_file)
        image.save(current_file)
        return str(history_file)

    def _grab_screen_image(self, bbox: Optional[List[int]] = None):
        try:
            from PIL import ImageGrab  # type: ignore
            if bbox:
                return ImageGrab.grab(bbox=tuple(bbox)).convert("RGB")
            return ImageGrab.grab().convert("RGB")
        except Exception as exc:
            logging.warning("Pillow ImageGrab 截图失败，尝试回退 mss: %s", exc)
        try:
            import mss  # type: ignore
            from PIL import Image  # type: ignore
            with mss.mss() as sct:
                if bbox:
                    x1, y1, x2, y2 = bbox
                    monitor = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
                else:
                    monitor = sct.monitors[1]
                shot = sct.grab(monitor)
                return Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception as exc:
            raise RuntimeError(f"屏幕截图失败，请检查 pillow/mss 安装情况: {exc}") from exc

    def _detect_regions(self, screen_image: str) -> Dict[str, Any]:
        if self.detector is None:
            return self._default_detected_regions()
        try:
            setattr(self.detector, "runtime_capture_bbox", self._last_capture_bbox)
            if hasattr(self.detector, "detect"):
                result = self.detector.detect(screen_image=screen_image)
            elif hasattr(self.detector, "run"):
                result = self.detector.run(screen_image=screen_image)
            else:
                result = self._default_detected_regions()
            if not isinstance(result, dict):
                return self._default_detected_regions()
            return result
        except Exception as exc:
            logging.warning("Detector 检测失败，使用默认区域: %s", exc)
            return self._default_detected_regions()

    def _default_detected_regions(self) -> Dict[str, Any]:
        return {
            "scene_hint": "unknown",
            "scene_variant": None,
            "scene_scores": {},
            "match_confidence": 0.0,
            "matched_template": None,
            "window_bbox": [0, 0, 1280, 720],
            "hand_area": [120, 560, 980, 720],
            "enemy_area": [700, 150, 1150, 500],
            "player_hp_bar": [80, 650, 260, 700],
            "energy_area": [540, 650, 640, 720],
            "end_turn_button": [1080, 520, 1240, 620],
            "detected_cards": [],
            "detected_enemies": [],
            "map_options": [],
            "reward_options": [],
            "menu_options": [],
            "mode_options": [],
            "character_options": [],
            "selected_character": None,
            "detected_buttons": [],
            "player": {"hp": None, "max_hp": None, "energy": 0},
        }

    def _infer_scene_type(self, detected_regions: Dict[str, Any]) -> str:
        return str(detected_regions.get("scene_hint", "unknown"))

    def _infer_floor(self, detected_regions: Dict[str, Any]) -> Optional[int]:
        value = detected_regions.get("floor_hint")
        return int(value) if isinstance(value, int) else None

    def _infer_hp(self, detected_regions: Dict[str, Any]) -> Optional[int]:
        player = detected_regions.get("player", {})
        value = player.get("hp")
        return int(value) if isinstance(value, (int, float)) else None

    def _infer_max_hp(self, detected_regions: Dict[str, Any]) -> Optional[int]:
        player = detected_regions.get("player", {})
        value = player.get("max_hp")
        return int(value) if isinstance(value, (int, float)) else None

    def _infer_energy(self, detected_regions: Dict[str, Any]) -> int:
        player = detected_regions.get("player", {})
        value = player.get("energy")
        return int(value) if isinstance(value, (int, float)) else 0

    def _infer_hand_cards(self, detected_regions: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for card in detected_regions.get("detected_cards", []):
            result.append({"id": card.get("id"), "name": card.get("name", "Unknown Card"), "cost": card.get("cost", 1), "type": card.get("type", "unknown"), "bbox": card.get("bbox", []), "confidence": card.get("confidence", 0.0)})
        return result

    def _infer_enemies(self, detected_regions: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for enemy in detected_regions.get("detected_enemies", []):
            result.append({"id": enemy.get("id"), "name": enemy.get("name", "Unknown Enemy"), "hp": enemy.get("hp"), "intent": enemy.get("intent", "unknown"), "bbox": enemy.get("bbox", []), "confidence": enemy.get("confidence", 0.0)})
        return result

    def _infer_map_options(self, detected_regions: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(detected_regions.get("map_options", []))

    def _infer_reward_options(self, detected_regions: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(detected_regions.get("reward_options", []))

    def _save_state_snapshot(self, state_data: Dict[str, Any]) -> None:
        try:
            state_dir = self.project_root / "data" / "states"
            state_dir.mkdir(parents=True, exist_ok=True)
            step_id = int(state_data.get("step_id", 0))
            phase = str(state_data.get("phase", "before"))
            file_path = state_dir / f"eye_raw_step_{step_id:03d}_{phase}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("保存 Eye 状态快照失败: %s", exc)