from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Hand:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.runtime = config.get("runtime", {})
        self.environment = config.get("environment", {})
        self.runtime_context = config.get("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()
        self.window_name = str(self.environment.get("window_name", "") or "").strip()
        self.bring_window_to_front = bool(self.environment.get("bring_window_to_front", False))

        self.dry_run = bool(self.runtime.get("dry_run", True))
        self.action_delay = float(self.runtime.get("action_delay", 0.2))
        self.move_duration = float(self.runtime.get("move_duration", 0.08))
        self.click_interval = float(self.runtime.get("click_interval", 0.05))
        self.post_action_wait = float(self.runtime.get("post_action_wait", 0.35))

        self.paths = self._init_paths()
        self._enable_windows_dpi_awareness()
        self._pyautogui = self._load_pyautogui()

        logging.info("Hand 初始化完成，dry_run=%s", self.dry_run)

    def _init_paths(self) -> Dict[str, Path]:
        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        result: Dict[str, Path] = {}
        defaults = {
            "action_logs": "logs/action_logs",
            "feedback": "data/feedback",
            "screenshots_current": "screenshots/current",
            "screenshots_history": "screenshots/history",
        }

        for key, default_value in defaults.items():
            value = raw_paths.get(key, default_value)
            path = Path(value)
            if not path.is_absolute():
                path = self.project_root / path
            path.mkdir(parents=True, exist_ok=True)
            result[key] = path.resolve()

        return result

    def _enable_windows_dpi_awareness(self) -> None:
        if platform.system().lower() != "windows":
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            shcore = getattr(ctypes.windll, "shcore", None)
            if shcore is not None:
                try:
                    shcore.SetProcessDpiAwareness(2)
                    logging.info("Hand 已启用 Windows Per-Monitor DPI Awareness。")
                    return
                except Exception:
                    pass
            try:
                user32.SetProcessDPIAware()
                logging.info("Hand 已启用 Windows DPI Awareness。")
            except Exception as exc:
                logging.warning("Hand 启用 Windows DPI Awareness 失败: %s", exc)
        except Exception as exc:
            logging.warning("Hand 初始化 Windows DPI 适配失败: %s", exc)

    def _load_pyautogui(self):
        if self.dry_run:
            return None
        try:
            import pyautogui  # type: ignore
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = self.action_delay
            try:
                screen_size = pyautogui.size()
                logging.info("pyautogui 屏幕尺寸=%sx%s", screen_size.width, screen_size.height)
            except Exception:
                pass
            return pyautogui
        except Exception as exc:
            logging.warning("pyautogui 加载失败，自动切回 dry-run: %s", exc)
            self.dry_run = True
            return None

    def execute(
        self,
        action_data: Dict[str, Any],
        state_data: Dict[str, Any],
        step_id: int,
        eye: Optional[Any] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        before_scene = state_data.get("scene_type", "unknown")
        execution_plan = action_data.get("execution_plan", []) or []

        click_positions: List[List[int]] = []
        execute_status = "success"
        error_message = ""
        state_after = None
        after_scene = before_scene

        try:
            for item in execution_plan:
                position = self._execute_one(item, state_data)
                if position is not None:
                    click_positions.append([position[0], position[1]])
                time.sleep(self.post_action_wait)
        except Exception as exc:
            execute_status = "failed"
            error_message = str(exc)
            logging.exception("执行动作失败: %s", exc)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        screenshot_after = self._save_execution_snapshot(step_id=step_id, action_data=action_data)

        if execute_status == "success" and eye is not None:
            try:
                state_after = eye.observe(step_id=step_id, phase="after_action")
                after_scene = state_after.get("scene_type", before_scene)
            except Exception as exc:
                logging.warning("Hand 执行后二次观察失败: %s", exc)

        feedback_data = {
            "action_type": action_data.get("action_type", "wait"),
            "execute_status": execute_status,
            "before_scene": before_scene,
            "after_scene": after_scene,
            "time_cost_ms": elapsed_ms,
            "click_position": click_positions[0] if click_positions else [],
            "click_positions": click_positions,
            "screen_diff": self._infer_screen_diff(action_data, execute_status, before_scene, after_scene),
            "error_message": error_message,
            "screenshot_after": screenshot_after,
            "timestamp": now_str(),
            "step_id": step_id,
        }
        if state_after is not None:
            feedback_data["state_after"] = state_after

        self._save_feedback_debug(step_id=step_id, feedback_data=feedback_data)
        return feedback_data

    def _execute_one(self, item: Dict[str, Any], state_data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        op = item.get("op", "wait")

        if op == "wait":
            duration = float(item.get("duration", 0.3))
            time.sleep(duration)
            return None

        if op == "click_button":
            pos = self._resolve_button_position(item=item, state_data=state_data)
            self._click(pos)
            return pos

        if op == "click_card":
            pos = self._resolve_card_position(item=item, state_data=state_data)
            self._click(pos)
            return pos

        if op == "click_enemy":
            pos = self._resolve_enemy_position(item=item, state_data=state_data)
            self._click(pos)
            return pos

        if op == "click_map_node":
            pos = self._resolve_map_node_position(item=item, state_data=state_data)
            self._click(pos)
            return pos

        if op == "click_reward":
            pos = self._resolve_reward_position(item=item, state_data=state_data)
            self._click(pos)
            return pos

        raise ValueError(f"未知执行操作 op={op}")

    def _resolve_map_node_position(self, item: Dict[str, Any], state_data: Dict[str, Any]) -> Tuple[int, int]:
        if item.get("node_bbox"):
            return self._center_of_bbox(item["node_bbox"], default=(960, 540))

        options = state_data.get("detected_regions", {}).get("map_options", []) or state_data.get("map_options", [])
        target_name = item.get("node_name")
        target_id = item.get("node_id")
        target_index = item.get("node_index")

        if isinstance(target_index, int) and 0 <= target_index < len(options):
            bbox = options[target_index].get("bbox")
            if bbox:
                return self._center_of_bbox(bbox, default=(960, 540))

        for node in options:
            if target_id and node.get("id") == target_id and node.get("bbox"):
                return self._center_of_bbox(node["bbox"], default=(960, 540))
            if target_name and node.get("name") == target_name and node.get("bbox"):
                return self._center_of_bbox(node["bbox"], default=(960, 540))

        if options and options[0].get("bbox"):
            return self._center_of_bbox(options[0]["bbox"], default=(960, 540))
        return 960, 540

    def _resolve_reward_position(self, item: Dict[str, Any], state_data: Dict[str, Any]) -> Tuple[int, int]:
        if item.get("reward_bbox"):
            return self._center_of_bbox(item["reward_bbox"], default=(960, 540))

        options = state_data.get("detected_regions", {}).get("reward_options", []) or state_data.get("reward_options", [])
        target_name = item.get("reward_name")
        target_id = item.get("reward_id")
        target_index = item.get("reward_index")

        if isinstance(target_index, int) and 0 <= target_index < len(options):
            bbox = options[target_index].get("bbox")
            if bbox:
                return self._center_of_bbox(bbox, default=(960, 540))

        for reward in options:
            if target_id and reward.get("id") == target_id and reward.get("bbox"):
                return self._center_of_bbox(reward["bbox"], default=(960, 540))
            if target_name and reward.get("name") == target_name and reward.get("bbox"):
                return self._center_of_bbox(reward["bbox"], default=(960, 540))

        if options and options[0].get("bbox"):
            return self._center_of_bbox(options[0]["bbox"], default=(960, 540))
        return 960, 540

    def _resolve_card_position(self, item: Dict[str, Any], state_data: Dict[str, Any]) -> Tuple[int, int]:
        cards = state_data.get("detected_regions", {}).get("detected_cards", []) or state_data.get("hand_cards", [])
        card_name = item.get("card_name")
        card_id = item.get("card_id")
        card_index = item.get("card_index")

        if item.get("card_bbox"):
            return self._center_of_bbox(item["card_bbox"], default=(320, 650))

        if isinstance(card_index, int) and 0 <= card_index < len(cards):
            bbox = cards[card_index].get("bbox")
            if bbox:
                return self._center_of_bbox(bbox, default=(320, 650))

        for card in cards:
            if card_id and card.get("id") == card_id and card.get("bbox"):
                return self._center_of_bbox(card["bbox"], default=(320, 650))
            if card_name and card.get("name") == card_name and card.get("bbox"):
                return self._center_of_bbox(card["bbox"], default=(320, 650))

        if cards and cards[0].get("bbox"):
            return self._center_of_bbox(cards[0]["bbox"], default=(320, 650))
        return 320, 650

    def _resolve_enemy_position(self, item: Dict[str, Any], state_data: Dict[str, Any]) -> Tuple[int, int]:
        enemies = state_data.get("detected_regions", {}).get("detected_enemies", []) or state_data.get("enemies", [])
        enemy_name = item.get("enemy_name")
        enemy_id = item.get("enemy_id")
        enemy_index = item.get("enemy_index")

        if item.get("enemy_bbox"):
            return self._center_of_bbox(item["enemy_bbox"], default=(890, 325))

        if isinstance(enemy_index, int) and 0 <= enemy_index < len(enemies):
            bbox = enemies[enemy_index].get("bbox")
            if bbox:
                return self._center_of_bbox(bbox, default=(890, 325))

        for enemy in enemies:
            if enemy_id and enemy.get("id") == enemy_id and enemy.get("bbox"):
                return self._center_of_bbox(enemy["bbox"], default=(890, 325))
            if enemy_name and enemy.get("name") == enemy_name and enemy.get("bbox"):
                return self._center_of_bbox(enemy["bbox"], default=(890, 325))

        enemy_area = state_data.get("detected_regions", {}).get("enemy_area", [])
        return self._center_of_bbox(enemy_area, default=(890, 325))

    def _resolve_button_position(self, item: Dict[str, Any], state_data: Dict[str, Any]) -> Tuple[int, int]:
        button_name = item.get("button_name", "")
        detected_regions = state_data.get("detected_regions", {}) or {}

        for btn in detected_regions.get("detected_buttons", []):
            if btn.get("name") == button_name and btn.get("bbox"):
                return self._center_of_bbox(btn["bbox"], default=(960, 540))

        key_map = {
            "single_mode": "single_mode_bbox",
            "standard_mode": "standard_mode_bbox",
            "confirm_character": "confirm_button_bbox",
            "continue": "continue_bbox",
            "back": "back_button_bbox",
            "end_turn": "end_turn_button",
            "ironclad": "character_ironclad_card",
        }
        key = key_map.get(button_name)
        if key and detected_regions.get(key):
            return self._center_of_bbox(detected_regions[key], default=(960, 540))

        if item.get("button_bbox"):
            return self._center_of_bbox(item["button_bbox"], default=(960, 540))

        defaults = {
            "single_mode": (940, 860),
            "standard_mode": (530, 560),
            "confirm_character": (1820, 970),
            "continue": (310, 1060),
            "back": (80, 900),
            "end_turn": (1160, 570),
            "ironclad": (640, 900),
        }
        return defaults.get(button_name, (960, 540))

    def _center_of_bbox(self, bbox: List[int], default: Tuple[int, int]) -> Tuple[int, int]:
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            return int((x1 + x2) / 2), int((y1 + y2) / 2)
        return default

    def _click(self, pos: Tuple[int, int]) -> None:
        x, y = pos
        if self.dry_run or self._pyautogui is None:
            logging.info("[DRY-RUN] click at (%s, %s)", x, y)
            time.sleep(self.action_delay)
            return

        self._activate_target_window()
        logging.info("执行点击坐标=(%s, %s)", x, y)
        self._pyautogui.moveTo(x, y, duration=self.move_duration)
        self._pyautogui.click(x=x, y=y, interval=self.click_interval)

    def _activate_target_window(self) -> None:
        if not self.window_name or not self.bring_window_to_front:
            return
        try:
            import pygetwindow as gw  # type: ignore

            target_name = " ".join(self.window_name.split()).strip().lower()
            candidates = []
            for win in gw.getAllWindows():
                title = (getattr(win, "title", "") or "").strip()
                norm_title = " ".join(title.split()).strip().lower()
                if not norm_title:
                    continue
                if target_name in norm_title or norm_title in target_name:
                    candidates.append(win)
            if not candidates:
                return
            target = max(candidates, key=lambda w: int(getattr(w, "width", 0)) * int(getattr(w, "height", 0)))
            target.activate()
            time.sleep(0.15)
        except Exception as exc:
            logging.warning("点击前激活目标窗口失败，将继续直接点击: %s", exc)

    def _infer_screen_diff(self, action_data: Dict[str, Any], execute_status: str, before_scene: str, after_scene: str) -> str:
        if execute_status != "success":
            return "execution_failed"
        if before_scene != after_scene:
            return f"scene_changed:{before_scene}->{after_scene}"

        action_type = action_data.get("action_type", "wait")
        mapping = {
            "enter_single_mode": "title_menu_advanced",
            "choose_standard_mode": "mode_selected",
            "select_ironclad": "character_selected",
            "confirm_character": "character_confirmed",
            "continue_act": "continued_to_next_page",
            "play_card": "possible_battle_state_changed",
            "click_card": "card_selected_or_used",
            "click_enemy": "enemy_target_selected",
            "end_turn": "turn_advanced",
            "choose_map_node": "map_route_changed",
            "choose_reward": "reward_selection_changed",
            "wait": "no_obvious_change",
            "finish": "task_finished",
        }
        return mapping.get(action_type, "unknown_change")

    def _save_execution_snapshot(self, step_id: int, action_data: Dict[str, Any]) -> str:
        try:
            from PIL import ImageGrab  # type: ignore

            file_path = self.paths["screenshots_history"] / f"after_action_{step_id:03d}.png"
            current_file = self.paths["screenshots_current"] / "latest_after_action.png"
            image = ImageGrab.grab()
            image.save(file_path)
            image.save(current_file)
            return str(file_path)
        except Exception:
            file_path = self.paths["screenshots_history"] / f"after_action_{step_id:03d}.txt"
            content = (
                f"execution snapshot\n"
                f"step_id={step_id}\n"
                f"action_type={action_data.get('action_type', 'unknown')}\n"
                f"time={now_str()}\n"
                f"dry_run={self.dry_run}\n"
            )
            file_path.write_text(content, encoding="utf-8")
            current_file = self.paths["screenshots_current"] / "latest_after_action.txt"
            current_file.write_text(content, encoding="utf-8")
            return str(file_path)

    def _save_feedback_debug(self, step_id: int, feedback_data: Dict[str, Any]) -> None:
        try:
            action_log_file = self.paths["action_logs"] / f"hand_step_{step_id:03d}.json"
            with action_log_file.open("w", encoding="utf-8") as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("保存 Hand 调试日志失败: %s", exc)