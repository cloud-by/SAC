from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class SceneFlowGuard:
    """
    基于《杀戮尖塔2》游玩流程的场景修正器。

    作用：
    1. 给静态模板匹配结果增加“流程上下文”约束
    2. 避免在已开局运行中又误判回 title_main / mode_select / character_select
    3. 对 map / event_unknown 等相近场景，在分数接近时按流程偏置修正

    注意：
    - 这是“软约束”，不是硬编码死板状态机
    - 仍以视觉分数为主，流程只在分数接近或当前结果明显不合理时介入
    """

    PRE_RUN_SCENES = {
        "title_main",
        "mode_select",
        "character_select",
        "blessing_choice",
    }

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

    ALL_SCENES = PRE_RUN_SCENES | IN_RUN_SCENES | {"unknown"}

    PRE_RUN_ONLY_SCENES = {"title_main", "mode_select", "character_select"}
    ROOM_ENTRY_SCENES = {"battle", "merchant", "campfire_choice", "event_unknown"}
    ROOM_EXIT_TO_MAP_SCENES = {"card_reward", "merchant_shop", "campfire_upgrade", "campfire_rest_done"}

    TRANSITIONS_BOOTSTRAP = {
        None: {"title_main", "mode_select", "unknown"},
        "title_main": {"title_main", "mode_select"},
        "mode_select": {"mode_select", "character_select", "title_main"},
        "character_select": {"character_select", "blessing_choice", "mode_select"},
        "blessing_choice": {"blessing_choice", "map"},
        "unknown": ALL_SCENES,
    }

    TRANSITIONS_IN_RUN = {
        None: IN_RUN_SCENES | {"blessing_choice", "unknown"},
        "map": {"map", "battle", "merchant", "campfire_choice", "event_unknown", "unknown"},
        "battle": {"battle", "battle_result", "unknown"},
        "battle_result": {"battle_result", "card_reward", "map", "blessing_choice", "unknown"},
        "card_reward": {"card_reward", "map", "blessing_choice", "unknown"},
        "merchant": {"merchant", "merchant_shop", "map", "unknown"},
        "merchant_shop": {"merchant_shop", "merchant", "map", "unknown"},
        "campfire_choice": {"campfire_choice", "campfire_upgrade", "campfire_rest_done", "map", "unknown"},
        "campfire_upgrade": {"campfire_upgrade", "map", "unknown"},
        "campfire_rest_done": {"campfire_rest_done", "map", "unknown"},
        "event_unknown": {"event_unknown", "battle", "merchant", "campfire_choice", "map", "battle_result", "unknown"},
        "blessing_choice": {"blessing_choice", "map", "unknown"},
        "unknown": IN_RUN_SCENES | {"blessing_choice", "unknown"},
    }

    def __init__(self, runtime_context: Dict[str, Any]) -> None:
        self.runtime_context = runtime_context
        self.flow_state = runtime_context.setdefault(
            "scene_flow",
            {
                "phase": "bootstrap",
                "last_scene": None,
                "history": [],
                "observed_in_run": False,
            },
        )

    def resolve(
        self,
        raw_scene: str,
        scene_scores: Optional[Dict[str, float]],
        match_confidence: float,
        legacy_scores: Optional[Dict[str, float]] = None,
        scene_variant: Optional[str] = None,
    ) -> Dict[str, Any]:
        prev_scene = self._safe_scene(self.flow_state.get("last_scene"))
        phase_before = str(self.flow_state.get("phase", "bootstrap") or "bootstrap")

        combined_scores = self._combine_scores(scene_scores or {}, legacy_scores or {})
        ordered = self._sort_scores(combined_scores)
        top_scene = raw_scene if raw_scene in combined_scores else (ordered[0][0] if ordered else "unknown")
        top_score = float(combined_scores.get(top_scene, match_confidence if top_scene != "unknown" else 0.0))

        allowed = self._allowed_scenes(prev_scene=prev_scene, phase=phase_before)
        selected_scene = top_scene
        flow_corrected = False
        flow_reason = "top_candidate_accepted"

        # 1) 如果已经在 run 中，强力阻止误回前置菜单场景。
        if phase_before == "in_run" and top_scene in self.PRE_RUN_ONLY_SCENES:
            alt_scene, alt_score = self._best_allowed_candidate(ordered, allowed)
            if alt_scene is not None and alt_score >= max(0.40, top_score - 0.22):
                selected_scene = alt_scene
                flow_corrected = True
                flow_reason = f"in_run_block_pre_run_scene:{top_scene}->{alt_scene}"

        # 2) 如果当前 top1 不符合流程，尝试选出最合理的 allowed 候选。
        if selected_scene not in allowed:
            alt_scene, alt_score = self._best_allowed_candidate(ordered, allowed)
            if alt_scene is not None and alt_score >= max(0.35, top_score - 0.18):
                selected_scene = alt_scene
                flow_corrected = True
                flow_reason = f"transition_fix:{top_scene}->{alt_scene}"

        # 3) map / 房间 纠结时，加一点流程偏置。
        selected_scene, map_room_fix = self._apply_map_room_bias(
            selected_scene=selected_scene,
            ordered=ordered,
            combined_scores=combined_scores,
            prev_scene=prev_scene,
            phase=phase_before,
            match_confidence=match_confidence,
        )
        if map_room_fix is not None:
            flow_corrected = True
            flow_reason = map_room_fix

        # 4) 战斗结算后，若 blessing_choice 与 map 非常接近，优先 blessing_choice。
        selected_scene, post_boss_fix = self._apply_post_boss_bias(
            selected_scene=selected_scene,
            combined_scores=combined_scores,
            prev_scene=prev_scene,
        )
        if post_boss_fix is not None:
            flow_corrected = True
            flow_reason = post_boss_fix

        # 5) 高置信主界面，视为一次新开局 / 回到主菜单。
        if (
            selected_scene == "title_main"
            and combined_scores.get("title_main", 0.0) >= 0.92
            and phase_before == "in_run"
        ):
            self._reset_flow()
            phase_before = "bootstrap"
            prev_scene = None
            flow_corrected = True
            flow_reason = "strong_title_main_reset_flow"

        phase_after = self._next_phase(current_phase=phase_before, selected_scene=selected_scene)
        self._update_flow(selected_scene=selected_scene, phase_after=phase_after)

        return {
            "scene_hint": selected_scene,
            "scene_variant": scene_variant,
            "flow_phase_before": phase_before,
            "flow_phase_after": phase_after,
            "flow_prev_scene": prev_scene,
            "flow_allowed_scenes": sorted(allowed),
            "flow_corrected": flow_corrected,
            "flow_reason": flow_reason,
            "flow_ordered_candidates": ordered[:6],
        }

    def _apply_map_room_bias(
        self,
        selected_scene: str,
        ordered: Sequence[Tuple[str, float]],
        combined_scores: Dict[str, float],
        prev_scene: Optional[str],
        phase: str,
        match_confidence: float,
    ) -> Tuple[str, Optional[str]]:
        if not ordered:
            return selected_scene, None

        if prev_scene == "map" and phase == "in_run" and selected_scene == "map":
            room_scene, room_score = self._best_scene_from_set(
                ordered,
                self.ROOM_ENTRY_SCENES | {"battle_result"},
            )
            map_score = float(combined_scores.get("map", 0.0))
            if (
                room_scene is not None
                and room_score >= 0.52
                and map_score - room_score <= 0.06
                and match_confidence < 0.88
            ):
                return room_scene, f"left_map_prefer_room:{selected_scene}->{room_scene}"

        if prev_scene in self.ROOM_EXIT_TO_MAP_SCENES and selected_scene != "map":
            map_score = float(combined_scores.get("map", 0.0))
            selected_score = float(combined_scores.get(selected_scene, 0.0))
            if map_score >= 0.48 and map_score >= selected_score - 0.08:
                return "map", f"room_exit_prefer_map:{selected_scene}->map"

        if prev_scene == "event_unknown" and selected_scene == "map":
            event_score = float(combined_scores.get("event_unknown", 0.0))
            map_score = float(combined_scores.get("map", 0.0))
            if event_score >= 0.46 and map_score - event_score <= 0.05:
                return "event_unknown", "event_chain_keep_event_unknown"

        return selected_scene, None

    def _apply_post_boss_bias(
        self,
        selected_scene: str,
        combined_scores: Dict[str, float],
        prev_scene: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        if prev_scene == "battle_result" and selected_scene == "map":
            blessing_score = float(combined_scores.get("blessing_choice", 0.0))
            map_score = float(combined_scores.get("map", 0.0))
            if blessing_score >= 0.42 and map_score - blessing_score <= 0.06:
                return "blessing_choice", "post_boss_prefer_blessing_choice"
        return selected_scene, None

    def _allowed_scenes(self, prev_scene: Optional[str], phase: str) -> set[str]:
        if phase == "in_run":
            return set(self.TRANSITIONS_IN_RUN.get(prev_scene, self.TRANSITIONS_IN_RUN["unknown"]))
        return set(self.TRANSITIONS_BOOTSTRAP.get(prev_scene, self.TRANSITIONS_BOOTSTRAP["unknown"]))

    def _next_phase(self, current_phase: str, selected_scene: str) -> str:
        if selected_scene in self.IN_RUN_SCENES:
            return "in_run"
        if current_phase == "in_run":
            return "in_run"
        return "bootstrap"

    def _update_flow(self, selected_scene: str, phase_after: str) -> None:
        history = list(self.flow_state.get("history", []))
        history.append(selected_scene)
        history = history[-12:]
        self.flow_state["history"] = history
        self.flow_state["last_scene"] = selected_scene
        self.flow_state["phase"] = phase_after
        if selected_scene in self.IN_RUN_SCENES:
            self.flow_state["observed_in_run"] = True

    def _reset_flow(self) -> None:
        self.flow_state["phase"] = "bootstrap"
        self.flow_state["last_scene"] = None
        self.flow_state["history"] = []
        self.flow_state["observed_in_run"] = False

    def _combine_scores(self, scene_scores: Dict[str, float], legacy_scores: Dict[str, float]) -> Dict[str, float]:
        combined: Dict[str, float] = {}
        for scene, score in scene_scores.items():
            combined[str(scene)] = max(0.0, float(score))
        for scene, score in legacy_scores.items():
            norm = min(max(float(score), 0.0) / 5.0, 0.95)
            combined[str(scene)] = max(combined.get(str(scene), 0.0), norm)
        if "unknown" not in combined:
            combined["unknown"] = 0.0
        return combined

    def _sort_scores(self, combined_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        ordered = sorted(
            ((self._safe_scene(scene), float(score)) for scene, score in combined_scores.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ordered

    def _best_allowed_candidate(
        self,
        ordered: Sequence[Tuple[str, float]],
        allowed: set[str],
    ) -> Tuple[Optional[str], float]:
        for scene, score in ordered:
            if scene in allowed:
                return scene, float(score)
        return None, 0.0

    def _best_scene_from_set(
        self,
        ordered: Sequence[Tuple[str, float]],
        scene_set: set[str],
    ) -> Tuple[Optional[str], float]:
        for scene, score in ordered:
            if scene in scene_set:
                return scene, float(score)
        return None, 0.0

    def _safe_scene(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        scene = str(value)
        return scene if scene in self.ALL_SCENES else scene