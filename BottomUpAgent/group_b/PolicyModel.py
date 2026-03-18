
"""
BottomUpAgent/group_b/PolicyModel.py

轻量策略模型：
1. 读取 Trainer 产出的统计模型（JSON）
2. 对候选动作进行打分
3. 适合作为 Mcts / Brain 的 learned_score 辅助层
4. 不依赖第三方 ML 框架，先把训练闭环跑起来
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from BottomUpAgent.group_b.StateAdapter import StateAdapter
except Exception:  # pragma: no cover
    try:
        from StateAdapter import StateAdapter  # type: ignore
    except Exception:  # pragma: no cover
        StateAdapter = None  # type: ignore


class PolicyModel:
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None) -> None:
        self.config = config
        self.runtime_context = config.setdefault("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()

        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        default_model_path = raw_paths.get("policy_model", "data/models/policy_model.json")
        path_value = model_path or default_model_path
        self.model_path = Path(path_value)
        if not self.model_path.is_absolute():
            self.model_path = self.project_root / self.model_path

        self.state_adapter = None
        if StateAdapter is not None:
            try:
                self.state_adapter = StateAdapter(config)
            except Exception as exc:
                logging.warning("PolicyModel 初始化 StateAdapter 失败，将仅接受外部 state_repr: %s", exc)

        self.model: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}
        self.global_action_stats: Dict[str, Any] = {}
        self.scene_action_stats: Dict[str, Any] = {}
        self.signature_action_stats: Dict[str, Any] = {}
        self.scene_target_kind_stats: Dict[str, Any] = {}
        self.scene_button_stats: Dict[str, Any] = {}
        self.memory_priority_action_stats: Dict[str, Any] = {}
        self.skill_key_action_stats: Dict[str, Any] = {}
        self.loaded = False

        self.load()

    def load(self) -> bool:
        if not self.model_path.exists():
            logging.info("PolicyModel 未找到模型文件，将以空模型运行: %s", self.model_path)
            self.loaded = False
            return False
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("policy model payload 不是 dict")
            self.model = payload
            self.meta = dict(payload.get("meta", {}) or {})
            self.global_action_stats = dict(payload.get("global_action_stats", {}) or {})
            self.scene_action_stats = dict(payload.get("scene_action_stats", {}) or {})
            self.signature_action_stats = dict(payload.get("signature_action_stats", {}) or {})
            self.scene_target_kind_stats = dict(payload.get("scene_target_kind_stats", {}) or {})
            self.scene_button_stats = dict(payload.get("scene_button_stats", {}) or {})
            self.memory_priority_action_stats = dict(payload.get("memory_priority_action_stats", {}) or {})
            self.skill_key_action_stats = dict(payload.get("skill_key_action_stats", {}) or {})
            self.loaded = True
            logging.info("PolicyModel 加载完成: %s", self.model_path)
            return True
        except Exception as exc:
            logging.warning("PolicyModel 加载失败，将以空模型运行: %s", exc)
            self.loaded = False
            return False

    def score_action(
        self,
        candidate_action: Dict[str, Any],
        *,
        state_data: Optional[Dict[str, Any]] = None,
        state_repr: Optional[Dict[str, Any]] = None,
        feature_dict: Optional[Dict[str, Any]] = None,
    ) -> float:
        scene_type, state_signature = self._resolve_state_keys(
            state_data=state_data,
            state_repr=state_repr,
            feature_dict=feature_dict,
        )

        action_type = str(candidate_action.get("action_type", "unknown") or "unknown")
        kind = self._extract_target_kind(candidate_action)
        button = self._extract_button(candidate_action)
        memory_priority = self._extract_memory_priority(candidate_action)
        skill_key = self._build_skill_key(scene_type, candidate_action)

        score = 0.0
        weight = 0.0

        # 全局动作统计
        score, weight = self._accumulate(score, weight, self._lookup_mean(self.global_action_stats, action_type), 0.10)

        # 场景-动作统计
        scene_stats = dict(self.scene_action_stats.get(scene_type, {}) or {})
        score, weight = self._accumulate(score, weight, self._lookup_mean(scene_stats, action_type), 0.30)

        # 状态签名-动作统计
        signature_stats = dict(self.signature_action_stats.get(state_signature, {}) or {})
        score, weight = self._accumulate(score, weight, self._lookup_mean(signature_stats, action_type), 0.45)

        # 地图 kind / 商店 / 营火等额外信号
        if kind:
            kind_stats = dict(self.scene_target_kind_stats.get(scene_type, {}) or {})
            score, weight = self._accumulate(score, weight, self._lookup_mean(kind_stats, f"{action_type}::{kind}"), 0.10)

        if button:
            btn_stats = dict(self.scene_button_stats.get(scene_type, {}) or {})
            score, weight = self._accumulate(score, weight, self._lookup_mean(btn_stats, f"{action_type}::{button}"), 0.05)

        if memory_priority:
            priority_stats = dict(self.memory_priority_action_stats.get(memory_priority, {}) or {})
            score, weight = self._accumulate(score, weight, self._lookup_mean(priority_stats, action_type), 0.07)

        if skill_key:
            skill_stats = dict(self.skill_key_action_stats.get(skill_key, {}) or {})
            score, weight = self._accumulate(score, weight, self._lookup_mean(skill_stats, action_type), 0.08)

        if weight <= 0:
            return 0.0
        return round(max(0.0, min(1.0, score / weight)), 4)

    def score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        state_data: Optional[Dict[str, Any]] = None,
        state_repr: Optional[Dict[str, Any]] = None,
        feature_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in candidates:
            s = self.score_action(item, state_data=state_data, state_repr=state_repr, feature_dict=feature_dict)
            scored.append((s, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def pick_best(
        self,
        candidates: List[Dict[str, Any]],
        *,
        state_data: Optional[Dict[str, Any]] = None,
        state_repr: Optional[Dict[str, Any]] = None,
        feature_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        if not candidates:
            return 0.0, {}
        scored = self.score_candidates(candidates, state_data=state_data, state_repr=state_repr, feature_dict=feature_dict)
        return scored[0]

    def _resolve_state_keys(
        self,
        *,
        state_data: Optional[Dict[str, Any]],
        state_repr: Optional[Dict[str, Any]],
        feature_dict: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        scene_type = "unknown"
        state_signature = "unknown"

        if state_repr:
            scene_type = str(state_repr.get("scene_type", "unknown") or "unknown")
            state_signature = str(state_repr.get("state_signature", "unknown") or "unknown")
            return scene_type, state_signature

        if state_data and self.state_adapter is not None:
            try:
                adapted = self.state_adapter.adapt(state_data)
                scene_type = str(adapted.get("scene_type", "unknown") or "unknown")
                state_signature = str(adapted.get("state_signature", "unknown") or "unknown")
                return scene_type, state_signature
            except Exception:
                pass

        if feature_dict:
            scene_type = str(feature_dict.get("scene_type", "unknown") or "unknown")
        return scene_type, state_signature

    def _lookup_mean(self, bucket: Dict[str, Any], key: str) -> Optional[float]:
        raw = bucket.get(key)
        if not isinstance(raw, dict):
            return None
        try:
            return float(raw.get("mean", raw.get("avg", 0.0)))
        except Exception:
            return None

    def _accumulate(self, score: float, weight: float, mean: Optional[float], w: float) -> Tuple[float, float]:
        if mean is None:
            return score, weight
        return score + mean * w, weight + w

    def _extract_target_kind(self, action: Dict[str, Any]) -> str:
        target = action.get("target", {}) if isinstance(action.get("target"), dict) else {}
        for key in ("kind", "node_kind", "reward_kind", "item_kind"):
            value = target.get(key) or action.get("params", {}).get(key) if isinstance(action.get("params"), dict) else None
            if value:
                return str(value).lower()
        if target.get("button"):
            return ""
        name = target.get("name")
        if name:
            return str(name).lower()
        return ""

    def _extract_button(self, action: Dict[str, Any]) -> str:
        target = action.get("target", {}) if isinstance(action.get("target"), dict) else {}
        button = target.get("button")
        if button:
            return str(button).lower()
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        button = params.get("button_name")
        return str(button).lower() if button else ""

    def _extract_memory_priority(self, action: Dict[str, Any]) -> str:
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        value = params.get("memory_priority") or action.get("memory_priority")
        return str(value).lower() if value else ""

    def _build_skill_key(self, scene_type: str, action: Dict[str, Any]) -> str:
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        explicit = params.get("skill_key") or action.get("skill_key")
        if explicit:
            return str(explicit)
        target = action.get("target", {}) if isinstance(action.get("target"), dict) else {}
        target_name = target.get("kind") or target.get("button") or target.get("name") or "generic"
        return f"{scene_type}::{str(action.get('action_type', 'unknown') or 'unknown')}::{str(target_name).lower()}"
