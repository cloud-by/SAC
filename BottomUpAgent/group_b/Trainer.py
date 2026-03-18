
"""
BottomUpAgent/group_b/Trainer.py

轻量训练器：
1. 从 data/trajectories/*.jsonl 读取轨迹
2. 聚合 scene/action/state_signature 的统计值
3. 产出 PolicyModel 可直接加载的 JSON 模型
4. 不依赖 sklearn / torch，先把课程项目训练闭环接起来
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Trainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.runtime_context = config.setdefault("_runtime_context", {})
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()

        raw_paths = self.runtime_context.get("paths", self.config.get("paths", {}))
        self.trajectories_dir = self._resolve_path(raw_paths.get("trajectories", "data/trajectories"), mkdir=True)
        self.models_dir = self._resolve_path(raw_paths.get("models", "data/models"), mkdir=True)
        self.model_path = self._resolve_path(raw_paths.get("policy_model", "data/models/policy_model.json"), mkdir=False)

        logging.info("Trainer 初始化完成，trajectories=%s, model=%s", self.trajectories_dir, self.model_path)

    def train(self, limit_files: Optional[int] = None) -> Dict[str, Any]:
        records = list(self._iter_records(limit_files=limit_files))
        stats = self._aggregate(records)
        model = {
            "meta": {
                "created_at": now_str(),
                "record_count": len(records),
                "source_dir": str(self.trajectories_dir),
            },
            **stats,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Trainer 训练完成，records=%s, model=%s", len(records), self.model_path)
        return model

    def fit(self, limit_files: Optional[int] = None) -> Dict[str, Any]:
        return self.train(limit_files=limit_files)

    def _iter_records(self, limit_files: Optional[int] = None) -> Iterable[Dict[str, Any]]:
        files = sorted(self.trajectories_dir.glob("run_*.jsonl"))
        if limit_files is not None:
            files = files[:limit_files]
        for path in files:
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if "state_data" not in payload or "action_data" not in payload:
                        continue
                    yield payload
            except Exception as exc:
                logging.warning("读取轨迹文件失败，已跳过: %s (%s)", path, exc)

    def _aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        global_action: DefaultDict[str, List[float]] = defaultdict(list)
        scene_action: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        signature_action: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        scene_target_kind: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        scene_button: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for record in records:
            state_repr = dict(record.get("state_repr", {}) or {})
            action_data = dict(record.get("action_data", {}) or {})
            label = dict(record.get("label", {}) or {})
            teacher = dict(record.get("teacher_feedback", {}) or {})

            scene_type = str(state_repr.get("scene_type", "unknown") or "unknown")
            state_signature = str(state_repr.get("state_signature", "unknown") or "unknown")
            action_type = str(action_data.get("action_type", "unknown") or "unknown")
            value = self._resolve_value(label, teacher)

            global_action[action_type].append(value)
            scene_action[scene_type][action_type].append(value)
            signature_action[state_signature][action_type].append(value)

            kind = self._extract_target_kind(action_data)
            if kind:
                scene_target_kind[scene_type][f"{action_type}::{kind}"].append(value)

            button = self._extract_button(action_data)
            if button:
                scene_button[scene_type][f"{action_type}::{button}"].append(value)

        return {
            "global_action_stats": self._finalize_bucket(global_action),
            "scene_action_stats": self._finalize_nested(scene_action),
            "signature_action_stats": self._finalize_nested(signature_action),
            "scene_target_kind_stats": self._finalize_nested(scene_target_kind),
            "scene_button_stats": self._finalize_nested(scene_button),
        }

    def _resolve_value(self, label: Dict[str, Any], teacher: Dict[str, Any]) -> float:
        try:
            if "value" in label:
                return float(label["value"])
        except Exception:
            pass
        try:
            if "score" in teacher:
                return float(teacher["score"])
        except Exception:
            pass
        return 0.0

    def _finalize_bucket(self, bucket: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for key, values in bucket.items():
            if not values:
                continue
            result[key] = self._stats(values)
        return result

    def _finalize_nested(self, bucket: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for outer_key, inner in bucket.items():
            result[outer_key] = self._finalize_bucket(inner)
        return result

    def _stats(self, values: List[float]) -> Dict[str, Any]:
        count = len(values)
        total = sum(values)
        mean = total / count if count else 0.0
        return {
            "count": count,
            "mean": round(mean, 6),
            "sum": round(total, 6),
            "max": round(max(values), 6) if values else 0.0,
            "min": round(min(values), 6) if values else 0.0,
        }

    def _extract_target_kind(self, action: Dict[str, Any]) -> str:
        target = action.get("target", {}) if isinstance(action.get("target"), dict) else {}
        if target.get("kind"):
            return str(target.get("kind")).lower()
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        for key in ("node_kind", "reward_kind", "item_kind"):
            if params.get(key):
                return str(params.get(key)).lower()
        name = target.get("name")
        return str(name).lower() if name else ""

    def _extract_button(self, action: Dict[str, Any]) -> str:
        target = action.get("target", {}) if isinstance(action.get("target"), dict) else {}
        button = target.get("button")
        if button:
            return str(button).lower()
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        button = params.get("button_name")
        return str(button).lower() if button else ""

    def _resolve_path(self, value: str, *, mkdir: bool) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
