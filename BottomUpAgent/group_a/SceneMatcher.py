from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


class SceneMatcher:
    """
    基于相对坐标 ROI + 感知哈希的场景匹配器。
    - 模板目录默认: pic/scene_templates
    - 配置文件默认: config/scene_match.yaml
    - 所有路径都以 project_root 为基准解析
    """

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.project_root = Path(config.get("_project_root", Path.cwd())).resolve()

        scene_match_root = dict(config.get("scene_match", {}) or {})
        file_cfg = self._load_external_config(scene_match_root.get("config_file", "config/scene_match.yaml"))
        self.settings = self._merge_dict(file_cfg, scene_match_root)

        self.enabled = bool(self.settings.get("enabled", True))
        self.template_root = self._resolve_path(self.settings.get("templates_root", "pic/scene_templates"))
        self.cache_dir = self._resolve_path(self.settings.get("cache_dir", "data/template_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hash_method = str(self.settings.get("hash_method", "dhash")).lower()
        self.hash_size = int(self.settings.get("hash_size", 8))
        self.phash_size = int(self.settings.get("phash_size", 32))
        self.min_scene_score = float(self.settings.get("min_scene_score", 0.78))
        self.min_margin = float(self.settings.get("min_margin", 0.06))
        self.allow_recursive_templates = bool(self.settings.get("allow_recursive_templates", True))
        self.scenes_cfg = dict(self.settings.get("scenes", {}) or {})

        self.template_index: Dict[str, List[Dict[str, Any]]] = {}
        self._build_index()

        logging.info(
            "SceneMatcher 初始化完成 enabled=%s, templates_root=%s, hash_method=%s",
            self.enabled,
            self.template_root,
            self.hash_method,
        )

    def match_scene(self, image: Image.Image) -> Dict[str, Any]:
        if not self.enabled or not self.template_index:
            return {
                "scene_hint": "unknown",
                "scene_variant": None,
                "scene_scores": {},
                "match_confidence": 0.0,
                "matched_template": None,
                "margin": 0.0,
                "debug": {"reason": "matcher_disabled_or_no_templates"},
            }

        scene_best: Dict[str, Dict[str, Any]] = {}
        debug_templates: List[Dict[str, Any]] = []

        for scene_name, template_entries in self.template_index.items():
            best_score = -1.0
            best_entry: Optional[Dict[str, Any]] = None
            best_detail: Optional[Dict[str, Any]] = None

            for entry in template_entries:
                score, detail = self._score_template(image, scene_name=scene_name, entry=entry)
                if score > best_score:
                    best_score = score
                    best_entry = entry
                    best_detail = detail

            if best_entry is not None:
                scene_best[scene_name] = {
                    "score": round(best_score, 4),
                    "template_path": best_entry.get("template_path"),
                    "variant": best_entry.get("variant"),
                    "detail": best_detail or {},
                }
                debug_templates.append(
                    {
                        "scene": scene_name,
                        "score": round(best_score, 4),
                        "template_path": best_entry.get("template_path"),
                        "variant": best_entry.get("variant"),
                    }
                )

        if not scene_best:
            return {
                "scene_hint": "unknown",
                "scene_variant": None,
                "scene_scores": {},
                "match_confidence": 0.0,
                "matched_template": None,
                "margin": 0.0,
                "debug": {"reason": "empty_scene_best"},
            }

        ordered = sorted(scene_best.items(), key=lambda kv: kv[1]["score"], reverse=True)
        best_scene, best_meta = ordered[0]
        second_score = ordered[1][1]["score"] if len(ordered) > 1 else 0.0
        margin = round(best_meta["score"] - second_score, 4)

        threshold = float(self.scenes_cfg.get(best_scene, {}).get("threshold", self.min_scene_score))
        accepted = best_meta["score"] >= threshold and margin >= self.min_margin

        scene_hint = best_scene if accepted else "unknown"
        return {
            "scene_hint": scene_hint,
            "scene_variant": best_meta.get("variant"),
            "scene_scores": {k: v["score"] for k, v in scene_best.items()},
            "match_confidence": best_meta["score"],
            "matched_template": best_meta.get("template_path"),
            "margin": margin,
            "debug": {
                "threshold": threshold,
                "accepted": accepted,
                "best_scene": best_scene,
                "best_detail": best_meta.get("detail", {}),
                "scene_rank": debug_templates,
            },
        }

    def _build_index(self) -> None:
        self.template_index = {}
        if not self.template_root.exists():
            logging.warning("SceneMatcher 模板目录不存在: %s", self.template_root)
            return

        for top_level in sorted(p for p in self.template_root.iterdir() if p.is_dir()):
            scene_name = top_level.name
            files = self._list_template_files(top_level)
            for file_path in files:
                rel = file_path.relative_to(top_level)
                variant = rel.parent.as_posix() if str(rel.parent) != "." else None
                roi_hashes = self._build_roi_hashes(file_path=file_path, scene_name=scene_name)
                if not roi_hashes:
                    continue

                entry = {
                    "scene_name": scene_name,
                    "variant": variant,
                    "template_path": str(file_path.relative_to(self.project_root)).replace("\\", "/"),
                    "fingerprints": roi_hashes,
                }
                self.template_index.setdefault(scene_name, []).append(entry)

        cache_path = self.cache_dir / "scene_index.json"
        try:
            cache_path.write_text(
                json.dumps(self.template_index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logging.warning("保存 SceneMatcher 索引缓存失败: %s", exc)

    def _list_template_files(self, root: Path) -> List[Path]:
        iterator = root.rglob("*") if self.allow_recursive_templates else root.glob("*")
        files = [p for p in iterator if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS]
        return sorted(files)

    def _build_roi_hashes(self, file_path: Path, scene_name: str) -> Dict[str, Dict[str, Any]]:
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as exc:
            logging.warning("读取模板图失败 %s: %s", file_path, exc)
            return {}

        rois = list(self.scenes_cfg.get(scene_name, {}).get("rois", []) or [])
        if not rois:
            rois = [{"name": "full", "bbox_rel": [0.0, 0.0, 1.0, 1.0], "weight": 1.0}]

        result: Dict[str, Dict[str, Any]] = {}
        for roi in rois:
            roi_name = str(roi.get("name", "full"))
            bbox_rel = roi.get("bbox_rel", [0.0, 0.0, 1.0, 1.0])
            crop = self._crop_rel(image, bbox_rel)
            if crop is None:
                continue
            fp = self._fingerprint(crop)
            result[roi_name] = {
                "hash": fp,
                "bbox_rel": bbox_rel,
                "weight": float(roi.get("weight", 1.0)),
            }
        return result

    def _score_template(self, image: Image.Image, scene_name: str, entry: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        fps = entry.get("fingerprints", {}) or {}
        if not fps:
            return 0.0, {"reason": "no_fingerprints"}

        score_sum = 0.0
        weight_sum = 0.0
        detail_items = []

        for roi_name, info in fps.items():
            bbox_rel = info.get("bbox_rel", [0.0, 0.0, 1.0, 1.0])
            crop = self._crop_rel(image, bbox_rel)
            if crop is None:
                continue

            current_hash = self._fingerprint(crop)
            template_hash = str(info.get("hash", ""))
            similarity = self._hash_similarity(current_hash, template_hash)
            weight = float(info.get("weight", 1.0))

            score_sum += similarity * weight
            weight_sum += weight
            detail_items.append(
                {
                    "roi": roi_name,
                    "similarity": round(similarity, 4),
                    "weight": weight,
                }
            )

        final_score = score_sum / weight_sum if weight_sum > 0 else 0.0
        return final_score, {"rois": detail_items}

    def _crop_rel(self, image: Image.Image, bbox_rel: List[float]) -> Optional[Image.Image]:
        if not isinstance(bbox_rel, (list, tuple)) or len(bbox_rel) != 4:
            return None

        w, h = image.size
        try:
            x1 = max(0, min(w - 1, int(w * float(bbox_rel[0]))))
            y1 = max(0, min(h - 1, int(h * float(bbox_rel[1]))))
            x2 = max(x1 + 1, min(w, int(w * float(bbox_rel[2]))))
            y2 = max(y1 + 1, min(h, int(h * float(bbox_rel[3]))))
        except Exception:
            return None

        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        crop = ImageOps.grayscale(crop)
        return crop

    def _fingerprint(self, image: Image.Image) -> str:
        if self.hash_method == "phash" and cv2 is not None:
            try:
                return self._phash(image)
            except Exception:
                logging.warning("pHash 计算失败，回退 dHash")
        return self._dhash(image)

    def _dhash(self, image: Image.Image) -> str:
        size = self.hash_size
        img = image.resize((size + 1, size), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.int16)
        diff = arr[:, 1:] > arr[:, :-1]
        bit_str = "".join("1" if v else "0" for v in diff.flatten())
        hex_len = (len(bit_str) + 3) // 4
        return f"{int(bit_str, 2):0{hex_len}x}"

    def _phash(self, image: Image.Image) -> str:
        if cv2 is None:
            return self._dhash(image)

        size = self.phash_size
        small = image.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        dct = cv2.dct(arr)
        dct_low = dct[:8, :8]
        flat = dct_low.flatten()
        median = float(np.median(flat[1:]))
        bits = flat > median
        bit_str = "".join("1" if v else "0" for v in bits)
        hex_len = (len(bit_str) + 3) // 4
        return f"{int(bit_str, 2):0{hex_len}x}"

    def _hash_similarity(self, h1: str, h2: str) -> float:
        if not h1 or not h2:
            return 0.0
        b1 = self._hex_to_bits(h1)
        b2 = self._hex_to_bits(h2)
        n = min(len(b1), len(b2))
        if n == 0:
            return 0.0
        dist = sum(1 for i in range(n) if b1[i] != b2[i])
        return max(0.0, 1.0 - dist / n)

    def _hex_to_bits(self, value: str) -> str:
        value = value.strip().lower().replace("0x", "")
        if not value:
            return ""
        return "".join(f"{int(ch, 16):04b}" for ch in value)

    def _load_external_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        if not config_file:
            return {}
        path = self._resolve_path(config_file)
        if not path.exists():
            return {}
        if yaml is None:
            logging.warning("未安装 PyYAML，无法读取 %s", path)
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logging.warning("读取 SceneMatcher 配置失败 %s: %s", path, exc)
            return {}

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = self._merge_dict(result[k], v)
            else:
                result[k] = v
        return result