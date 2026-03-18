
from __future__ import annotations

import json
import sys
from pathlib import Path

def _load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def main():
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    config = _load_yaml(project_root / "config" / "game.yaml")
    config["_project_root"] = str(project_root)
    config.setdefault("_runtime_context", {})
    from BottomUpAgent.group_b.Trainer import Trainer
    from BottomUpAgent.group_b.PolicyModel import PolicyModel

    trainer = Trainer(config)
    model = trainer.train()
    policy = PolicyModel(config)
    summary = {
        "record_count": model.get("meta", {}).get("record_count", 0),
        "model_path": str(policy.model_path),
        "global_actions": list((model.get("global_action_stats") or {}).keys())[:10],
        "scene_count": len(model.get("scene_action_stats") or {}),
        "loaded": policy.loaded,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
