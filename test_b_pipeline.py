
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("game.yaml 不是 dict")
    return data

def main() -> int:
    project_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    from BottomUpAgent.group_b.StateAdapter import StateAdapter
    from BottomUpAgent.group_b.StateEncoder import StateEncoder
    from BottomUpAgent.group_b.TrajectoryLogger import TrajectoryLogger

    config = load_yaml(project_root / "config" / "game.yaml")
    config["_project_root"] = str(project_root)
    config.setdefault("_runtime_context", {})

    adapter = StateAdapter(config)
    encoder = StateEncoder()
    logger = TrajectoryLogger(config)

    sample_state = {
        "scene_type": "map",
        "scene_variant": None,
        "floor": 5,
        "hp": 61,
        "max_hp": 80,
        "energy": 3,
        "map_options": [
            {"id": "node_1", "kind": "battle", "bbox": [1, 2, 3, 4]},
            {"id": "node_2", "kind": "event_unknown", "bbox": [5, 6, 7, 8]},
        ],
        "reward_options": [],
        "hand_cards": [],
        "enemies": [],
        "match_confidence": 0.88,
        "flow_corrected": True,
        "flow_reason": "room_exit_prefer_map",
        "detected_buttons": [{"name": "back"}],
        "available_buttons": ["back"],
    }
    sample_action = {
        "action_type": "choose_map_node",
        "target": {"id": "node_2", "kind": "event_unknown"},
        "reason": "演示：优先走问号房",
        "confidence": 0.74,
        "source": "Brain+Mcts",
    }
    sample_feedback = {
        "execute_status": "success",
        "before_scene": "map",
        "after_scene": "event_unknown",
        "screen_diff": "major_change",
        "time_cost_ms": 632,
    }
    sample_teacher = {
        "score": 0.87,
        "feedback": "动作执行效果较好",
        "should_promote_to_skill": True,
    }

    state_repr = adapter.adapt(sample_state, step_id=1, episode_id="debug_episode")
    features = adapter.build_feature_dict(state_repr)
    keys, vec = encoder.encode_to_vector(features)

    episode_id = logger.start_episode(task="debug_pipeline")
    record = logger.log_step(
        step_id=1,
        state_data=sample_state,
        action_data=sample_action,
        feedback_data=sample_feedback,
        teacher_feedback=sample_teacher,
        memory_summary={"skill_count": 3, "total_records": 12, "recent_skills": []},
        task="debug_pipeline",
        episode_id=episode_id,
    )
    logger.end_episode({"steps": 1}, episode_id=episode_id)

    output = {
        "episode_id": episode_id,
        "state_repr": state_repr,
        "features": features,
        "vector_preview": dict(list(zip(keys, vec))[:15]),
        "record_preview": {
            "episode_id": record["episode_id"],
            "step_id": record["step_id"],
            "label": record["label"],
            "state_signature": (record.get("state_repr") or {}).get("state_signature"),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
