from __future__ import annotations

from BottomUpAgent.common.protocols import ensure_action_protocol, ensure_feedback_protocol, ensure_state_protocol, ensure_teacher_protocol
from BottomUpAgent.group_b.Brain import Brain
from BottomUpAgent.group_b.Mcts import Mcts
from BottomUpAgent.group_b.Teacher import Teacher
from BottomUpAgent.group_b.LongMemory import LongMemory
from BottomUpAgent.group_b.TrajectoryLogger import TrajectoryLogger
from BottomUpAgent.group_b.Trainer import Trainer
from BottomUpAgent.group_b.PolicyModel import PolicyModel
from BottomUpAgent.common.config_loader import load_yaml_file
from pathlib import Path


def _config():
    root = Path(__file__).resolve().parent
    config = load_yaml_file(root / 'config' / 'game.yaml') or {}
    config['_project_root'] = str(root)
    config.setdefault('_runtime_context', {})
    return config


def main() -> int:
    state = ensure_state_protocol({}, step_id=1, phase='before')
    assert state['scene_type'] == 'unknown'
    assert state['hand_cards'] == []

    action = ensure_action_protocol({}, scene_type='battle', episode_id='ep1', step_id=1)
    assert action['action_type'] == 'wait'
    assert action['scene_type'] == 'battle'
    assert action['params']['episode_id'] == 'ep1'
    assert action['params']['step_id'] == 1

    feedback = ensure_feedback_protocol(
        {},
        action_type='wait',
        before_scene='battle',
        after_scene='battle',
        elapsed_ms=10,
        screenshot_after='',
        step_id=1,
        screen_diff='no_change',
    )
    assert feedback['execute_status'] == 'success'
    assert feedback['before_scene'] == 'battle'

    teacher = ensure_teacher_protocol({}, step_id=1, scene_type='battle', action_type='wait', execute_status='success', episode_id='ep1')
    assert teacher['memory_priority'] == 'low'
    assert teacher['skill_key'] == 'battle::wait'

    config = _config()
    brain = Brain(config)
    result = brain.plan('Play', {'scene_type': 'map', 'map_options': []}, {}, 1)
    assert result['action_type'] in {'wait', 'continue', 'choose_map_node'}
    assert result['scene_type'] == 'map'
    assert result['params']['step_id'] == 1

    mcts = Mcts(config)
    searched = mcts.search(
        result,
        {
            'scene_type': 'map',
            'map_options': [
                {'id': 'node_1', 'kind': 'battle'},
                {'id': 'node_2', 'kind': 'event_unknown'},
            ],
        },
        2,
    )
    assert searched['scene_type'] == 'map'
    assert searched['params']['step_id'] == 2
    assert searched['params']['mcts_candidate_count'] >= 1

    teacher_model = Teacher(config)
    teacher_feedback = teacher_model.reflect(
        {'scene_type': 'map', 'episode_id': 'ep1'},
        searched,
        {'execute_status': 'success', 'before_scene': 'map', 'after_scene': 'event_unknown', 'screen_diff': 'major_change'},
        3,
    )
    assert teacher_feedback['action_type'] == searched['action_type']
    assert teacher_feedback['scene_type'] == 'map'
    assert teacher_feedback['memory_priority'] in {'low', 'medium', 'high'}
    assert isinstance(teacher_feedback['outcome_tags'], list)

    memory = LongMemory(config)
    skill = memory.update_skill(
        {'scene_type': 'map', 'episode_id': 'ep1'},
        searched,
        {'execute_status': 'success', 'before_scene': 'map', 'after_scene': 'event_unknown', 'screen_diff': 'major_change', 'step_id': 3},
        teacher_feedback,
    )
    assert skill['skill_id'] == teacher_feedback['skill_key']
    assert skill['memory_priority'] == teacher_feedback['memory_priority']
    assert isinstance(skill['outcome_tags'], list)

    logger = TrajectoryLogger(config)
    sample = logger.build_training_sample(
        step_id=4,
        state_data={'scene_type': 'map', 'episode_id': 'ep1'},
        action_data=searched,
        feedback_data={'execute_status': 'success', 'before_scene': 'map', 'after_scene': 'event_unknown', 'screen_diff': 'major_change'},
        teacher_feedback=teacher_feedback,
        skill_data=skill,
        memory_summary={'skill_count': 1},
        episode_id='ep1',
    )
    assert sample['teacher_feedback']['memory_priority'] == teacher_feedback['memory_priority']
    assert sample['skill_data']['skill_id'] == skill['skill_id']
    assert sample['label']['skill_key'] == teacher_feedback['skill_key']

    trainer = Trainer(config)
    aggregated = trainer._aggregate([sample])
    assert 'memory_priority_action_stats' in aggregated
    assert 'skill_key_action_stats' in aggregated
    assert sample['label']['memory_priority'] in aggregated['memory_priority_action_stats']
    assert teacher_feedback['skill_key'] in aggregated['skill_key_action_stats']

    policy = PolicyModel(config)
    policy.global_action_stats = aggregated['global_action_stats']
    policy.scene_action_stats = aggregated['scene_action_stats']
    policy.signature_action_stats = aggregated['signature_action_stats']
    policy.scene_target_kind_stats = aggregated['scene_target_kind_stats']
    policy.scene_button_stats = aggregated['scene_button_stats']
    policy.memory_priority_action_stats = aggregated['memory_priority_action_stats']
    policy.skill_key_action_stats = aggregated['skill_key_action_stats']
    score = policy.score_action(
        {
            'action_type': sample['action_data']['action_type'],
            'target': sample['action_data'].get('target', {}),
            'params': {
                'memory_priority': sample['label']['memory_priority'],
                'skill_key': sample['label']['skill_key'],
            },
        },
        state_repr=sample['state_repr'],
    )
    assert score >= 0.0

    mcts.policy_model = policy
    linked = mcts.search(
        sample['action_data'],
        {'scene_type': 'map', 'map_options': [{'id': 'node_2', 'kind': 'event_unknown'}], 'hp': 60, 'max_hp': 80},
        5,
    )
    assert 'mcts_learned_score' in linked['params']
    assert 'mcts_final_score' in linked['params']
    return 0


if __name__ == '__main__':
    raise SystemExit(main())