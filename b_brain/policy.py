# b_brain/policy.py
class SimplePolicy:
    def decide(self, state):
        scene = state["scene"]

        if scene == "battle":
            for action in state["available_actions"]:
                if action.startswith("play_card"):
                    return {
                        "action": action,
                        "reason": "战斗场景下优先出牌"
                    }
            return {
                "action": "end_turn",
                "reason": "当前无可执行有效牌，结束回合"
            }

        if scene == "reward":
            return {
                "action": "confirm",
                "reason": "奖励界面优先确认"
            }

        if scene == "map":
            return {
                "action": "choose_path_0",
                "reason": "地图界面选择第一条可行路径"
            }

        return {
            "action": "noop",
            "reason": "未知场景，暂不执行"
        }