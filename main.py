from a_platform.capture import STSCapture
from a_platform.executor import STSExecutor
from b_brain.policy import SimplePolicy

def main():
    cap = STSCapture("Slay the Spire", 1280, 720)
    exe = STSExecutor()
    policy = SimplePolicy()

    frame = cap.get_frame()

    # 第一版先假装 parser 已经输出状态
    state = {
        "scene": "battle",
        "available_actions": ["play_card_0", "end_turn"]
    }

    decision = policy.decide(state)
    print("决策结果：", decision)

if __name__ == "__main__":
    main()