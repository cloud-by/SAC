# a_platform/executor.py
import pyautogui
import time

class STSExecutor:
    def __init__(self):
        pyautogui.PAUSE = 0.2

    def click(self, x, y):
        pyautogui.click(x, y)

    def end_turn(self):
        # 这里先写死一个坐标，后面再配到 config
        pyautogui.click(1180, 650)

    def confirm(self):
        pyautogui.click(1100, 650)