# a_platform/capture.py
import cv2
import numpy as np
import mss
import win32gui

class STSCapture:
    def __init__(self, window_name: str, width: int, height: int):
        self.window_name = window_name
        self.width = width
        self.height = height

    def get_frame(self):
        hwnd = win32gui.FindWindow(None, self.window_name)
        if not hwnd:
            raise RuntimeError("找不到《杀戮尖塔》窗口，请先启动游戏并保持标题为 Slay the Spire")

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        with mss.mss() as sct:
            monitor = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top
            }
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = cv2.resize(img, (self.width, self.height))
            return img