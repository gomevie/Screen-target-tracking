# coding=utf-8
import threading

import pynput
from pynput.mouse import Listener
import pyautogui

def mouse_check(x, y, button, pressed):
    print(button, pressed)
    if pressed and button == pynput.mouse.Button.x2:
        print('侧键2已经按下')




def mouse_listen():
    with Listener(on_click=mouse_check) as listener:
        listener.join()


if __name__ == "__main__":
    threading.Thread(target=mouse_listen).start()
    # 模拟鼠标左键点击
    for i in range(10):
        pyautogui.click()