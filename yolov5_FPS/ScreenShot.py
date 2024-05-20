# coding=utf-8

from mss import mss
import cv2
import numpy as np

ScreenX = 2880
ScreenY = 1800
Screen_size = 480

window_size = (
    int(ScreenX / 2 - Screen_size),
    int(ScreenY / 2 - Screen_size),
    int(ScreenX / 2 + Screen_size),
    int(ScreenY / 2 + Screen_size)
)

Screenshot_value = mss()

def screenshot():
    img = Screenshot_value.grab(window_size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# while True:
#     cv2.imshow('a', np.array(screenshot()))
#     cv2.waitKey(1)
