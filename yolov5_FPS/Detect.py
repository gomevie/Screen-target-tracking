# coding=utf-8

import argparse
import math
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from ScreenShot import screenshot, Screen_size
from SendInput import *

import pynput
from pynput.mouse import Listener

import pyautogui

is_x2_pressed = False

def mouse_check(x, y, button, pressed):
    # print(button, pressed)
    global is_x2_pressed
    if pressed and button == pynput.mouse.Button.x2:
        print('执行鼠标操作')
        is_x2_pressed = True
    elif not pressed and button == pynput.mouse.Button.x2:
        print('鼠标操作结束')
        is_x2_pressed = False

def mouse_listen():
    with Listener(on_click=mouse_check) as listener:
        listener.join()

@smart_inference_mode()
def run():
    weights = 'weights/yolov5n.pt'

    # device = torch.device('cpu')
    # fp16 = False
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        fp16 = True
        print('use gpu')
    else:
        device = torch.device('cpu')
        fp16 = False
        print('use cpu')

    # Load model
    model = DetectMultiBackend(weights=weights, device=device, dnn=False, data=False, fp16=fp16)

    while True:
        # 读取图片
        im = screenshot()
        im0 = im

        # 图片处理
        im = letterbox(im0, (640, 640), stride=32, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # Run inference
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45, classes=0, max_det=1000)


        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, line_width=1)
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                distance_list = []
                target_list = []
                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    line = cls, *xywh, conf  # label format

                    # print(xywh, line)

                    x = xywh[0] - Screen_size
                    y = xywh[1] - Screen_size

                    distance = math.sqrt(x ** 2 + y ** 2)
                    xywh.append(distance)
                    annotator.box_label(xyxy, label=f"{int(cls)} Distance:{round(distance, 2)}", color=(34, 139, 34),
                                        txt_color=(0, 191, 255))

                    distance_list.append(distance)
                    target_list.append([-1*x, -1*y])

                target_info = target_list[distance_list.index(min(distance_list))]

                if is_x2_pressed:
                    print(target_info)
                    mouse_xy(int(target_info[0] / 2), int(target_info[1] / 2))
                    time.sleep(0.03) # 防止推理速度过快导致出错
                    # 模拟鼠标左键点击
                    # pyautogui.click()

            im0 = annotator.result()
            cv2.imshow('window', im0)
            cv2.waitKey(1)

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


if __name__ == "__main__":
    threading.Thread(target=mouse_listen).start()
    run()
