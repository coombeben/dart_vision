import cv2 as cv
import numpy as np
import timeit
import time

import consts
import utils.vision as vision
import utils.gui as gui
from utils.dartboard_detector import Detector
from utils.dart_detector import DartDetector
from utils.frame_grouper import FrameGrouper

camera = cv.VideoCapture('IMG_E1024.mov')
detector = Detector()
grouper = FrameGrouper()
frame_number = 0
sleep_frames = 0

make_new_subtractor = True


start_time = time.time()
while camera.isOpened():
    ret, frame = camera.read()
    frame_number += 1
    if sleep_frames > 0:
        sleep_frames -= 1
    if ret:
        if sleep_frames == 0:
            frame_adj, recalculate = detector.correct_image(frame)
            if frame_adj is None:
                make_new_subtractor = True
                sleep_frames = consts.SLEEP_FRAMES
            else:
                if make_new_subtractor:
                    # Reset the subtractor if the board becomes obscured (i.e. someone removes their darts)
                    back_sub = cv.createBackgroundSubtractorMOG2(history=100)
                    make_new_subtractor = False

                foreground_mask = back_sub.apply(frame_adj)

                simm = cv.mean(foreground_mask)[0]
                if simm > consts.MIN_SIMM and frame_number > 1:
                    grouper.append_frame(frame_number, foreground_mask, simm)
                elif frame_number == grouper.last_sig_frame + 1:
                    best_mask = grouper.get_best_frame()
                    # gui.showImage(best_mask, f'Score: {frame_scores[best_frame_idx]}, Frame: {significant_frames[best_frame_idx]}')
                    # Process best mask
    else:
        break

camera.release()
print(f'Execution time: {round(time.time() - start_time, 2)}')
