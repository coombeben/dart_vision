import cv2 as cv
import numpy as np
import timeit
import time

import consts
import utils.vision as vision
import utils.gui as gui
from utils.dartboard_detector import Detector
from utils.dart_detector import DartDetector


start_time = time.time()
camera = cv.VideoCapture('IMG_E1024.mov')
# back_sub = cv.createBackgroundSubtractorMOG2(history=100)
detector = Detector()
frame_number = 0

make_new_subtractor = True
masks = None
significant_frames = [-3]
historic = []
frame_scores = []

while camera.isOpened():
    ret, frame = camera.read()
    if ret:
        frame_number += 1
        frame_adj, recalculate = detector.correct_image(frame)
        if frame_adj is None:
            make_new_subtractor = True
        else:
            if make_new_subtractor:
                # Reset the subtractor if the board becomes obscured (i.e. someone removes their darts)
                back_sub = cv.createBackgroundSubtractorMOG2(history=100)
                make_new_subtractor = False

            foreground_mask = back_sub.apply(frame_adj)

            simm = cv.mean(foreground_mask)[0]
            last_sig_frame = max(significant_frames)
            if simm > 5 and frame_number > 1:
                if frame_number - 1 == last_sig_frame:
                    significant_frames.append(frame_number)
                    masks = np.dstack((masks, foreground_mask))
                    frame_scores.append(simm)
                else:
                    significant_frames = [frame_number]
                    masks = np.expand_dims(foreground_mask, axis=2)
                    frame_scores = [simm]
            elif frame_number - 1 == last_sig_frame:
                best_frame_idx = frame_scores.index(max(frame_scores))
                best_mask = masks[:, :, best_frame_idx]
                masks = None  # Clear some memory
                gui.showImage(best_mask, f'Score: {frame_scores[best_frame_idx]}, Frame: {significant_frames[best_frame_idx]}')
                # Process best mask
    else:
        break

camera.release()

print(f'Execution time: {round(time.time() - start_time, 2)}')

