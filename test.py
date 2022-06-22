import cv2 as cv
import numpy as np
import time

import consts
from utils.dartboard_detector import Detector
from utils.dart_detector import DartDetector
from utils.frame_grouper import FrameGrouper

detector = Detector()
grouper = FrameGrouper()
dart_detector = DartDetector()

frame_number = 0
next_calculate_frame = 0
next_dart_check_frame = 0

make_new_subtractor = True

camera = cv.VideoCapture("IMG_E1024.mov")
start_time = time.time()

while camera.isOpened():
    ret, frame = camera.read()
    if ret:
        frame_number += 1
        if frame_number == 521:
            print(f'Internal Status.\n'
                  f'Next calculate frame: {next_calculate_frame}\n'
                  f'next_dart_check_frame: {next_dart_check_frame}\n'
                  f'Grouper last sig frame: {grouper.last_sig_frame}')

        if frame_number >= next_calculate_frame:
            frame_adj, recalculate = detector.correct_image(frame, frame_number)
            if frame_adj is None:
                make_new_subtractor = True
                next_calculate_frame = frame_number + consts.RETRY_PERSPECTIVE_FRAMES
            else:
                if make_new_subtractor:
                    # Reset the subtractor if the board becomes obscured (i.e. someone removes their darts)
                    print('Creating new subtractor')
                    back_sub = cv.createBackgroundSubtractorMOG2(history=100)
                    make_new_subtractor = False

                # noinspection PyUnboundLocalVariable
                foreground_mask = back_sub.apply(frame_adj)

                if frame_number >= next_dart_check_frame:
                    simm = cv.mean(foreground_mask)[0]

                    if simm > consts.MIN_SIMM and frame_number > 1:
                        grouper.append_frame(frame_number, foreground_mask, simm)
                    elif frame_number - 1 == grouper.last_sig_frame:
                        best_mask = grouper.get_best_frame()
                        intersect_point = dart_detector.find_dart_b(best_mask)  # , frame_adj, True)
                        if intersect_point is not None:
                            points, double = dart_detector.get_points(intersect_point, frame_adj, True)
                            if points > 0:
                                next_dart_check_frame = frame_number + consts.RETRY_DART_DETECTION_FRAMES
    else:
        break

camera.release()
print(f'Execution time: {round(time.time() - start_time, 2)}')
