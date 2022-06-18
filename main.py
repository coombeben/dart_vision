import time
# from picamera import PiCamera
# from picamera.array import PiRGBArray
import cv2 as cv
import numpy as np
from utils import gui

from utils.dartboard import Dartboard
import utils.vision as vision
import consts


def get_similarity(img_a, img_b) -> float:
    error_l2 = cv.norm(img_a, img_b, cv.NORM_L2)
    return 1 - error_l2 / (consts.TRANSFORM_X * consts.TRANSFORM_Y)


# def main_loop():
dartboard = Dartboard()

# camera = PiCamera()
# camera.consts.RESOLUTION = (consts.RESOLUTION_X, consts.RESOLUTION_Y)
# camera.framerate = consts.FRAMERATE
# raw_capture = PiRGBArray(camera, size=(consts.RESOLUTION_X, consts.RESOLUTION_Y))
camera = cv.VideoCapture('IMG_1016.mov')  # cv specific

# Wait for camera to become ready
# time.sleep(1)
prev_frame = np.zeros((consts.TRANSFORM_X, consts.TRANSFORM_Y, 3), np.uint8)
frame_number = 1
back_sub = cv.createBackgroundSubtractorMOG2(history=30, detectShadows=True)
# back_sub = cv.bgsegm.createBackgroundSubtractorCNT()

start_time = time.time()

# for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
while camera.isOpened():  # cv specific
    ret, frame = camera.read()  # cv specific
    # frame = frame.array
    if ret:  # cv specific

        # if not dartboard.calibrated:
        #     dartboard.update_perspective_mat(frame)
        dartboard.update_perspective_mat(frame)

        if dartboard.perspective_matrix is not None:
            frame_adj = cv.warpPerspective(frame, dartboard.perspective_matrix,
                                           (consts.TRANSFORM_X, consts.TRANSFORM_Y))
            foregound_mask = back_sub.apply(frame_adj)

            # Calculate difference between frames
            similarity = get_similarity(prev_frame, frame_adj)

            # If difference is too great, recalculate perspective matrix
            if similarity < consts.RECALIBRATE_THRESH:
                dartboard.update_perspective_mat(frame)
                frame_adj = cv.warpPerspective(frame, dartboard.perspective_matrix,
                                               (consts.TRANSFORM_X, consts.TRANSFORM_Y))

            elif similarity < consts.DART_THRESH and frame_number > 1:

                # print(frame_number, similarity)
                # gui.showImage(prev_frame)
                # gui.showImage(foregound_mask)
                gui.showImage(frame)
                gui.showImage(frame_adj)

                thresh = vision.clean_diff(foregound_mask)
                conts, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                largest_cont, _ = vision.get_largest_contor(conts)
                frame_conts = cv.drawContours(frame_adj, [largest_cont], 0, consts.GREEN, 3)

                gui.showImage(frame_conts)

                # impact_point = vision.get_arrow_point(largest_cont)
                # if impact_point is not None:
                #     frame_conts = cv.circle(frame_adj, impact_point, 4, consts.GREEN, 3)
                #     gui.showImage(frame_conts)
                # else:
                #     print('Nothing returned')
                # bounding_box = vision.get_arrow_point(frame_adj, largest_cont)
                # gui.showImage(bounding_box)

                exec_time = time.time() - start_time
                print(f'Exection time: {np.round(exec_time, 2)}, fps: {np.round(frame_number / exec_time, 2)}')
                break

            prev_frame = frame_adj

        # Clear steam in preparation for next frame
        frame_number += 1
        # raw_capture.truncate(0)
    else:
        break

camera.release()  # cv specific


# if __name__ == '__main__':
#     main_loop()
