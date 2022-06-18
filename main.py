from time import sleep
# from picamera import PiCamera
# from picamera.array import PiRGBArray
import cv2 as cv
import numpy as np
from utils import gui

from utils.dartboard import Dartboard
import consts


def get_similarity(img_a, img_b) -> float:
    error_l2 = cv.norm(img_a, img_b, cv.NORM_L2)
    return 1 - error_l2 / (consts.TRANSFORM_X * consts.TRANSFORM_Y)


def main_loop():
    dartboard = Dartboard()

    # camera = PiCamera()
    # camera.consts.RESOLUTION = (consts.RESOLUTION_X, consts.RESOLUTION_Y)
    # camera.framerate = consts.FRAMERATE
    # raw_capture = PiRGBArray(camera, size=(consts.RESOLUTION_X, consts.RESOLUTION_Y))
    camera = cv.VideoCapture('IMG_1016.mov')  # cv specific

    # Wait for camera to become ready
    # sleep(1)
    prev_frame = np.zeros((consts.TRANSFORM_X, consts.TRANSFORM_Y, 3), np.uint8)
    frame_number = 1

    # for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
    while camera.isOpened():   # cv specific
        ret, img = camera.read()  # cv specific
        # img = frame.array
        if ret:  # cv specific

            if not dartboard.calibrated:
                dartboard.update_perspective_mat(img)

            img_face = cv.warpPerspective(img, dartboard.perspective_matrix, (consts.TRANSFORM_X, consts.TRANSFORM_Y))

            # Calculate difference between frames
            similarity = get_similarity(prev_frame, img_face)

            # If difference is too great, recalculate perspective matrix
            if similarity < consts.RECALIBRATE_THRESH:
                dartboard.update_perspective_mat(img)

            elif similarity < consts.DART_THRESH and frame_number > 1:
                dartboard.get_points(img_face)
                print(frame_number, similarity)
                gui.showImage(prev_frame)
                gui.showImage(img_face)
                break

            # Clear steam in preparation for next frame
            frame_number += 1
            prev_frame = img_face
            # raw_capture.truncate(0)
        else:
            break

    camera.release()  # cv specific


if __name__ == 'main':
    main_loop()
