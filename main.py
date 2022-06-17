from time import sleep
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2 as cv
import numpy as np

from dartboard import Dartboard


RESOLUTION_X, RESOLUTION_Y = 1920, 1080

dartboard = Dartboard()

camera = PiCamera()
camera.resolution = (RESOLUTION_X, RESOLUTION_Y)
camera.framerate = 15
raw_capture = PiRGBArray(camera, size=(RESOLUTION_X, RESOLUTION_Y))

# Wait for camera to become ready
sleep(1)
prev_frame = np.zeros((RESOLUTION_X, RESOLUTION_Y, 3), )

for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
    img = frame.array

    if not dartboard.calibrated:
        dartboard.update_perspective_mat(img)

    img_face = cv.perspectiveTransform(img, dartboard.perspective_matrix)

    # Calculate difference between frames
    error_l2 = cv.norm(prev_frame, img_face, cv.NORM_L2)
    similarity = 1 - error_l2 / (RESOLUTION_X * RESOLUTION_Y)
    if similarity < 0.99:
        dartboard.get_points(img_face)

    # Clear steam in preparation for next frame
    prev_frame = img_face
    raw_capture.truncate(0)
