from time import sleep
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2 as cv

from dartboard import Dartboard

dartboard = Dartboard()

camera = PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 15
raw_capture = PiRGBArray(camera, size=(1980, 1080))

sleep(1)

for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
    img = frame.array

    if not dartboard.calibrated:
        dartboard.update_perspective_mat(img)

    raw_capture.truncate(0)
