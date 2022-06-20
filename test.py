import cv2 as cv
import numpy as np
import timeit

import consts
import utils.vision as vision
import utils.gui as gui
from utils.dartboard_detector import Detector

frame1 = cv.imread('frame1.jpg')
frame2 = cv.imread('frame2.jpg')

det = Detector()
frame1_adj = det.correct_image(frame1)
frame2_adj = det.correct_image(frame2)

TREBLE_OUTER = consts.TRANSFORM_X // 2 - consts.PAD_SCOREZONE
arr = [TREBLE_OUTER, int(TREBLE_OUTER * (161 / 170)), int(TREBLE_OUTER * (107 / 170)),
       int(TREBLE_OUTER * (98 / 170)), int(TREBLE_OUTER * (16 / 170)), int(TREBLE_OUTER * (6.35 / 170))]

for rad in arr:
    frame1_adj = cv.circle(frame1_adj, (540, 540), rad, consts.GREEN, 3)

gui.showImage(frame1_adj)
