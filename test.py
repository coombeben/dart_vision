import cv2 as cv
import numpy as np
import timeit

import consts
import utils.vision as vision
import utils.gui as gui
from utils.dartboard_detector import Detector
from utils.dart_detector import DartDetector


def check_points(intersect_point, dart, canvas):
    pts, double = dart.get_points(intersect_point)
    print(pts, double)
    show_canvas = cv.circle(canvas.copy(), intersect_point, 3, consts.RED, 3)
    gui.showImage(show_canvas)


frame1 = cv.imread('frame_1.jpg')
frame2 = cv.imread('frame_2.jpg')

det = Detector()
dart = DartDetector()
frame1_adj = det.correct_image(frame1)
frame2_adj = det.correct_image(frame2)

# TREBLE_OUTER = consts.TRANSFORM_X // 2 - consts.PAD_SCOREZONE
# arr = [TREBLE_OUTER, int(TREBLE_OUTER * (161 / 170)), int(TREBLE_OUTER * (107 / 170)),
#        int(TREBLE_OUTER * (98 / 170)), int(TREBLE_OUTER * (16 / 170)), int(TREBLE_OUTER * (6.35 / 170))]
#
# for rad in arr:
#     frame1_adj = cv.circle(frame1_adj, (540, 540), rad, consts.GREEN, 3)
#
# gui.showImage(frame1_adj)

check_points((560, 560), dart, frame1_adj)
