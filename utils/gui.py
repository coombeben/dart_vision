from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

import consts

"""Misc debug function for printing result"""


def showImage(img):
    """Shows img with axis indicating (x, y) coords"""
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), interpolation='nearest')
    plt.show()


def draw_polar_line(img, src, theta):
    """Given a center and angle, computes and plots a polar line of radius height/2"""
    r = img.shape[0] // 2

    dst = (int(r * 1/np.cos(theta)) + src[0], int(r * 1/np.sin(theta)) + src[1])

    return cv.line(img, src, dst, consts.GREEN, 2)

