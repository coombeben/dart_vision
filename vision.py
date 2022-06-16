import cv2 as cv
import numpy as np

import gui
# If cv2.imshow is not used, should use opencv-python-headless

GREEN = (0, 255, 0)


def crop_image(img):
    apriltags = cv.aruco.DICT_APRILTAG_36h11

    height, width = img.shape[:2]

    region_top = 0
    region_left = width
    region_right = 0
    region_bottom = height

    aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv.aruco.DetectorParameters_create()
    corners, ids, rejected = cv.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

    ids = ids.flatten()
    if (np.sort(ids) != np.array(range(4))).any():
        print('Id tag error')

    for (marker_corner_arr, marker_id) in zip(corners, ids):
        marker_corners = marker_corner_arr.reshape((4, 2)).astype('int')
        # top_left, top_right, bottom_right, bottom_left = marker_corners
        bottom_right, bottom_left, top_left, top_right = marker_corners

        if marker_id == 0:
            region_top = max(bottom_right[1], bottom_left[1])
        if marker_id == 1:
            region_left = max(top_right[0], bottom_right[0])
        if marker_id == 2:
            region_right = min(top_left[0], bottom_left[0])
        if marker_id == 3:
            region_bottom = min(top_left[1], top_right[1])

        # cv.aruco.drawDetectedMarkers(img, corners, ids)

    return img[region_top:region_bottom, region_left:region_right]


def fill_holes(img, thresh_param=127):
    _, im_th = cv.threshold(img, thresh_param, 255, cv.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    return im_th | im_floodfill_inv


img = cv.imread('IMG_1010.jpg')  # cv.IMREAD_GRAYSCALE

img_cropped = crop_image(img)

img_grey = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(img_grey, THRESH_PARAM, 255, cv.THRESH_BINARY)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# cnt = contours[0]
# cv.drawContours(img_cropped, contours, -1, GREEN, 3)
# ellipse = cv.fitEllipse(cnt)
# cv.ellipse(img_cropped, ellipse, GREEN, 2)

filled = fill_holes(img_grey, 120)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
thresh = cv.morphologyEx(filled, cv.MORPH_OPEN, kernel)
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)


gui.showImage(img)
gui.showImage(img_cropped)
gui.showImage(thresh)
#
# kernel = cv.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
#
