import cv2
import cv2 as cv
import numpy as np


def crop_image(img):
    apriltags = cv.aruco.DICT_APRILTAG_36h11

    height, width = img.shape[:2]
    green = (0, 255, 0)

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


# If cv2.imshow is not used, should use opencv-python-headless
img = cv.imread('IMG_1010.jpg')  # cv.IMREAD_GRAYSCALE

img = crop_image(img)

kernel = cv.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))

cv.imshow('Lines', img)
cv.waitKey(0)
cv.destroyAllWindows()
