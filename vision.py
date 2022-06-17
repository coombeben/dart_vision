import cv2 as cv
import numpy as np

import gui
# If cv2.imshow is not used, should use opencv-python-headless

GREEN = (0, 255, 0)


def crop_image(img):
    apriltags = cv.aruco.DICT_APRILTAG_36h11

    height, width = img.shape[:2]
    region_top, region_left, region_right, region_bottom = (0, width, 0, height)

    aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv.aruco.DetectorParameters_create()
    corners, ids, rejected = cv.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

    ids = ids.flatten()
    if (np.sort(ids) != np.array(range(4))).any():
        print('Id tag error')

    for (marker_corner_arr, marker_id) in zip(corners, ids):
        marker_corners = marker_corner_arr.reshape((4, 2)).astype('int')
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
    _, img_th = cv.threshold(img, thresh_param, 255, cv.THRESH_BINARY_INV)
    img_floodfill = img_th.copy()
    h, w = img_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(img_floodfill, mask, (0, 0), 255)
    img_floodfill_inv = cv.bitwise_not(img_floodfill)
    return img_th | img_floodfill_inv


def perspective_correction(img_cropped, thresh):
    h, w = thresh.shape[:2]

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = []
    max_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt

    ellipse = cv.fitEllipse(largest_contour)
    # img_cropped = cv.drawContours(img_cropped, largest_contour, -1, GREEN, 3)
    # cv.ellipse(img_cropped, ellipse, GREEN, 3)

    e_center = ellipse[0]
    e_size = ellipse[1]

    top, right, bottom, left = ([e_center[0], e_center[1] - e_size[1] / 2], [e_center[0] + e_size[0] / 2, e_center[1]],
                                [e_center[0], e_center[1] + e_size[1] / 2], [e_center[0] - e_size[0] / 2, e_center[1]])

    source_pts = np.float32([top, right,
                             bottom, left])
    dest_pts = np.float32([[w / 2, 0], [w, h / 2],
                           [w / 2, h], [0, h / 2]])

    perspective_mat = cv.getPerspectiveTransform(source_pts, dest_pts)
    return cv.warpPerspective(img_cropped, perspective_mat, (w, h))


def main():
    img = cv.imread('IMG_1010.jpg')  # cv.IMREAD_GRAYSCALE

    img_cropped = crop_image(img)

    img_grey = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
    thresh = fill_holes(img_grey, 100)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    img_warped = perspective_correction(img_cropped, thresh)

    # When adjusting thresh_param, need to retry until the bounding box of the fitted ellipse fits within (0, 0), (w, h)

    gui.showImage(img)
    gui.showImage(img_cropped)
    gui.showImage(img_warped)


if __name__ == '__main__':
    main()
