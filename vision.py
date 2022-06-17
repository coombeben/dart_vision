import cv2 as cv
import numpy as np
import math

import gui
# If cv2.imshow is not used, should use opencv-python-headless

GREEN = (0, 255, 0)


def crop_image(img):
    """Finds a reasonable region to start work from using the position of the 4 apriltags"""
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


# def fill_holes(img, thresh_param=127):
#     """Returns a threshold image of the board"""
#     _, img_th = cv.threshold(img, thresh_param, 255, cv.THRESH_BINARY_INV)
#     img_floodfill = img_th.copy()
#     h, w = img_th.shape[:2]
#     mask = np.zeros((h+2, w+2), np.uint8)
#     cv.floodFill(img_floodfill, mask, (0, 0), 255)
#     img_floodfill_inv = cv.bitwise_not(img_floodfill)
#     return img_th | img_floodfill_inv


def get_face(img):
    """Returns the play face of the dartboard"""
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Find a threshold of only green and red shapes
    lower_red_hue_rng = cv.inRange(img_hsv, (0, 100, 100), (10, 255, 255))
    upper_red_hug_rng = cv.inRange(img_hsv, (160, 100, 100), (179, 255, 255))
    img_red_hue = cv.addWeighted(lower_red_hue_rng, 1, upper_red_hug_rng, 1, 0)
    img_green_hue = cv.inRange(img_hsv, (32, 38, 70), (85, 255, 200))

    img_comb_hue = cv.addWeighted(img_green_hue, 1, img_red_hue, 1, 0)

    # Postprocessing: blur the output so that the silver lines are ignored
    img_comb_hue = cv.GaussianBlur(img_comb_hue, (5, 5), cv.BORDER_DEFAULT)

    return img_comb_hue


def get_perspective_mat(thresh):
    """Gets the"""
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = []
    max_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt

    ellipse = cv.fitEllipse(largest_contour)

    e_center = ellipse[0]
    e_size = ellipse[1]

    top, right, bottom, left = ([e_center[0], e_center[1] - e_size[1] / 2], [e_center[0] + e_size[0] / 2, e_center[1]],
                                [e_center[0], e_center[1] + e_size[1] / 2], [e_center[0] - e_size[0] / 2, e_center[1]])

    source_pts = np.float32([top, right,
                             bottom, left])
    dest_pts = np.float32([[540, 0], [1080, 540],
                           [540, 1080], [0, 540]])

    return cv.getPerspectiveTransform(source_pts, dest_pts)


def perspective_correction(img_cropped, thresh):
    """Adjusts the perspective of img_cropped so that it is square on"""
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
    dest_pts = np.float32([[540, 0], [1080, 540],
                           [540, 1080], [0, 540]])

    perspective_mat = cv.getPerspectiveTransform(source_pts, dest_pts)
    return cv.warpPerspective(img_cropped, perspective_mat, (1080, 1080))


def center_bull(img):
    """Returns img with the center of the bullseye at the center of the image"""
    # Select only the center of the image, zoom by factor of 4
    h, w = img.shape[:2]
    mid_y, mid_x = (h//2, w//2)
    offset_y, offset_x = (mid_y // 4, mid_x // 4)

    img_mid = img[mid_y-offset_y:mid_y+offset_y, mid_x-offset_x:mid_x+offset_x].copy()

    # Get a mask of only 'red' pixels
    img_mid_hsv = cv.cvtColor(img_mid, cv.COLOR_BGR2HSV)

    # lower_red_hue_rng = cv.inRange(img_mid_hsv, (0, 100, 100), (10, 255, 255))
    # upper_red_hug_rng = cv.inRange(img_mid_hsv, (160, 100, 100), (179, 255, 255))
    #
    # img_red_hue = cv.addWeighted(lower_red_hue_rng, 1, upper_red_hug_rng, 1, 0)
    # img_red_hue = cv.GaussianBlur(img_red_hue, (5, 5), cv.BORDER_DEFAULT)

    img_green_hue = cv.inRange(img_mid_hsv, (32, 38, 70), (85, 255, 200))
    img_green_hue = cv.GaussianBlur(img_green_hue, (5, 5), cv.BORDER_DEFAULT)

    # Detect the circles in this mask
    circles = cv.HoughCircles(img_green_hue, cv.HOUGH_GRADIENT, 1, w // 4, param1=50, param2=20)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Select the largest circle as the bull
        max_radius = 0
        bull_center = (0, 0)
        for c in circles[0, :]:
            if c[2] > max_radius:
                max_radius = c[2]
                bull_center = (c[0], c[1])
            # img_mid = cv.circle(img_mid, (c[0], c[1]), c[2], GREEN, 3)

        true_center = (mid_x-offset_x+bull_center[0], mid_y-offset_y+bull_center[1])
        return true_center
    #     img = cv.circle(img, true_center, max_radius, GREEN, 2)
    # else:
    #     # Need to handle no bull detected
    #     pass
    #
    # return img


def draw_polar_line(img, src, theta):
    r = img.shape[0] // 2

    dst = (int(r * 1/math.cos(theta)) + src[0], int(r * 1/math.sin(theta)) + src[1])

    return cv.line(img, src, dst, GREEN, 2)


img = cv.imread('IMG_1010.jpg')  # cv.IMREAD_GRAYSCALE

# If source image is very noisy, use cv.medianBlur()

# img_cropped = crop_image(img)

# When adjusting thresh_param, need to retry until the bounding box of the fitted ellipse fits within (0, 0), (w, h)
# thresh = fill_holes(img_grey, 100)

# Calculate the threshold of the dart face
# img_face = get_face(img_cropped)

img_face = get_face(img)

# Correct the perspective so that the face is square on
# img_warped = perspective_correction(img_cropped, img_face)
img_warped = perspective_correction(img, img_face)
height, width = img_warped.shape[:2]
# Next function: detect bull/semibull
bull = center_bull(img_warped)

# TODO: fix rotation
for i in range(20):
    theta = (i + 0.5) * (math.pi / 10)
    # (width // 2, height // 2)
    img_warped = draw_polar_line(img_warped, bull, theta)


gui.showImage(img)
gui.showImage(img_face)
# gui.showImage(img_cropped)
gui.showImage(img_warped)
# gui.showImage(img_bull)
