import cv2 as cv
import numpy as np
from numpy.linalg import norm

import consts
import utils.gui as gui
# If cv2.imshow is not used, should use opencv-python-headless


def get_largest_contour(cnts):
    largest_contour = []
    max_area = 0
    for cnt in cnts:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt
    return largest_contour, max_area


def get_smallest_contour(cnts):
    smallest_contour = []
    min_area = consts.RESOLUTION_X * consts.RESOLUTION_Y
    for cnt in cnts:
        area = cv.contourArea(cnt)
        if area < min_area:
            min_area = area
            smallest_contour = cnt
    return smallest_contour, min_area


def get_min_max_contours(cnts):
    largest_contour = []
    smallest_contour = []
    max_area = 0
    min_area = consts.RESOLUTION_X * consts.RESOLUTION_Y
    for cnt in cnts:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt
        if area < min_area:
            min_area = area
            smallest_contour = cnt
    return largest_contour, max_area, smallest_contour, min_area


# def get_angle(pt_a, pt_b):
#     return np.arccos(np.dot(pt_a, pt_b) / (norm(pt_a) * norm(pt_a)))





def fill_holes(thresh):
    """Returns a threshold image of the board"""
    # _, img_th = cv.threshold(img, thresh_param, 255, cv.THRESH_BINARY_INV)
    img_floodfill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(img_floodfill, mask, (0, 0), 255)
    img_floodfill_inv = cv.bitwise_not(img_floodfill)
    return thresh | img_floodfill_inv


def get_face(img):
    """Returns the play face of the dartboard"""
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Find a threshold of only consts.GREEN and red shapes
    lower_red_hue_rng = cv.inRange(img_hsv, (0, 100, 100), (10, 255, 255))
    upper_red_hug_rng = cv.inRange(img_hsv, (160, 100, 100), (179, 255, 255))
    img_red_hue = cv.addWeighted(lower_red_hue_rng, 1, upper_red_hug_rng, 1, 0)
    img_green_hue = cv.inRange(img_hsv, (32, 38, 70), (85, 255, 200))

    img_comb_hue = cv.addWeighted(img_green_hue, 1, img_red_hue, 1, 0)

    # Postprocessing: blur the output so that the silver lines are ignored
    img_comb_hue = cv.GaussianBlur(img_comb_hue, (7, 7), cv.BORDER_DEFAULT)

    return img_comb_hue


def get_perspective_mat(thresh, debug=False):
    """Gets the"""
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour, max_area = get_largest_contour(contours)

    e_center, e_size, _ = cv.fitEllipse(largest_contour)

    # Validate the found ellipse. If the ellipse is too eccentric, reject the found ellipse
    e_eccentricity = np.sqrt(1 - e_size[0] ** 2 / e_size[1] ** 2)
    if debug:
        print(f'Area: {max_area}, Eccentricity: {e_eccentricity}')

    perspective_mat = None
    if e_eccentricity < consts.MAX_ECCENTRICITY and max_area > consts.MIN_DARTBOARD_AREA:
        top, right, bottom, left = ([e_center[0], e_center[1] - e_size[1] / 2],
                                    [e_center[0] + e_size[0] / 2, e_center[1]],
                                    [e_center[0], e_center[1] + e_size[1] / 2],
                                    [e_center[0] - e_size[0] / 2, e_center[1]])

        source_pts = np.float32([top, right, bottom, left])
        dest_pts = np.float32([[consts.TRANSFORM_X // 2, consts.PAD_SCOREZONE],
                               [consts.TRANSFORM_X - consts.PAD_SCOREZONE, consts.TRANSFORM_Y // 2],
                               [consts.TRANSFORM_X // 2, consts.TRANSFORM_Y - consts.PAD_SCOREZONE],
                               [consts.PAD_SCOREZONE, consts.TRANSFORM_Y // 2]])

        perspective_mat = cv.getPerspectiveTransform(source_pts, dest_pts)

    return perspective_mat


# noinspection DuplicatedCode
def get_homography_mat(img, debug=False):
    """Gets the homography matrix to adjust the perspective"""
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Find a threshold of only consts.GREEN and red shapes
    lower_red_hue_rng = cv.inRange(img_hsv, (0, 100, 100), (10, 255, 255))
    upper_red_hug_rng = cv.inRange(img_hsv, (160, 100, 100), (179, 255, 255))
    img_red_hue = cv.addWeighted(lower_red_hue_rng, 1, upper_red_hug_rng, 1, 0)
    img_green_hue = cv.inRange(img_hsv, (32, 38, 70), (85, 255, 200))

    img_comb_hue = cv.addWeighted(img_green_hue, 1, img_red_hue, 1, 0)
    # Postprocessing: blur the output so that the silver lines are ignored
    img_comb_hue = cv.GaussianBlur(img_comb_hue, (7, 7), cv.BORDER_DEFAULT)

    _, red_green_thresh = cv.threshold(img_comb_hue, 127, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(red_green_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour, max_area = get_largest_contour(contours)

    max_ellipse = cv.fitEllipse(largest_contour)
    e_center, e_size, e_angle = max_ellipse

    mask = np.zeros(img.shape[:2], np.uint8)
    mask = cv.ellipse(mask, max_ellipse, (255, 255, 255), -1)

    masked_thresh = cv.bitwise_and(red_green_thresh, red_green_thresh, mask=mask)

    if debug:
        gui.showImage(masked_thresh)

    # Maybe don't need to recalculate conts, can just filter existing based on proximity to e_center
    interior_contours, _ = cv.findContours(masked_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    smallest_contour, min_area = get_smallest_contour(interior_contours)

    if debug:
        debug_img = cv.drawContours(img, [smallest_contour], 0, consts.BLUE, 3)
        debug_img = cv.drawContours(debug_img, [largest_contour], 0, consts.GREEN, 3)
        gui.showImage(debug_img)

    min_ellipse = cv.fitEllipse(smallest_contour)
    true_center = min_ellipse[0]

    print(np.sqrt((e_center[0]-true_center[0])**2+(e_center[1]-true_center[1])**2))

    # Validate the found ellipse. If the ellipse is too eccentric, reject the found ellipse
    e_eccentricity = np.sqrt(1 - e_size[0] ** 2 / e_size[1] ** 2)
    if debug:
        print(f'Area: {max_area}, Eccentricity: {e_eccentricity}')

    homography_mat = None
    for i in range(4):
        i

    # homography_mat = cv.findHomography(source_pts, dest_pts)

    return homography_mat


def center_bull(img):
    """Returns img with the center of the bullseye at the center of the image"""
    # Select only the center of the image, zoom by factor of 4
    h, w = img.shape[:2]
    mid_y, mid_x = (h//2, w//2)
    offset_y, offset_x = (mid_y // 4, mid_x // 4)

    img_mid = img[mid_y-offset_y:mid_y+offset_y, mid_x-offset_x:mid_x+offset_x].copy()

    # Get a mask of only 'green' pixels
    img_mid_hsv = cv.cvtColor(img_mid, cv.COLOR_BGR2HSV)

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
            # img_mid = cv.circle(img_mid, (c[0], c[1]), c[2], consts.GREEN, 3)

        true_center = (mid_x-offset_x+bull_center[0], mid_y-offset_y+bull_center[1])
        return true_center
    #     img = cv.circle(img, true_center, max_radius, consts.GREEN, 2)
    # else:
    #     # Need to handle no bull detected
    #     pass
    #
    # return img


def frame_diff(img_a, img_b):
    img_a_grey = cv.cvtColor(img_a, cv.COLOR_BGR2GRAY)
    img_b_grey = cv.cvtColor(img_b, cv.COLOR_BGR2GRAY)

    img_a_grey = cv.GaussianBlur(img_a_grey, (5, 5), cv.BORDER_DEFAULT)
    img_b_grey = cv.GaussianBlur(img_b_grey, (5, 5), cv.BORDER_DEFAULT)

    img_a_grey = cv.equalizeHist(img_a_grey)
    img_b_grey = cv.equalizeHist(img_b_grey)

    diff = cv.absdiff(img_a_grey, img_b_grey)

    diff = cv.GaussianBlur(diff, (consts.BLUR_KSIZE, consts.BLUR_KSIZE), cv.BORDER_DEFAULT)

    _, thresh = cv.threshold(diff, 70, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (consts.KERNEL_SIZE, consts.KERNEL_SIZE))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    return thresh


def filter_diff_noise(thresh, debug=False):
    # Remove shadows
    _, thresh = cv.threshold(thresh, 200, 255, cv.THRESH_BINARY)

    # Remove noise
    thresh = cv.GaussianBlur(thresh, (9, 9), cv.BORDER_DEFAULT)

    kernel = np.ones((9, 9), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    _, thresh = cv.threshold(thresh, 127, 255, cv.THRESH_BINARY)

    if debug:
        gui.showImage(thresh)

    return thresh


def get_blur(img, mask, debug=False):
    x, y, w, h = cv.boundingRect(mask)
    if w < 2 or h < 2:
        return 0

    img_cropped = img[y:y+h, x:x+w].copy()
    img_cropped = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    return cv.Laplacian(img_cropped, cv.CV_64F).var()


def get_arrow_point(cont, img=None, debug=False):
    """Returns the coordinates of the intersection of the dart and the board"""
    # Basic logic:
    # 1) fit a rectangle and line through the dart contour.
    # 2) for each side of the rectangle, check that it is somewhat perpendicular to the line
    # 3) if it is perpendicular, calculate the point at which the fitted line meets the rectangle edge
    # 4) for all intersection points calculated, select the one furthest from the center of mass of the contour

    center_of_mass = cont.mean(axis=0)[0]

    rect_center, rect_size, rect_angle = cv.minAreaRect(cont)

    box = cv.boxPoints((rect_center, rect_size, rect_angle))

    line = cv.fitLine(cont, cv.DIST_L2, 0, 0.01, 0.01)
    line = line.flatten()

    possible_intersect_points = []
    for i in range(4):
        # CV returns line as an array of [m_x, m_y, pt_x, pt_y] so we convert the box points into a similar form to solve
        box_line = np.array([box[i % 4][0] - box[(i + 1) % 4][0], box[i % 4][1] - box[(i + 1) % 4][1],
                             box[i % 4][0], box[i % 4][1]], np.float32)

        # Check that the gradients are somewhat tangential
        relative_angle = np.arccos(np.dot(line[0:2], box_line[0:2]) / (norm(line[0:2]) * norm(box_line[0:2])))
        if (np.pi/4 <= relative_angle <= 3*np.pi/4) or (5*np.pi/4 <= relative_angle <= 7*np.pi/4):
            a = np.array([[line[0], -box_line[0]], [line[1], -box_line[1]]], np.float32)
            b = np.array([box_line[2] - line[2], box_line[3] - line[3]], np.float32)
            r = np.linalg.solve(a, b)

            intersect_point = np.array((line[2] + r[0] * line[0], line[3] + r[0] * line[1]), np.float32)

            possible_intersect_points.append(intersect_point)

    max_dist = 0
    intersect_point = None
    for ip in possible_intersect_points:
        if norm(ip - center_of_mass) > max_dist:
            intersect_point = ip

    if debug:
        debug_img = cv.polylines(img, [np.intp(box)], True, consts.GREEN)
        debug_img = cv.line(debug_img, np.intp(line[2:4]), np.intp(intersect_point), consts.GREEN, 3)
        debug_img = cv.circle(debug_img, np.intp(intersect_point), 3, consts.RED, 3)
        gui.showImage(debug_img)

    return intersect_point
