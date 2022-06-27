import cv2 as cv
import numpy as np

import consts
import utils.gui as gui

KERNEL_WIDTH = 21
KERNEL_THICKNESS = 3
KERNEL_ROT = 45
kernel = np.zeros((KERNEL_WIDTH, KERNEL_WIDTH), np.uint8)
kernel = cv.line(kernel, (0, 0), (KERNEL_WIDTH, KERNEL_WIDTH), 1, KERNEL_THICKNESS)

# Parameters used to define acceptable bounds for the radius.
DOUBLE_OUTER = consts.TRANSFORM_X / 2 - consts.PAD_SCOREZONE
DOUBLE_INNER = DOUBLE_OUTER * (159 / 170)
TREBLE_OUTER = DOUBLE_OUTER * (106 / 170)
TREBLE_INNER = DOUBLE_OUTER * (97 / 170)
SEMI_BULL = DOUBLE_OUTER * (16 / 170)
BULL = DOUBLE_OUTER * (6.35 / 170)


def _get_largest_contour(conts):
    largest_contour = []
    max_area = 0
    for cont in conts:
        area = cv.contourArea(cont)
        if area > max_area:
            max_area = area
            largest_contour = cont
    return largest_contour, max_area


def angle_between(a, b=(0, -1)) -> float:
    """Returns clockwise angle from vector b to vector a"""
    if a[0] < 0:
        return 2 * np.pi - np.arccos((np.dot(a, b)) / (cv.norm(a) * cv.norm(b)))
    else:
        return np.arccos((np.dot(a, b)) / (cv.norm(a)*cv.norm(b)))


# def find_dart(img_a, img_b, debug=False):
#     a_grey = cv.cvtColor(img_a, cv.COLOR_BGR2GRAY)
#     b_grey = cv.cvtColor(img_b, cv.COLOR_BGR2GRAY)
#
#     diff = get_ssim(a_grey, b_grey)
#     diff = (diff * 255).astype('uint8')
#
#     _, thresh = cv.threshold(diff, 150, 255, cv.THRESH_BINARY_INV)  # | cv.THRESH_OTSU
#     contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     max_cont, cont_area = _get_largest_contour(contours)
#
#     intersect_point = None
#     if cont_area > consts.MIN_DART_AREA:
#         if debug:
#             gui.showImage(diff)
#             debug_img = img_a.copy()
#             debug_img = cv.drawContours(debug_img, [max_cont], 0, consts.GREEN, 3)
#             debug_img = cv.putText(debug_img, f'{cont_area}', (0, 1080), cv.FONT_HERSHEY_PLAIN,
#                                    3, (255, 255, 255), 3)
#             gui.showImage(debug_img)
#
#         intersect_point = _get_impact_point(max_cont, img_a, debug=debug)
#     return intersect_point


def find_dart_b(foreground_mask, debug_img=None, debug=False):
    foreground_mask = cv.GaussianBlur(foreground_mask, (9, 9), cv.BORDER_DEFAULT)
    _, thresh = cv.threshold(foreground_mask, 128, 255, cv.THRESH_BINARY)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_cont, cont_area = _get_largest_contour(contours)

    intersect_point = None
    if consts.MIN_DART_AREA < cont_area < consts.MAX_DART_AREA:
        if debug:
            gui.showImage(thresh)
            cont_img = debug_img.copy()
            cont_img = cv.drawContours(cont_img, [max_cont], 0, consts.GREEN, 3)
            gui.showImage(cont_img, f'Area: {cont_area}')

        intersect_point = _get_impact_point(max_cont, debug_img, debug=debug)
    return intersect_point


def get_points(intersect_point, img=None, debug=False) -> (int, int):
    assert not debug or not (img is None)
    """Given an intersection point, returns the corresponding score and whether it was a double"""
    pt_x, pt_y = intersect_point

    score_zones = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    radius = cv.norm((consts.TRANSFORM_X / 2 - pt_x, consts.TRANSFORM_Y / 2 - pt_y))
    theta = angle_between((pt_x - consts.TRANSFORM_X / 2, pt_y - consts.TRANSFORM_Y / 2))

    points = 0
    multiplier = 1

    if radius < SEMI_BULL:
        points = 25
        if radius < BULL:
            multiplier = 2
    elif radius < DOUBLE_OUTER:  # If dart landed in board:
        points = score_zones[int(np.floor(((theta + np.pi/20) % (2 * np.pi)) * (10 / np.pi)))]
        if DOUBLE_INNER <= radius < DOUBLE_OUTER:
            multiplier = 2
        elif TREBLE_INNER <= radius < TREBLE_OUTER:
            multiplier = 3

    if debug:
        debug_img = img.copy()
        debug_img = cv.circle(debug_img, np.intp(intersect_point), 3, consts.BLUE, 3)
        gui.showImage(debug_img, f'Points: {points*multiplier}')

    return points, multiplier


def _get_impact_point(cont, img=None, debug=False):
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
        relative_angle = np.arccos(np.dot(line[0:2], box_line[0:2]) / (np.linalg.norm(line[0:2]) * np.linalg.norm(box_line[0:2])))
        if (np.pi / 4 <= relative_angle <= 3 * np.pi / 4) or (5 * np.pi / 4 <= relative_angle <= 7 * np.pi / 4):
            a = np.array([[line[0], -box_line[0]], [line[1], -box_line[1]]], np.float32)
            b = np.array([box_line[2] - line[2], box_line[3] - line[3]], np.float32)
            r = np.linalg.solve(a, b)

            intersect_point = np.array((line[2] + r[0] * line[0], line[3] + r[0] * line[1]), np.float32)

            possible_intersect_points.append(intersect_point)

    max_dist = 0
    intersect_point = None
    for ip in possible_intersect_points:
        if np.linalg.norm(ip - center_of_mass) > max_dist:
            intersect_point = ip

    if debug:
        debug_img = cv.polylines(img.copy(), [np.intp(box)], True, consts.GREEN)
        debug_img = cv.line(debug_img, np.intp(line[2:4]), np.intp(intersect_point), consts.GREEN, 3)
        debug_img = cv.circle(debug_img, np.intp(intersect_point), 3, consts.RED, 3)
        gui.showImage(debug_img)

    return intersect_point
