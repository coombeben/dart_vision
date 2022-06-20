import cv2 as cv
import numpy as np

import consts
import utils.gui as gui
import utils.vision as vision


class Detector:
    """Dartboard detector class. Used to adjust the perspective of and center the dartboard"""
    def __init__(self):
        self.aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv.aruco.DetectorParameters_create()

        self.PAD_A = 5

        self.perspective_mat_a = None
        self.perspective_mat_b = None
        self.perspective_mat = None

    def correct_image(self, img, debug=False):
        recalculate = False
        # First step, required for initial setup only
        if not self._is_valid_a(img):
            self._get_perspective_mat_a(img, debug)
            self._get_perspective_mat_b(img, debug)
            recalculate = True
        if not self._is_valid_b(img):
            self._get_perspective_mat_b(img, debug)
            recalculate = True

        # If either perspective matrix could not be calculated from the scene, return nothing
        if self.perspective_mat_a is None or self.perspective_mat_b is None:
            return None

        # Only need to calculate this again if either matrix was redefined
        if recalculate:
            self.perspective_mat = self.perspective_mat_b.dot(self.perspective_mat_a)

        return cv.warpPerspective(img, self.perspective_mat, consts.TRANSFORM)

    def recalculate_perspective(self, img, debug=False):
        """Used to manually recalculate the perspective matrices"""
        self._get_perspective_mat_a(img, debug)
        self._get_perspective_mat_b(img, debug)

    def _is_valid_a(self, img) -> bool:
        """Checks that all 4 aruco tags can still be detected using current perspective matrix
        If this returns True then perspective_mat_a is still valid"""
        if self.perspective_mat_a is None:
            return False
        else:
            img_a = cv.warpPerspective(img, self.perspective_mat_a, consts.TRANSFORM)
            _, ids, _ = cv.aruco.detectMarkers(img_a, self.aruco_dict, parameters=self.aruco_params)
            if ids is None:
                return False
        return ids.shape[0] == 4

    def _is_valid_b(self, img):
        """Performs some check to see if the dartboard has moved.
        Maybe compare green pixels in hsv and check difference is below threshold?"""
        return self.perspective_mat_b is None

    def _get_perspective_mat_a(self, img, debug=False):
        """Corrects the perspective to be square on with the board. Required if the camera moves.
        Will trigger the recalculation of mat_b too"""
        corners, ids, _ = cv.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)
        ids = ids.flatten()

        if debug:
            debug_img = cv.aruco.drawDetectedMarkers(img, corners, ids)
            gui.showImage(debug_img)

        if len(ids) < 4:
            self.perspective_mat_a = None
            return

        source_pts = np.zeros((4, 2), np.float32)
        tag_position = [2, 3, 0, 1]  # Indicates which corner of the marker to take for the ith marker id

        for (marker_corner_arr, marker_id) in zip(corners, ids):
            marker_corners = marker_corner_arr.reshape((4, 2)).astype('float32')

            source_pts[marker_id] = marker_corners[tag_position[marker_id]]

        # Pad the corners slightly so that we can still search for the aruco tags in later frames
        dest_pts = np.array([[self.PAD_A, self.PAD_A],
                             [consts.TRANSFORM_X - self.PAD_A, self.PAD_A],
                             [consts.TRANSFORM_X - self.PAD_A, consts.TRANSFORM_Y - self.PAD_A],
                             [self.PAD_A, consts.TRANSFORM_Y - self.PAD_A]], np.float32)

        self.perspective_mat_a = cv.getPerspectiveTransform(source_pts, dest_pts)

    def _get_perspective_mat_b(self, img, debug=False):
        """Centers the dartboard and makes final adjustments to scale. Required if the dartboard moves
        Should only be called if perspective_mat_a is not None"""
        if self.perspective_mat_a is None:
            return
        img = cv.warpPerspective(img, self.perspective_mat_a, consts.TRANSFORM)

        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        lower_red_hue_rng = cv.inRange(img_hsv, (0, 100, 100), (10, 255, 255))
        upper_red_hug_rng = cv.inRange(img_hsv, (160, 100, 100), (179, 255, 255))
        img_red_hue = cv.addWeighted(lower_red_hue_rng, 1, upper_red_hug_rng, 1, 0)
        img_green_hue = cv.inRange(img_hsv, (32, 38, 70), (85, 255, 200))
        img_comb_hue = cv.addWeighted(img_green_hue, 1, img_red_hue, 1, 0)

        # Postprocessing: blur the output so that the silver lines are ignored
        img_comb_hue = cv.GaussianBlur(img_comb_hue, (9, 9), cv.BORDER_DEFAULT)

        if debug:
            gui.showImage(img_comb_hue)

        contours, _ = cv.findContours(img_comb_hue, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        largest_contour, max_area = vision.get_largest_contour(contours)

        e_center, e_size, e_angle = cv.fitEllipse(largest_contour)
        e_eccentricity = np.sqrt(1 - e_size[0] ** 2 / e_size[1] ** 2)

        if debug:
            print(f'Area: {max_area}, Eccentricity {e_eccentricity}')
            debug_img = cv.ellipse(img, (e_center, e_size, e_angle), consts.GREEN, 3)
            gui.showImage(debug_img)

        # Opencv normalises ellipses so that h > w. Here we ensure that h refers to y-height and w refers to x-width
        max_x, max_y = largest_contour.max(axis=0).flatten()
        semi_major, semi_minor = e_size
        if max_x < max_y:
            h, w = (semi_major, semi_minor)
        else:
            h, w = (semi_minor, semi_major)

        perspective_mat_b = None
        if e_eccentricity < consts.MAX_ECCENTRICITY and max_area > consts.MIN_DARTBOARD_AREA:
            source_pts = np.float32([[e_center[0], e_center[1] - h / 2],
                                     [e_center[0] + w / 2, e_center[1]],
                                     [e_center[0], e_center[1] + h / 2],
                                     [e_center[0] - w / 2, e_center[1]]])
            dest_pts = np.float32([[consts.TRANSFORM_X // 2, consts.PAD_SCOREZONE],
                                   [consts.TRANSFORM_X - consts.PAD_SCOREZONE, consts.TRANSFORM_Y // 2],
                                   [consts.TRANSFORM_X // 2, consts.TRANSFORM_Y - consts.PAD_SCOREZONE],
                                   [consts.PAD_SCOREZONE, consts.TRANSFORM_Y // 2]])

            perspective_mat_b = cv.getPerspectiveTransform(source_pts, dest_pts)

        self.perspective_mat_b = perspective_mat_b
