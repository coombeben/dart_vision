import cv2 as cv
import numpy as np
import os

from consts import RESOLUTION_X, RESOLUTION_Y


# noinspection PyTypeChecker
class Calibrator:
    def __init__(self, target_obj_points=10):
        self.CHESSBOARD = (9, 6)
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.target_obj_points = target_obj_points
        self.obj_points = []
        self.img_points = []
        self.objp = np.zeros((1, self.CHESSBOARD[0] * self.CHESSBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHESSBOARD[0], 0:self.CHESSBOARD[1]].T.reshape(-1, 2)
        self.prev_img_shape = None
        self.obj_points_built = False
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def build_obj_points(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Optional other flags: + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv.findChessboardCorners(frame, self.CHESSBOARD, cv.CALIB_CB_ADAPTIVE_THRESH)

        if ret:
            refined_corners = cv.cornerSubPix(frame, corners, (11, 11), (-1, -1), self.criteria)

            self.obj_points.append(self.objp)
            self.img_points.append(refined_corners)

            self.obj_points_built = len(self.obj_points) == self.target_obj_points

    def calibrate_camera(self):
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.obj_points, self.img_points,
                                                          (RESOLUTION_X, RESOLUTION_Y), None, None)
        self.ret = ret
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        return ret, mtx, dist, rvecs, tvecs

    def get_calibration(self):
        return self.ret, self.mtx, self.dist, self.rvecs, self.tvecs

    def export_calibration(self):
        if not any([x is None for x in [self.ret, self.mtx, self.dist, self.rvecs, self.tvecs]]):
            if os.path.exists('calib.npz'):
                os.remove('calib.npz')

            # with open('calib.npz', 'wb') as f:
            #     np.save(f, self.ret)
            #     np.save(f, self.mtx)
            #     np.save(f, self.dist)
            #     np.save(f, self.rvecs)
            #     np.save(f, self.tvecs)
            with open('calib.npz', 'wb') as f:
                np.savez(f, ret=self.ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
        else:
            print('No data to save')

    def import_calibration(self):
        if os.path.exists('calib.npz'):
            with np.load('calib.npz') as calib_dict:
                self.ret = calib_dict['ret']
                self.mtx = calib_dict['mtx']
                self.dist = calib_dict['dist']
                self.rvecs = calib_dict['rvecs']
                self.tvecs = calib_dict['tvecs']
