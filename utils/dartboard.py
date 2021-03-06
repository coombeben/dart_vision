from datetime import datetime

from .vision import get_face, get_perspective_mat


class Dartboard:
    def __init__(self):
        self.calibrated = False
        self.last_calibrated = None
        self.perspective_matrix = None

    def update_perspective_mat(self, img):
        # Optional: use vision.crop_image() to limit search area first
        img_face = get_face(img)
        self.perspective_matrix = get_perspective_mat(img_face)
        self.last_calibrated = datetime.now()
        self.calibrated = True

    def get_points(self, img):
        pass
