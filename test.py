import cv2 as cv
import numpy as np

import consts
import utils.vision as vision
import utils.gui as gui


frame_1 = cv.imread('Frame_1.jpg')
frame_2 = cv.imread('Frame_2.jpg')

pers_max = vision.get_perspective_mat(vision.get_face(frame_1))
frame_1_t = cv.warpPerspective(frame_1, pers_max, (1080, 1080))
frame_2_t = cv.warpPerspective(frame_2, pers_max, (1080, 1080))

back_sub = cv.createBackgroundSubtractorMOG2()
_ = back_sub.apply(frame_1_t)
foregound_mask = back_sub.apply(frame_2_t)

gui.showImage(foregound_mask)

th = vision.frame_diff(frame_1_t, frame_2_t)
gui.showImage(th)

conts, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
largest_cont, _ = vision.get_largest_contor(conts)
frame_conts = cv.drawContours(frame_2_t, [largest_cont], 0, consts.GREEN, 3)
gui.showImage(frame_conts)

# gui.showImage(img)
# gui.showImage(img_face)
# # gui.showImage(img_cropped)
# gui.showImage(img_warped)
# gui.showImage(img_bull)

# img = cv.imread('IMG_1010.jpg')
# # img_cropped = crop_image(img)
#
# # Calculate the threshold of the dart face
# # img_face = get_face(img_cropped)
# img_face = vision.get_face(img)
#
# # Correct the perspective so that the face is square on
# # img_warped = perspective_correction(img_cropped, img_face)
# img_warped = vision.perspective_correction(img, img_face)
# height, width = img_warped.shape[:2]
# # Next function: detect bull/semibull
# bull = vision.center_bull(img_warped)
#
# # TODO: fix rotation
# for i in range(20):
#     theta = (i + 0.5) * (np.pi / 10)
#     # (width // 2, height // 2)
#     img_warped = gui.draw_polar_line(img_warped, bull, theta)


