from matplotlib import pyplot as plt
import cv2 as cv


def showImage(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), interpolation='nearest')
    plt.show()
