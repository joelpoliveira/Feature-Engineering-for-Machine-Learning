import os
import cv2
from matplotlib import image as img

def read_image(PATH, filename, width=256, height=256):
    shape = (width, height)
    return cv2.resize(
        img.imread(PATH + filename),
        shape
    )



