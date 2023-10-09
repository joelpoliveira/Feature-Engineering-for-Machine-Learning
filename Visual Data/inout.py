import os
import cv2
from matplotlib import image as img
PATH = "./flowers/"

def get_filename(flower_type, idx = 0):
    flowers = list(filter(lambda filename: flower_type in filename, sorted(os.listdir(PATH)) ))
    return flowers[idx]


def read_image(filename, width=256, height=256):
    shape = (width, height)
    return cv2.resize(
        img.imread(PATH + filename),
        shape
    )


def get_image(type_idx=0, idx=0):
    return read_image(
        get_filename(flower_types[type_idx], idx)
    )

