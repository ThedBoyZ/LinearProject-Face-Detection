

import cv2
import numpy as np
import matplotlib.pyplot as plt

def scale_up(scale_var, img):
    scale_percent = scale_var  # percent of original size
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Load your image
img = cv2.imread('face.jpg')
new_img = scale_up(1.5, img)
cv2.imshow('Original', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
