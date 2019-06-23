from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import cv2

def search_largest_region(image):
    labeling = measure.label(image)
    regions = measure.regionprops(labeling)

    largest_region = None
    area_max = 0.
    for region in regions:
        if region.area > area_max:
            area_max = region.area
            largest_region = region

    return largest_region


def generate_largest_region(image):
    region = search_largest_region(image)
    bin_image = np.zeros_like(image)
    for coord in region.coords:
        bin_image[coord[0], coord[1]] = 1
    return bin_image

def fillHole(im_in):
    im_floodfill = im_in.copy().astype(np.uint8)
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = (im_in | im_floodfill_inv)/255

    return im_out.astype(int)

m = loadmat("./xxxxxxx.mat")
q = generate_largest_region(m["pred"])
im_out = fillHole(q)
plt.imshow(im_out)
plt.show()