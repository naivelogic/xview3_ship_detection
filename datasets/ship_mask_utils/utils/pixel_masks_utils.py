from PIL import Image, ImageFilter
import numpy as np
from skimage.filters import threshold_otsu


def im_crop_around(img, xc, yc, w, h):    
    img_width, img_height = img.size  # Get dimensions
    left, right = xc - w / 2, xc + w / 2
    top, bottom = yc - h / 2, yc + h / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    new_box_coord = (left, top, right, bottom)
    return new_box_coord


def closeContour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_filter(array, thresh_override=None):
    im = Image.fromarray(array *255).convert("L")
    otsu_thresh = threshold_otsu(np.array(im)) -2 #-5

    if thresh_override==None:
        override = otsu_thresh
        thr = im.point(lambda p: p > override and 255)
    else:
        override = thresh_override
        thr = im.point(lambda p: p > override and 255)

    # Median filter to remove noise
    fil = thr.filter(ImageFilter.MedianFilter(3))
    return fil, override


def ray_tracing_method(x, y, poly):
    '''
       Determines if point x, y is inside polygon poly

       Source: "What's the fastest way of checking if a point is inside a polygon in python"
             at URL: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

    '''
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside
