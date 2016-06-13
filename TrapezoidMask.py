#!/usr/bin/python2
import sys, os, glob
from operator import itemgetter
from PIL import Image

from ssim.ssimlib import SSIM
from ssim.utils import get_gaussian_kernel

from skimage.io import imread
from skimage.transform import resize
from skimage.measure import compare_mse
from skimage.morphology import label

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

def adjust_gamma(image, gamma=1.0):
    from pylab import array, uint8 
    # Parameters for manipulating image data
    phi = 1
    theta = 1
    maxIntensity = 255.0 # depends on dtype of image data
    newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
    newImage1 = array(newImage1,dtype=uint8)
    return newImage1

def connected_regions(img, name, blur_radius = 2, threshold = 50):
    import scipy
    from scipy import ndimage
    import matplotlib.pyplot as plt
    # smooth the image (to remove small objects)
    imgf = ndimage.gaussian_filter(img, blur_radius)
    # find connected components
    labeled, nr_objects = ndimage.label(imgf > threshold)
    #plt.imsave('/tmp/out.png', labeled)
    #plt.imshow(labeled)
    #plt.show()
    #plt.savefig(name)
    return labeled, nr_objects

def trapezoid_mask(trapezoid, blur_radius, threshold, path, out_dir = './tmp/'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    result = {}
    result["S"] = 0
    result["L"] = 0
    result["R"] = 0
#    failures = {}
#    failures["S"] = []
#    failures["L"] = []
#    failures["R"] = []
    #print("[TrapezoidMask] Processing " + path)
    img = cv2.imread(path,0)
    # Darken
    img = adjust_gamma(img)
    # Mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([trapezoid], dtype=np.int32)
    channel_count = 1  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    basename = os.path.basename(path)
    #cv2.imwrite(out_dir + basename.replace('.png', '.masked.png'), masked_image)
    # Find connected regions
    path = out_dir + basename.replace('.png', '.connected.png')
    labeled, n = connected_regions(masked_image, path, blur_radius, threshold)
    # Find extreme points on contour
    thresh = cv2.threshold(masked_image, 45, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.erode(thresh, None, iterations=2)
    #thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # Classify
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        if extLeft[1] < extRight[1]: # Left curve
            # print("[TrapezoidMask] " + str(n) + " parts and " + str(extLeft) + " < " + str(extRight) + ": this is a LEFT curve")
            result["L"] += 1
            #if basename[0] != 'L': failures["L"].append(basename)
        else: # Right curve
            #print("[TrapezoidMask] " + str(n) + " parts and " + str(extLeft) + " < " + str(extRight) + ": this is a RIGHT curve")
            result["R"] += 1
            #if basename[0] != 'R': failures["R"].append(basename)
    else:
        #print("[TrapezoidMask] " + str(n) + " parts and no contours: this is a STRAIGHT lane")
        result["S"] += 1
        #if basename[0] != 'S': failures["S"].append(basename)
    #print("[TrapezoidMask] Intermediate result: " + str(result))
    return result #, failures
