#!/usr/bin/python2

import os
from PIL import Image, ImageFilter
import numpy as np
from Loggable import Loggable

class ImageReader(Loggable):
    def __init__(self, logger):
        Loggable.__init__(self,logger,ImageReader.__name__)
    
    def read_single_png(self, filename, resize=True, newWidth=180, newHeight=120):
        images = []
        im = Image.open(filename)
        
        if resize:
            im = im.resize((newWidth,newHeight))
        
        im = np.asarray(im, np.uint8)
        # get only images name, not path
        # image_name = filename.split('/')[-1].split('.')[0]
        # images.append([int(image_name), im])
        images.append([0, im])
        images = sorted(images, key=lambda image: image[0])
        images_only = [np.asarray(image[1], np.uint8) for image in images]  
        images_only = np.array(images_only)
        return images_only
