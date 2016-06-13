#!/usr/bin/python2

import os
from PIL import Image, ImageFilter
import numpy as np
from Loggable import Loggable
from ImageReader import ImageReader

class Preprocessor(Loggable):
    """This script uses the trained classifier to prepare a database 
       of manipulated images with their estimated vanishing points.
    """
    def __init__(self, logger):
        Loggable.__init__(self,logger,Preprocessor.__name__)
        self.ir = ImageReader(logger)
    
    def read_single_png(self, path, resize=True, newWidth=180, newHeight=120):
        """Reads the single PNG image indicated by path and resizes it 
           according to the passed dimensions.
        """
        return self.ir.read_single_png(path, resize, newWidth, newHeight) 
    
    def __prepareActionDirectory(self, directory):
        do_action =  True if not os.path.exists(directory) else False
        if do_action:
            self.create_file_with_content(directory, '1')
            self.info('Iteration will include action in ' + directory)
        return do_action
    
    
    def distort_images(self, input_directory, output_directory, blur_flag_path, unsharpen_flag_path, contour_flag_path):
        """Distorts the PNG images in input_directory and saves the 
           results to output_directory. Returns true if new images
           were produced.
        """
        self.debug('Distorting images from ' + input_directory + ' to ' + output_directory)
        image = None
        result = None
        
        do_blur      = self.__prepareActionDirectory(blur_flag_path)   
        do_unsharpen = self.__prepareActionDirectory(unsharpen_flag_path) 
        do_contour   = self.__prepareActionDirectory(contour_flag_path) 
        
        for filename in os.listdir(input_directory):
            if filename.endswith(".png"):
                full_filename = input_directory + "/" + filename
                image = Image.open(full_filename)
                # Maybe produce different blur levels
                if do_blur:
                    self.__applyGaussianBlurLevels(image, filename, output_directory)
                # Maybe produce different unsharpen levels
                if do_blur:
                    self.__applyUnsharpMaskLevels(image, filename, output_directory)
                # Maybe produce contour
                if do_contour:
                    self.__applyContour(image, filename, output_directory)
        
        if do_blur:      self.create_file_with_content(blur_flag_path, "1")
        if do_unsharpen: self.create_file_with_content(unsharpen_flag_path, "1")
        if do_contour:   self.create_file_with_content(contour_flag_path, "1")
            
        return do_blur or do_unsharpen or do_contour
    
    def __createPath(self, output_directory, filename, infix, suffix):
        identifier = infix + str(suffix)
        product_filename = filename.replace(".png", identifier + ".png")
        return output_directory + product_filename
        
    def __applyContour(self, image, filename, output_directory):
        """ TODO 
        """
        res = None
        infix = '-contour='
        # Create manipulated image
        p_path = self.__createPath(output_directory, filename, infix, 'X')
        self.info('Contouring image ' + p_path)
                    
        product = image.filter(ImageFilter.CONTOUR)
        
        self.debug('Saving contoured image to' + p_path)
        product.save(p_path, "PNG")
        
    def __applyEdgeEnhance(self, image, filename, output_directory):
        """ TODO 
        """
        res = None
        infix = '-edgeEnhance='
        # Create manipulated image
        p_path = self.__createPath(output_directory, filename, infix, 'X')
        self.info('Edge enhancing image ' + p_path)
                    
        product = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        self.debug('Saving edge enhanced image to' + p_path)
        product.save(p_path, "PNG")
            
    def __applyUnsharpMaskLevels(self, image, filename, output_directory):
        """ TODO 
        """
        res = None
        infix = '-unsharp='
        for i in range(2):
            # Create manipulated image
            unsharp_level = 100 + (i+1)*100
            p_path = self.__createPath(output_directory, filename, infix, unsharp_level)
            self.info('Unsharpening image ' + p_path)
                        
            product = image.filter(ImageFilter.UnsharpMask(2, unsharp_level))
            
            self.debug('Saving unsharpened image to' + p_path)
            product.save(p_path, "PNG")
                
    def __applyGaussianBlurLevels(self, image, filename, output_directory):
        """ Applies 2 levels of Gaussian blur to the sourceImage,
            estimates its VP, and saves the blurred image as well as the
            calculated VP to file.
        """
        res = None
        infix = '-blur='
        for i in range(2):
            # Create manipulated image
            blur_level = (i+1)*5
            p_path = self.__createPath(output_directory, filename, infix, blur_level)
            self.info('Blurring image ' + p_path)
            
            product = image.filter(ImageFilter.GaussianBlur(blur_level))
            
            self.debug('Saving blurred image to' + p_path)
            product.save(p_path, "PNG")
    
    def create_file_with_content(self, path, content):
        """Creates (and possibly overwrites) a file in path with the
           given content.
        """
        f = open(path, 'w')
        f.write(str(content))
        f.close()
