#!/usr/bin/python2

import logging, sys, os, datetime
from PIL import Image
#from scipy.spatial import Delaunay
import numpy as np
from classifier import imageRead
from classifier.VPClassifier import VPClassifier
#from evaluation.SSIMFinder import SSIMFinder
from evaluation.TrapezoidMask import *
from ImageReader import ImageReader
from Loggable import Loggable

# Equivalency classes
LEFT_CURVE_WIDTH  = 390
RIGHT_CURVE_WIDTH = 351 
STRAIGHT_WIDTH = 751 - LEFT_CURVE_WIDTH - RIGHT_CURVE_WIDTH # 10

LEFT_CURVE_MIN_X  = 0                                         # 0
LEFT_CURVE_MAX_X  = LEFT_CURVE_WIDTH - 1                      # 389
STRAIGHT_MIN_X    = LEFT_CURVE_MAX_X + 1                      # 390
STRAIGHT_MAX_X    = LEFT_CURVE_MAX_X + STRAIGHT_WIDTH         # 400
RIGHT_CURVE_MIN_X = STRAIGHT_MAX_X + 1                        # 401
RIGHT_CURVE_MAX_X = RIGHT_CURVE_MIN_X + RIGHT_CURVE_WIDTH - 1 # 751

# Deterministic classification properties
TRAPEZOID   = [(200,299), (LEFT_CURVE_WIDTH,75), (LEFT_CURVE_WIDTH+STRAIGHT_WIDTH,75), (590,299)]
BLUR_RADIUS = 3
THRESHOLD   = 25

class Evaluator(Loggable):
    def __init__(self, logger, graph, database_dir, start_reliability, reliability_delta):
        Loggable.__init__(self,logger,Evaluator.__name__)
        self.graph = graph
        self.classifier = VPClassifier(logger, graph) 
        self.ir = ImageReader(logger)
        #self.sf = SSIMFinder(logger) # finds similar images
        self.database_dir = database_dir
        self.start_reliability = 100
        self.reliability_delta = 1
        self.nn_reliability = self.start_reliability
        self.num_validated = 0
        
    def get_nn_reliability(self):
        return self.nn_reliability
                   
    def evaluate_nn_for_image(self, path, resize_height, resize_width):
        """ Checks whether the NN behaviors for the test image passed as path as
            expected. Returns +1 if it does, -1 otherwise. Uses the 
            TrapezoidMask matcher to identify the test image as a left curve,
            right curve or straight road.
        """
        self.info("Starting evaluate_nn_for_image with image " + path)
        # Read test image and predict VP using NN
        classificationStart = datetime.datetime.now()
        vp = self.classifier.classify(path, resize_height, resize_width)
        classificationDuration = (datetime.datetime.now() - classificationStart).total_seconds()   
        self.info("Classification for " + path + " took " + str(classificationDuration) + "s")        
        
        # Predict equivalency class using deterministic detection
        predictionStart = datetime.datetime.now()
        results = trapezoid_mask(TRAPEZOID,  BLUR_RADIUS, THRESHOLD, path)    
        predictionDuration = (datetime.datetime.now() - predictionStart).total_seconds()   
        self.info("Prediction for " + path + " took " + str(predictionDuration) + "s")        
        
        # Check whether estimated VP is in appropriate region
        x = vp[0][0]
        nn_delta = -1 * self.reliability_delta
        failure = None
        error = 0
        if results['S'] is 1: # We found a straight lane
            self.info("Image depicts a STRAIGHT lane")
            if STRAIGHT_MIN_X <= x <= STRAIGHT_MAX_X:
                nn_delta = self.reliability_delta
                self.num_validated += 1
            else:
                if x < STRAIGHT_MIN_X:
                    error = STRAIGHT_MIN_X - x
                else:
                    error = STRAIGHT_MAX_X - x
                self.info("ERROR: VP of image " + path + " is not in expected STRAIGHT lane range")
                failure = (path, 'STRAIGHT', x) # Image, Expectation, NN VP
        elif results['L'] is 1: # We found a left curve
            self.info("Image depicts a LEFT curve")
            if LEFT_CURVE_MIN_X <= x <= LEFT_CURVE_MAX_X:
                nn_delta = self.reliability_delta
                self.num_validated += 1
            else:
                if x < LEFT_CURVE_MIN_X:
                    error = LEFT_CURVE_MIN_X - x
                else: 
                    error = LEFT_CURVE_MAX_X - x
                self.info("ERROR: VP of image " + path + " is not in expected LEFT curve range")
                failure = (path, 'LEFT CURVE', x) # Image, Expectation, NN VP
        elif results['R'] is 1: # We found a right curve
            self.info("Image depicts a RIGHT curve")
            if RIGHT_CURVE_MIN_X <= x <= RIGHT_CURVE_MAX_X:
                nn_delta = self.reliability_delta
                self.num_validated += 1
            else:
                if x < RIGHT_CURVE_MIN_X:
                    error = RIGHT_CURVE_MIN_X - x
                else:
                    error = RIGHT_CURVE_MAX_X - x
                self.info("ERROR: VP of image " + path + " is not in expected RIGHT curve range")
                failure = (path, 'RIGHT CURVE', x) # Image, Expectation, NN VP
        else:
            self.info("Terminal error: no results retured")
            sys.exit(1)    
        self.nn_reliability += nn_delta    
        self.info("NN reliability is now: " + str(self.nn_reliability))
        return failure, error, classificationDuration, predictionDuration
