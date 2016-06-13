import time
import tensorflow as tf
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from Loggable import Loggable

class VPClassifier(Loggable):
    """ Wraps the functionalities of the frozen tensor flow neural network 
        stored int self.filename.
    """ 
    def __init__(self, logger, filename):
        Loggable.__init__(self,logger,VPClassifier.__name__)
        self.filename = filename
    
    def classify(self, path, resize_height, resize_width):
        """ Resizes the passed image to indicated dimensions and estimates its
            VP using the graph stored self.filename.
        """ 
        self.info("Manually classifying the image in " + str(path))
        # Load freezed graph from file.
        graph_def = tf.GraphDef()
        with open(self.filename, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        predictions = []
        with tf.Session() as sess:
            # Load output node to use for predictions.
            output_node_processed = sess.graph.get_tensor_by_name('import/output_processed:0')
            # Iterate files from directory.
            start_time = time.time()
            # Read image 
            img = cv.imread(path, 1)
            # Process image that will be evaluated by the model.
            img_pred = imresize(img, [resize_height, resize_width], 'bilinear')
            img_pred = img_pred.astype(np.float32)
            img_pred = np.multiply(img_pred, 1.0 / 256.0)
            img_pred = img_pred.flatten()
            # Compute prediction point.
            predictions = output_node_processed.eval(
                feed_dict = {
                    'import/input_images:0': img_pred,
                    'import/keep_prob:0': 1.0
                }
            )
            predictions = np.round(predictions).astype(int)
            self.info('Predicted Point Processed: (' + str(int(round(predictions[0][0]))) + ', ' + str(int(round(predictions[0][1]))) + ')')
        return predictions
   
    def show(self, test_images, test_labels):
        """Runs the classifier with the passed test data and prints 
           various results.
        """
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with open(self.filename, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def)

            with tf.Session() as sess:

                # Original Predictions
                output_node = sess.graph.get_tensor_by_name('import/output:0')
                predictions = output_node.eval(
                    feed_dict = {
                        'import/input_images:0': test_images,
                        'import/keep_prob:0': 1.0
                    }
                )
                print("Test Set Example Predictions (0-10):")
                print(predictions[0:10])
                print("")

                # Processed Predictions
                output_node_processed = sess.graph.get_tensor_by_name('import/output_processed:0')
                predictions = output_node_processed.eval(
                    feed_dict = {
                        'import/input_images:0': test_images,
                        'import/keep_prob:0': 1.0
                    }
                )
                print("Test Set Example Predictions Processed (0-10):")
                print(predictions[0:10])
                print(" ")

                # Mean loss euclidian
                loss_euc = sess.graph.get_tensor_by_name('import/loss_euclidian:0')
                mean_loss = loss_euc.eval(
                    feed_dict = {
                        'import/input_images:0': test_images,
                        'import/input_labels:0': test_labels,
                        'import/keep_prob:0': 1.0
                    }
                )
                print('Test Set Accuracy (euclidian): ' + str(mean_loss))
                print("")

                # Mean loss angle
                loss_ang = sess.graph.get_tensor_by_name('import/loss_angular:0')
                mean_loss = loss_ang.eval(
                    feed_dict = {
                        'import/input_images:0': test_images,
                        'import/input_labels:0': test_labels,
                        'import/keep_prob:0': 1.0
                    }
                )
                accuracy = np.arccos(1.0 - mean_loss) * (180.0 / np.pi)
                print('Test Set Accuracy (angle degrees): ' + str(accuracy))
                print("")
                
                return accuracy
