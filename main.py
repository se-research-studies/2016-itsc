#!/usr/bin/python2

import logging, sys, os, datetime
from PIL import Image
from scipy.spatial import Delaunay

from classifier import imageRead
from classifier.VPClassifier import VPClassifier
from Preprocessor import Preprocessor
from Evaluator import Evaluator
from classifier.config import * # Important shared configuration
import cv2


def terminate(message):
    """ Exits with the passed message.
    """
    sys.stderr.write("[Main] " + message + "\n\n")
    sys.exit(1)

def usage():
    """ Prints usage information.
    """
    terminate("\nSYNOPSIS: main.py [<operation>], where\n\n" \
            + "  operation.. train, preprocess, evaluate, reset\n\n" \
            + "  <evaluate> requires a path to a test image directory"\
            + "  <evaluate> also produces a file ./data.csv with overall results")

def prepare_logger(log_file):
    """ Prepares a logger that logs messages > INFO to the standard 
        output and all other messages to the passed file.
    """
    logger = logging.getLogger('nn')
    logger.setLevel(logging.DEBUG)
    # Log DEBUG to file 
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # Log INFO to stream
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Format logging
    formatter = logging.Formatter('<%(levelname)s> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def check_file_existence(path):
    """ Checks whether the passed file or path exists. Aborts 
        otherwise.
    """    
    if not os.path.exists(path):
        terminate("ERROR: File or directory " + str(path) + " not found.")


def produce_result(filename,pred,proc):
    """ Produces a string representation of the passed predicted and 
        processed VPs.
    """
    print("[Main] Predicted VP " + str(pred))
    print("[Main] Predicted VP_processed " + str(proc))
    result = filename + ";" + str(pred[0]) + ";" + str(pred[1]) + ";" + str(proc[0]) + ";" + str(proc[1]) + "\n"
    return result

def train(data_root, models_root):
    """Trains the NN and saves the resulting classifier to the file
       ./classifier/models/graph_freezed.pb
    """
    os.system("python ./classifier/main.py --models " + models_root + " --data " + data_root)

def preprocess(logger, graph, training_dir, database_dir, blur_flag_path, unsharpen_flag_path, contour_flag_path, labels_path):
    """ Produces distorted images from the training data, classifies 
        their VPs, and persists the results.
    """
    if not os.path.isfile(graph):
        terminate("[Main] ERROR: Graph file " + str(graph) + " not found.")
        
    # Produce distorted images
#    preprocessor = Preprocessor(logger)
#    new_images = preprocessor.distort_images(training_dir, database_dir, blur_flag_path, unsharpen_flag_path, contour_flag_path)    
    
    # For each png in database_dir: classify and save result
#    if new_images:
#        logger.info("New images were produced. Saving labels to " + labels_path)
#        clf = VPClassifier(logger, graph)
#        result = ""
#        for filename in os.listdir(database_dir):
#            if filename.endswith(".png"):
#                path = database_dir + "/" + filename
#                logger.info("Predicting for image " + path)
#                image = preprocessor.read_single_png(path)
#                predictions = clf.classify(image)
#                pred = predictions[0][0]
#                proc = predictions[1][0]
#                result = result + produce_result(path, pred, proc)  
#        preprocessor.create_file_with_content(labels_path, result)
#    else:
#        logger.info("Did not produce new images. Nothing to classify.")

def evaluate(logger, graph, database_dir, test_data_dir, num_similar, treshold):
    """ Evaluates the classifier frozen in graph using the images of
        database_dir for comparison and the images of test_image_path
        for evaluation. The parameters num_similar nad treshold describe
        how many of the most similar images should be considered for
        construction of the convex hull and which images to consider 
        similar at all.
    """
    evaluator = Evaluator(logger, graph, database_dir, NN_START_RELIABILITY, NN_RELIABILITY_DELTA)

    iterations = 1
    nn_reliability = 100
    
    datasetStart = datetime.datetime.now()
    sum_classification_time = 0
    min_classification_time = 100000
    max_classification_time = 0
    sum_prediction_time = 0
    min_prediction_time = 100000
    max_prediction_time = 0
    sum_error = 0
    min_error = 100000
    max_error = 0
    
    # Write CSV file header    
    HEADER = "Data set;N;D_N;D_%;C_Min;C_Avg;C_Max;P_Min;P_Avg;P_Max;E_Min;E_Avg;E_Max;TP;FP\n"
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, "a") as myfile:
            myfile.write(HEADER)
        
    # Start estimation
    logger.info("[Main] Base NN reliability is: " + str(nn_reliability))
    differences = []
    for filename in sorted(os.listdir(test_data_dir)):
        if filename.endswith(".png"):
            path = test_data_dir + filename
            failure, error, c_time, p_time = evaluator.evaluate_nn_for_image(path, IM_RESIZE_HEIGHT, IM_RESIZE_WIDTH)
            if not failure is None:
                differences.append(failure)
           
            min_classification_time = min(min_classification_time, c_time)
            max_classification_time = max(max_classification_time, c_time)
            sum_classification_time = sum_classification_time + c_time
            
            min_prediction_time = min(min_prediction_time, p_time)
            max_prediction_time = max(max_prediction_time, p_time)
            sum_prediction_time = sum_prediction_time + p_time#
            
            min_error = min(min_error, error)
            max_error = max(max_error, error)
            sum_error = sum_error + error
            
            logger.info("[Main] NN reliability after seeing " + str(iterations) + " files is now: " + str(evaluator.get_nn_reliability()))
            iterations += 1
    
    datasetDuration         = (datetime.datetime.now() - datasetStart).total_seconds()
    difference_quota        = float(len(differences))        / iterations
    avg_classification_time = float(sum_classification_time) / iterations
    avg_prediction_time     = float(sum_prediction_time)     / iterations
    avg_error               = float(sum_error)               / iterations
    
    logger.info("[Main] Resulting NN reliability is: " + str(evaluator.get_nn_reliability()))
    logger.info("[Main] NN and predictor differ in " + str(difference_quota) + " %: " + str(differences))
    logger.info("[Main] Overall data set processing time   = " + str(datasetDuration) + " s")
    logger.info("[Main] Classfication times (min,mean,max)s  = (" + str(min_classification_time) + ", " + str(avg_classification_time) + ", " + str(max_classification_time) + ")")
    logger.info("[Main] Prediction times    (min,mean,max)s  = (" + str(min_prediction_time) + ", " + str(avg_prediction_time) + ", " + str(max_prediction_time) + ")")
    logger.info("[Main] Absolute errors     (min,mean,max)px = (" + str(min_error) + ", " + str(avg_error) + ", " + str(max_error) + ")")
    
    line = test_data_dir + ";" \
         + str(iterations-1) \
         + ";" + str(len(differences)) \
         + ";" + str(difference_quota) \
         + ";" + str(min_classification_time) \
         + ";" + str(avg_classification_time) \
         + ";" + str(max_classification_time) \
         + ";" + str(min_prediction_time) \
         + ";" + str(avg_prediction_time) \
         + ";" + str(max_prediction_time) \
         + ";" + str(min_error) \
         + ";" + str(avg_error) \
         + ";" + str(max_error) \
         + "\n"
         
    logger.info("[Main] For CSV <" + line + ">")
    
    with open(CSV_FILE, "a") as myfile:
        myfile.write(line)
    
    if len(differences) > 0:
        # Report differences
        failure_dir = test_data_dir.replace('/test-data','/tmp')
        # Maybe create output dir
        if not os.path.exists(failure_dir):
            os.makedirs(failure_dir)
        logger.info("[Main] Reporting " + str(len(differences)) + " differences to " + failure_dir) 
        for diff in differences:
            path = diff[0]
            expectation = diff[1]
            x = diff[2]
            # Load differencing image
            img = cv2.imread(path,0)
            # Annotate and save image
            failure_path = path.replace('.png', '.error.png').replace('/test-data','/tmp')
            logger.info("[Main] Reporting failed image to " + failure_path) 
            cv2.circle(img, (x,0), 5, (255,255,255), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,expectation,(30,260), font, 1,(255,255,255), 2)
            cv2.imwrite(failure_path, img)

def remove_file(logger, path):
    if os.path.isfile(path):
        logger.info("[Main] Removing " + path)
        os.remove(path)
    else:
        logger.info("[Main] The file " + path + " was already removed")

def reset(logger, pathes):
    for path in pathes:
        remove_file(logger, path)

def main(args):
    if not os.path.exists(LOG_ROOT):
        os.mkdir(LOG_ROOT)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    for directory in DATA_FOLDERS:
        if not os.path.exists(DATA_ROOT + directory):
            terminate("ERROR: Data directory " + str(DATA_ROOT + directory) + " not found.")

    graph = './models/graph_freezed.pb'    
    
    if len(args) < 2:
        usage()
    
    # TRAINING
    if args[1] == 'train':
        logger = prepare_logger(LOG_FILE)
        train(DATA_ROOT, MODELS_ROOT)
        
    # PREPROCESSING
    elif args[1] == 'preprocess':
        logger = prepare_logger(LOG_FILE)
        for directory in sorted(DATA_FOLDERS):
            training_dir = DATA_ROOT + directory + "png/"
            database_root = DATA_ROOT + directory[:-1] + "-manipulated/"
            database_dir = database_root + "png/"
            labels_path  = database_root + 'labels.csv'
            blur_flag_path = database_dir.replace("png/", "blur.txt")
            unsharpen_flag_path = database_dir.replace("png/", "unsharpen.txt")
            contour_flag_path = database_dir.replace("png/", "contour.txt")
            
            check_file_existence(graph)
            
            if not os.path.exists(training_dir):
                os.makedirs(training_dir)
                
            if not os.path.exists(database_dir):
                os.makedirs(database_dir)
            
            preprocess(logger, graph, training_dir, database_dir, blur_flag_path, unsharpen_flag_path, contour_flag_path, labels_path)
    
    # EVALUATION  
    elif args[1] == 'evaluate':
        if len(args) < 3:
            usage()
        
        test_data_dir = args[2]
        eval_log_file = LOG_FILE.replace('/log_', '/' + test_data_dir + '/log_')
        logger = prepare_logger(eval_log_file)
        
        database_root = DATA_ROOT + directory[:-1] + "-manipulated/"
        database_dir = database_root + "png/"
        
        if test_data_dir[-1:] != '/':
            test_data_dir = test_data_dir + '/'
        
        check_file_existence(graph)
        check_file_existence(test_data_dir)
        
        evaluate(logger, graph, database_dir, test_data_dir, 10, 0.38)
        
    # RESETTING            
    elif args[1] == 'reset':
        logger = prepare_logger(LOG_FILE)
        for directory in sorted(DATA_FOLDERS):
            database_root = DATA_ROOT + directory[:-1] + "-manipulated/"
            database_dir = database_root + "png/"
            labels_path  = database_root + 'labels.csv'
            blur_flag_path = database_dir.replace("png/", "blurred.txt")
            unsharpen_flag_path = database_dir.replace("png/", "unsharpened.txt")
            contour_flag_path = database_dir.replace("png/", "contour.txt")
            
            pathes = [ labels_path, blur_flag_path, unsharpen_flag_path, contour_flag_path, CSV_FILE ]
            
            reset(logger, pathes)
    else:
        usage()

if __name__ == "__main__":
    main(sys.argv)
