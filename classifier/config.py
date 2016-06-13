import datetime

# Testing Framework Parameters
DEBUG                = True
DATA_ROOT            = './training-data/'
LOG_ROOT             = './'
MODELS_ROOT          = './models/'
LOG_FILE             = LOG_ROOT + 'log_{:%Y-%m-%d_%H-%M-%S}.txt'.format(datetime.datetime.now())
IM_RESIZE_HEIGHT     = 120
IM_RESIZE_WIDTH      = 180
IM_RESIZE            = True
NN_START_RELIABILITY = 100
NN_RELIABILITY_DELTA = 1

# NN Learning Parameters
LOAD_MODEL           = False
TRAIN_MODEL          = True
FREEZE_GRAPH         = True
BATCH_SIZE           = 50
STEP_SIZE_MAX        = 10000
STEP_SIZE_PRINT      = 100
STEP_SIZE_SAVE       = 1000
TRAINING_RATIO       = 0.9
DISTORTION_RATE      = 1.0
ADD_FLIPPED          = True
CAR_ORIGIN_POS       = [376.0, 480.0]
# Set parameters
learning_rate        = 0.001
beta1                = 0.9
beta2                = 0.999
epsilon              = 0.1

# Evaluation Parameters
CSV_FILE = "data.csv"

# Common Parameters
DATA_FOLDERS = [
     'dashedlinesmissing/',
     'fulltrack1/',
     #'fulltrack2/', # Used for testing
     'leftcurve/',
     'rightcurve/',
     'rightlanemissing/',
     'roadnear/',
     'startbox/',
     'straightroad/'
]
