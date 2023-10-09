import enum
import os


MODEL_TYPE = 'scnn'                                         # model_type chosen for the training.
FORECAST_HORIZON = 30                                       # time horizon.
NO_IMAGES = 3
EPSILON = 1e-3

# Image parameters
IMG_SIZE = [128, 128]                                       # image size.
CHANNELS = 1                                                # image channels.
ELEVATION_THRESHOLD = 20

TRAIN_BATCHSIZE = 32                                        # batch size for train.
TEST_BATCHSIZE = 1                                          # batch size for test.
EPOCHS = 100                                                # maximum number of epochs.
TRAIN_SIZE = 0.8


# Paths
PATH_INPUT_FOLDER = r'..\dataset\IR_images_nowcasting'              # Raw images folder path
PATH_OUTPUT_FOLDER = r'..\dataset\IR_images_postprocess'            # Postprocessed images folder path
PATH_WEATHER_FILES = r'..\dataset\sensors'                          # Weather station data folder path

CHECKPOINT_PATH = r'.\model\checkpoints'                       # path to the model weight folder
LOG_PATH = None                                                     # path to save the log csv folder




# Parameter dictionaries
train_params = {'batch_size': TRAIN_BATCHSIZE,
           'dim': (IMG_SIZE[0], IMG_SIZE[1], 1 * NO_IMAGES),
           'channel_IMG': CHANNELS,
           'shuffle': False,
           'iftest': False}

valid_params = {'batch_size': TRAIN_BATCHSIZE,
           'dim': (IMG_SIZE[0], IMG_SIZE[1], 1 * NO_IMAGES),
           'channel_IMG': CHANNELS,
           'iftest': False}

test_params = {'batch_size': TEST_BATCHSIZE,
           'dim': (IMG_SIZE[0], IMG_SIZE[1], 1 * NO_IMAGES),
           'channel_IMG': CHANNELS,
           'shuffle': False,
           'iftest': False}