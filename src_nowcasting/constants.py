import enum
import os

# Data.
N_CLASSES = 5

# Paths.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

CLASSES_CHECKPOINTS_DIR = os.path.join(ROOT_DIR, r'run_classification\checkpoints')
CLASSES_DATASET_DIR = os.path.join(ROOT_DIR, r'dataset\classes_dataset')
CLASSES_PREPROC_DIR = os.path.join(ROOT_DIR, r'dataset\classes_preprocessing')
CLASSES_RESULTS_DIR = os.path.join(ROOT_DIR, r'results\classification_forecast')
CLASS_SEQ_DATASET_DIR = os.path.join(ROOT_DIR, r'dataset\class_seq_dataset')
CLASS_TEST_SEQ = os.path.join(ROOT_DIR, r'dataset\class_sequence_test')
CLASS_TRAIN_SEQ = os.path.join(ROOT_DIR, r'dataset\class_seq_dataset\train')
CLASS_VAL_SEQ = os.path.join(ROOT_DIR, r'dataset\class_seq_dataset\validation')
GHI_DATASET_DIR = os.path.join(ROOT_DIR, r'dataset\ghi_dataset')
GREY_IMAGES_PATH = os.path.join(ROOT_DIR, r'dataset\pro_img')
IMAGES_PATH = os.path.join(ROOT_DIR, '003_Cusa')
MASK_DIR = os.path.join(ROOT_DIR, r'dataset\three2one\maschera.npy')
MATLAB_DATA_DIR = os.path.join(ROOT_DIR, r'dataset\raw\*')
PNG = os.path.join(ROOT_DIR, r'dataset\png')
PRO_IMAGES_PATH = os.path.join(ROOT_DIR, r'dataset\pro_img')
PROVA = os.path.join(ROOT_DIR, r'prova')
REGR_CHECKPOINTS_DIR = os.path.join(ROOT_DIR, r'run_regression\checkpoints')
REGR_RESULTS_DIR = os.path.join(ROOT_DIR, r'results\regression_forecast')
REGR_SEQ_DATASET_DIR = os.path.join(ROOT_DIR, r'dataset\regr_seq_dataset')
REUNIWATT = os.path.join(ROOT_DIR, r'dataset\dati_reuniwatt')
TEST = os.path.join(ROOT_DIR, r'dataset\classes_test')
TRAIN = os.path.join(ROOT_DIR, r'dataset\classes_dataset\train')
VAL = os.path.join(ROOT_DIR, r'dataset\classes_dataset\validation')
