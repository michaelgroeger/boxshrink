import os

import torch

# Training settings
BASE_DIR = "/".join(os.getcwd().split("/"))
DATA_DIR = BASE_DIR + "/data"
BOX_DIR = DATA_DIR + "/boxmasks"
RAPID_DIR = DATA_DIR + "/testing/rapid_boxshrink"
ROBUST_DIR = DATA_DIR + "/testing/robust_boxshrink"
BEST_MODEL_DIR = BASE_DIR + "/model/best"
CHECKPOINT_MODEL_DIR = BASE_DIR + "/model/checkpoint"
EXPORT_CSV_DIR = BASE_DIR + "/model/metrics"
EVAL_ON_MASKS = True
# Training input should be one of boxes, rapid_boxshrink, robust_boxshrink
TRAINING_INPUT = "robust_boxshrink"
# Choose here any prefix to identify your runs
STATE = "_".join(["Benchmarking", TRAINING_INPUT])
EXPORT_BEST_MODEL = True
if EXPORT_BEST_MODEL == False:
    model_name = None

# Generic thresholds and superpixel settings
IOU_THRESHOLD = 0.1
MASK_OCCUPANCY_THRESHOLD = 0.04
N_SEGMENTS = 250
SLIC_COMPACTNESS = 10
SUPERPIXEL_OVERLAP_THRESHOLD = 0.1

# robustBoxshrink settings and thresholds
N_SEGMENTS_ROBUST = 300
# If || dist(background) - dist(foreground)|| <= THRESHOLD_CLOSNESS, then foreground
THRESHOLD_CLOSNESS = 0

# rapidBoxshrink settings and threshold
SUPERPIXEL_OVERLAP_THRESHOLD_RAPID = 0.5
N_SEGMENTS_RAPID = 200

# FCRF settings
PAIRWISE_GAUSSIAN = (5, 5)
PAIRWISE_BILATERAL = (25, 25)
RGB_STD = (10, 10, 10)
NUM_INFERENCE = 10

ENCODER = "vgg16"
DECODER = "Unet"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["Background", "Finding"]
ACTIVATION = "sigmoid"  # None for Logits
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPTIMIZER = "Adam"
LEARNING_RATE = 0.0001
# Whether to use learning rate scheduling and which one
LEARNING_RATE_SCHEDULING = True
SCHEDULE_TYPE = "STEP"
WEIGHT_DECAY = 0

STEP_SIZE = 5
GAMMA = 0.5

LOSS = "CrossEntropyLoss"
BATCH_SIZE = 10
N_EPOCHS = 25
START_EPOCH = 0
# Return intermediate results & Plot losses
PER_X_BATCH = 1
PER_X_EPOCH = 2
PER_X_EPOCH_PLOT = 1

# Mode for model name
MODE = "Unet_colonoscopy"
