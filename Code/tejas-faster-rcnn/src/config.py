import os
import random

import numpy as np
import torch


# Data paths
ROOT_DIR = "/home/ubuntu/Final-Project/Final-Project-Group-3"
TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, "Data/train/images")
TRAIN_ANNOTATIONS_PATH = os.path.join(ROOT_DIR, "Data/train/annotations.json")
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "Data/test/images")
TEST_ANNOTATIONS_PATH = os.path.join(ROOT_DIR, "Data/test/annotations.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Code/tejas-faster-rcnn/output")

# image config
CHANNELS = 3
IMAGE_SIZE = 512
NUM_CLASSES = 273

#### Hyperparameters ####
# detection threshold
THRESHOLD = 0.75
# Drouput layer value
DROPOUT = 0.4
# batch size for training
BATCH_SIZE = 16
# learning rate for the optimizer
LEARNING_RATE = 1e-3
# momentum for the optimizer
MOMENTUM = 0.999
# gradient clipping value (for stability while training)
GRADIENT_CLIPPING = 1.0
# weight decay (L2 regularization) for the optimizer
WEIGHT_DECAY = 1e-8
# number of epochs for training
EPOCHS = 1

# cuda config
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# seeding
SEED = 1122
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# random color map
COLORS = np.random.uniform(0, 255, size=(len(NUM_CLASSES), 3))