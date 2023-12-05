import os
import random

import numpy as np
import torch


# Data and directory paths
ROOT_DIR = "/home/ubuntu/Final-Project/Final-Project-Group-3"
OUTPUT_DIR = os.path.join(ROOT_DIR, "Code/tejas-faster-rcnn/output")
TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, "Data/train/images")
TRAIN_ANNOTATIONS_PATH = os.path.join(ROOT_DIR, "Data/train/annotations.json")
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "Data/test/images")
TEST_ANNOTATIONS_PATH = os.path.join(ROOT_DIR, "Data/test/annotations.json")

# image config
CHANNELS = 3
IMAGE_SIZE = 256
NUM_CLASSES = 323

#### Hyperparameters ####
# detection threshold
THRESHOLD = 0.6
# Drouput layer value
DROPOUT = 0.4
# batch size for training
BATCH_SIZE = 16
# learning rate for the optimizer
LEARNING_RATE = 1e-3
# momentum for the optimizer
MOMENTUM = 0.9
# gradient clipping value (for stability while training)
GRADIENT_CLIPPING = 1.0
# weight decay (L2 regularization) for the optimizer
WEIGHT_DECAY = 1e-5
# number of epochs for training
EPOCHS = 3

# cuda config
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# seeding
SEED = 1122
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# random color map
COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))