import os
import random

import numpy as np
import torch


DATA_DIR = os.path.join("..", "Data")

# Data paths
TRAIN_IMAGES_PATH = os.path.join(DATA_DIR, "/train/images/")
TRAIN_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "/train/annotations.json")
TEST_IMAGES_PATH = os.path.join(DATA_DIR, "/val/images/")
TEST_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "/val/annotations.json")

# image config
CHANNELS = 3
IMAGE_SIZE = 256

#### Hyperparameters ####
# target probability threshold
THRESHOLD = 0.5
# Drouput layer value
DROPOUT = 0.4
# batch size for training
BATCH_SIZE = 64
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
