import os
import random

import numpy as np
import torch


DATA_DIR = os.path.join("..", "Data")

# image config
CHANNELS = 3
IMAGE_SIZE = 256

# hyperparameters
N_EPOCH = 3
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
THRESHOLD = 0.6
# DROPOUT = 0.4
SAVE_MODEL = True

# cuda config
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# seeding
SEED = 1122
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)