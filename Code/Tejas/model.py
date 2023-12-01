import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.ops as operators


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# hyperparameters
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
N_EPOCH = 10
# THRESHOLD = 0.6
# DROPOUT = 0.4
# SAVE_MODEL = True

# seeding
SEED = 1122
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

def save_model(model: nn.Module, name: str):
    print(model, file=open(f"model_{name}_summary.txt", "w"))

def build_model(pretrained: bool | None=False):
    model = CustomModel()
    if pretrained:
        # mask rcnn
        model = models.detection.maskrcnn_resnet50_fpn_v2(weights=models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss = operators.generalized_box_iou_loss

    if pretrained:
        save_model(model, "mask_rcnn")
    else:
        save_model(model, "custom")

    return model, optimizer, loss
