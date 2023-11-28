import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.ops as operators

N_EPOCH = 10
BATCH_SIZE = 64
LR = 1e-3
THRESHOLD = 0.6
DROPOUT = 0.4
SAVE_MODEL = True

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# seeding
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

def build_model(pretrained: bool | None=False):
    model = CustomModel()
    if pretrained:
        # mask rcnn
        model = models.detection.maskrcnn_resnet50_fpn_v2(weights=models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss = operators.complete_box_iou_loss

    return model, optimizer, loss