from params import DEVICE, LEARNING_RATE, MOMENTUM

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.ops as operators

class CustomModel(nn.Module):
    def __init__(self, target_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, target_classes)
        self.activation = nn.ReLU(inplace=True)

        self.target_class = self._create_classification()
        self.bbox_conv, self.bbox_fc = self._create_bbox_regression()
    
    def _create_classification(self):
        return nn.Sequential(
            #32 -> 64
            self.conv2,
            self.activation,
            self.pool,
            
            #64 -> 64
            self.conv3,
            self.activation,
            self.pool,
            
            #64 -> 128
            self.conv4,
            self.activation,
            self.pool,
            
            #128 -> 128
            self.conv5,
            self.activation,
            self.pool,
            self.bn2,

            #128 -> 64
            self.conv6,
            self.activation,
            self.pool,

            #64 -> 64
            self.conv3,
            self.activation,
            self.pool,
            self.bn1,

            #63 -> 32
            self.conv7,
            self.activation,
            self.pool
        )

    def _create_bbox_regression(self):
        conv_seq = nn.Sequential(
            self.conv2,
            self.activation,
            self.pool,
            self.conv3,
            self.activation,
            self.pool,
            self.bn1,
            self.conv7,
            self.activation,
            self.pool
        )
        fc_seq = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            self.activation,
            nn.Linear(in_features=1024, out_features=512),
            self.activation,
            nn.Linear(512, 4) # output for bbox coordinates
        )

        return conv_seq, fc_seq

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x_clone = x.clone()
        
        # target classification
        x1 = self.target_class(x)
        x1 = nn.functional.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        # bbox regression
        x2 = self.bbox_conv(x_clone)
        # reshape/flatten output
        x2 = self.bbox_fc(x2)
        
        return x1, x2

def save_model(model: nn.Module, name: str):
    print(model, file=open(f"model_{name}_summary.txt", "w"))

def build_model(num_classes: int, pretrained: bool | None=False):
    model = CustomModel(target_classes=num_classes)
    if pretrained:
        # mask rcnn
        model = models.detection.maskrcnn_resnet50_fpn_v2(weights=models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    catgeory_criterion = nn.BCEWithLogitsLoss()
    mask_criterion = nn.MSELoss()
    # loss = operators.generalized_box_iou_loss

    if pretrained:
        save_model(model, "mask_rcnn")
    else:
        save_model(model, "custom")

    return model, optimizer, catgeory_criterion, mask_criterion
