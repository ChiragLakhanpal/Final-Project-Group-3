from .params import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualConv(nn.Module):
    """
    This module represents a common architectural pattern in convolutional neural networks,
    especially in U-Net-like architectures."""

    def __init__(self, input_ch, output_ch):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_ch, output_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv_block(x)

class Contract(nn.Module):
    """
    Represents the contracting or downsampling path in U-Net architecture.
    """

    def __init__(self, input_ch, output_ch):
        super().__init__()

        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DualConv(input_ch, output_ch)
        )
    
    def forward(self, x):
        return self.down_conv(x)

class Expand(nn.Module):
    """
    Represents the expansive or upsampling path in U-Net architecture.
    """
    def __init__(self, input_ch, output_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(input_ch, input_ch // 2, kernel_size=2, stride=2)
        self.conv = DualConv(input_ch, output_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1) # upsample x1 tensor
        
        # compute height differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        # pad the x1 tensor to make its spatial dimensions match those of x2
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        # concatenate tensors along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CustomRPN(nn.Module):
    def __init__():
        super().__init__()
    
    def forward(self, x):
        x


class BboxHead(nn.Module):
    """
    Head for bouding box prediction
    """
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.in_features = in_features
        self.bbox_head = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            # dropout
            nn.Linear(128 * in_features * in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=num_classes * 4)
        )
    
    def forward(self, x):
        x = x.view(-1, 128 * self.in_features * self.in_features)
        return self.bbox_head(x)

class MaskHead(nn.Module):
    """
    Head for mask prediction
    """
    def __init__(self, input_ch, output_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_ch, out_channels=output_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ClassHead(nn.Module):
    """
    Head for category classification
    """
    def __init__(self, input_features, num_classes):
        super().__init__()

        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=num_classes)
        )
    
    def forward(self, x):
        return self.class_head(x)

class CustomUNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.initial = DualConv(input_channels, 64)
        self.down1 = Contract(64, 128)
        self.down2 = Contract(128, 256)
        self.down3 = Contract(256, 512)
        self.down4 = Contract(512, 1024)
        self.up1 = Expand(1024, 512)
        self.up2 = Expand(512, 256)
        self.up3 = Expand(256, 128)
        self.up4 = Expand(128, 64)

        # self.custom_unet = nn.Sequential(
        #     DualConv(input_channels, 64),
        #     # downsampling
        #     Contract(64, 128),
        #     Contract(128, 256),
        #     Contract(256, 512),
        #     Contract(512, 1024),
        #     # upsampling
        #     Expand(1024, 512),
        #     Expand(512, 256),
        #     Expand(256, 128),
        #     Expand(128, 64),
        # )

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # return self.custom_unet(x)
        return x
        
class CustomMultiLabelUNet(nn.Module):
    def __init__(self, output_classes):
        super().__init__()

        self.backbone = CustomUNet(input_channels=CHANNELS)
        # self.rpn = CustomRPN()
        self.bbox_head = BboxHead(in_features=64, num_classes=output_classes)
        self.class_head = ClassHead(input_features=64, num_classes=output_classes)
        self.mask_head = MaskHead(input_ch=64, output_ch=output_classes)

    def forward(self, x):
        features = self.backbone(x)
        
        # RPN forward pass
        # proposals, proposal_loss = self.rpn(features)
        
        # bbox prediction
        bbox_pred = self.bbox_head(features)

        # multi-label classification
        class_logits = self.class_head(features)

        # mask prediction
        mask_pred = self.mask_head(features)
        
        return {
            "boxes": bbox_pred,
            "labels": class_logits,
            "masks": mask_pred
        }

def save_model_summary(model: nn.Module, name: str):
    print(model, file=open(f"model_{name}_summary.txt", "w"))

def build_model():
    model = CustomMultiLabelUNet(output_classes=NUM_CLASSES)
    
    model = model.to(DEVICE)
    # catgeory_criterion = nn.BCEWithLogitsLoss()
    # mask_criterion = nn.MSELoss()
    # loss = operators.generalized_box_iou_loss
    save_model_summary(model, "custom")

    # return model, catgeory_criterion, mask_criterion
    return model

def set_optimizer(model):
    return torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
        # weight_decay=params.WEIGHT_DECAY,
    )

def set_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.1
    )

def build_custom_criterion(output, target):
    bounding_box_pred, class_logits, mask_pred = output["boxes"], output["labels"], output["masks"]

    # Bounding box loss using Smooth L1 Loss
    bounding_box_loss = F.smooth_l1_loss(bounding_box_pred, target["boxes"])

    # Multi-label classification loss using Cross Entropy Loss
    class_loss = F.binary_cross_entropy_with_logits(class_logits, target["labels"])

    # Mask loss using Binary Cross Entropy
    mask_loss = F.binary_cross_entropy_with_logits(mask_pred, target["masks"])

    # Combine losses
    total_loss = bounding_box_loss + class_loss + mask_loss

    return total_loss