from enum import Enum

from .config import *

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class Phase(Enum):
    TRAIN = "train"
    TEST = "test"

class Pretrained(Enum):
    MOBILE_NET = "fasterrcnn_mobilenet_v3_large_fpn"
    RESNET = "fasterrcnn_resnet50_fpn_v2"


def get_model_object_detection(phase: Phase=Phase.TEST, pretrained: Pretrained=Pretrained.MOBILE_NET):
    # load a model pre-trained on COCO
    try:
        model = torchvision.models.get_model(pretrained.value, weights="DEFAULT")
        # replace the classifier with a new one, that has get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        
        if phase == Phase.TRAIN:
            # freeze all parameters of the model
            for param in model.parameters():
                param.requires_grad = False
            
            # make ROI head layer trainable
            for param in model.roi_heads.parameters():
                param.requires_grad = True
            
            # unfreeze last 50% of parameters for training
            max_index_param = len(list(model.parameters())) - 1
            unfreeze_from = int(0.5 * max_index_param) - 1
            for i, param in enumerate(model.parameters()):
                if i >= unfreeze_from and i < max_index_param: 
                    param.requires_grad = True

        model = model.to(DEVICE)
        save_model_summary(model, pretrained.value)

        return model
    except Exception:
        raise (f"Unable to load model {pretrained.value}")


def get_model_instance_segmentation():
    # load mobilenetv2 pre-trained model for classification
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # ``FasterRCNN`` needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    backbone.out_channels = 1280

    # generate 5 x 3 anchors per spatial location,
    # with 5 different sizes and 3 different aspect ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2,
    )

    rcnn_head = nn.Sequential(
        nn.Linear(in_features=backbone.out_channels * 7 * 7, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=256),
        nn.ReLU(),
    )
    mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=NUM_CLASSES)


    # put the pieces together inside a Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        roi_heads=dict(box_head=rcnn_head, mask_predictor=mask_predictor),
    )

    model = model.to(DEVICE)
    save_model_summary(model, "custom-faster-rcnn")

    return model

def set_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    
    return torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )

def set_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1, verbose=True
    )

def save_model_summary(model: nn.Module, name: str):
    print(model, file=open(f"{OUTPUT_DIR}/model_{name}_summary.txt", "w"))