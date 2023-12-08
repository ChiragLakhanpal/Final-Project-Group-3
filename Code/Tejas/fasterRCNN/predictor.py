

import os
import random

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2

ROOT_DIR = "/home/ubuntu/Final-Project/Final-Project-Group-3"
OUTPUT_DIR = os.path.join(ROOT_DIR, "Code/tejas-faster-rcnn/output")
TRAIN_ANNOTATIONS_PATH = os.path.join(ROOT_DIR, "Data/train/annotations.json")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
IMAGE_SIZE = 256

# seeding
SEED = 1122
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def get_annotations():
    coco_annotations = COCO(TRAIN_ANNOTATIONS_PATH)
    category_ids = sorted(coco_annotations.getCatIds())
    categories_map = {
        int(class_id): int(category_id) for class_id, category_id in enumerate(category_ids)
    }

    
    return coco_annotations, categories_map

def get_model_object_detection(num_classes: int):
    # load a model pre-trained on COCO
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    )

    # replace the classifier with a new one, that has get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(DEVICE)

    return model

class Predictor:
    def __init__(self, categories_map, annotations):
        self.categories_map = categories_map
        self.annotations = annotations
        self.model = get_model_object_detection(323)
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    
    def __call__(self, image_path: str, model_checkpoint):
        # load best model
        self.model.load_state_dict(model_checkpoint)
        # put model into evaluation mode
        self.model.eval()

        # load the image
        image = cv2.imread(image_path)
        image_copy = image.copy()
        # convert the image from BGR to RGB 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # normalize image and covert to tensor
        image = cv2.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255.0
        # change image shape from channels last to channels first
        image = image.transpose((2, 0, 1))

        # apply transforms
        image = torch.FloatTensor(image)
        image = self.transforms(image)
        
        # add the batch dimension
        image = np.expand_dims(image, axis=0)
        image = torch.as_tensor(image, dtype=torch.float32).to(DEVICE)

        # get the detections and predictions
        detections = [{}]
        with torch.no_grad():
            detections = self.model(image)
    
        image, category_names = self.process(image_copy, detections[0])
        return image, category_names

    def process(self, image, prediction):
        category_names = []

        for i in range(0, len(prediction["boxes"])):
            if prediction["scores"][i] >= THRESHOLD:
                label = int(prediction["labels"][i])
                category_id = self.categories_map.get(label)
                box = prediction["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                
                category_label = ""
                if category_id in self.annotations["categories"]:
                    # display the prediction to our terminal
                    category_label = self.annotations["categories"][category_id]["name"]
                    category_names.append(category_label)
                    category_label = f"{category_label}: {prediction['scores'][i]:.2f}"
                
                # random color
                color = np.random.uniform(0, 255, size=(1, 3))
                # draw the bounding box and label on the image
                cv2.rectangle(
                    image, (startX, startY), (endX, endY), color, 2
                )
                
                position = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    image, category_label, (startX, position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        return image, category_names



# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# checkpoint = torch.load(
#     os.path.join(ROOT_DIR, "Code", "best_fasterrcnn_resnet_model.pt"), map_location=device
# )
# annotations, categories = get_annotations()
# fasterrcnn_predictor = Predictor(categories, annotations)

# image_paths = [
#     f"{ROOT_DIR}/Code/apples.jpeg"
# ]

# for pth in image_paths:
#     image, names = fasterrcnn_predictor(pth, checkpoint)
#     # dispaly annotated image
#     print(names)
