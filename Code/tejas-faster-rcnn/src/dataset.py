import os

from .config import CHANNELS, IMAGE_SIZE

import cv2
from matplotlib.path import Path
import numpy as np
from pycocotools.coco import COCO
import torch
from torch.utils import data
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class CustomCocoDataset(data.Dataset):
    def __init__(self, annotations_path, images_dir, transforms=None):
        self.coco_annotations = COCO(annotations_path)
        self.image_ids = list(sorted(self.coco_annotations.imgs.keys()))
        self.images_dir = images_dir
        self.transforms = transforms

        self.category_ids = sorted(self.coco_annotations.getCatIds())
        self.categories_map = { int(class_id): int(category_id) for class_id, category_id in enumerate(self.category_ids) }
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        # List: get annotation id from coco
        ann_ids = self.coco_annotations.getAnnIds(imgIds=image_id)
        # Dictionary: target coco_annotation file for an image
        annotations = self.coco_annotations.loadAnns(ann_ids)
        
        # path for input image
        img_filename = self.coco_annotations.loadImgs(image_id)[0]['file_name']
        filepath = os.path.join(self.images_dir, img_filename)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        # Size of bbox (Rectangular)
        areas = []
        # category class ID based on category ID
        labels = []
        # collection is_crowd values
        is_crowd = []

        for ann in annotations:
            coco_bbox = ann["bbox"]
            cat_id = ann["category_id"]
            torch_bbox = self.build_bbox(coco_bbox)
            boxes.append(torch_bbox)
            labels.append(self.category_ids.index(cat_id))
            is_crowd.append(ann["iscrowd"])
            areas.append(ann["area"])

        # load image
        image = cv2.imread(filepath)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # reshaped array to C x H x W
        image = np.transpose(image, (2, 0, 1))
        image = torch.FloatTensor(image)

        annotations_target = {
            "image_id": torch.Tensor(image_id),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image)),
            # "masks": tv_tensors.Mask(annotations_data["masks"]),
            "iscrowd": torch.as_tensor(is_crowd, dtype=torch.int64)
        }

        if self.transforms:
            image = self.transforms(image)

        
        image = torch.reshape(image, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

        return image, annotations_target

    def polygon_to_mask(polygon_vertices, image_shape):
        # Create a blank mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Create a Path object from the polygon vertices
        path = Path(polygon_vertices)

        # Create a grid of coordinates covering the entire image
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        points = np.vstack((x.flatten(), y.flatten())).T

        # Check if each point is inside the polygon
        mask_values = path.contains_points(points).reshape(image_shape)

        # Set the pixels inside the polygon to 1
        mask[mask_values] = 1

        return mask
    
    def build_bbox(self, bbox):
        [xmin, ymin, width, height] = bbox
        xmax = xmin + width
        ymax = ymin + height

        return [xmin, ymin, xmax, ymax]
