from enum import Enum
import os
from typing import Callable, Mapping, Sequence

from .params import CHANNELS, IMAGE_SIZE

import cv2
from matplotlib.path import Path
import numpy as np
from pycocotools.coco import COCO
import torch
from torch.utils import data
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"


class CustomDataset(data.Dataset):
    def __init__(self, annotations, images_dir, transforms=None):
        self.coco_annotations = COCO(annotations)
        self.image_ids = list(sorted(self.coco_annotations.imgs.keys()))
        self.images_dir = images_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        # List: get annotation id from coco
        ann_ids = self.coco_annotations.getAnnIds(imgIds=image_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = self.coco_annotations.loadAnns(ann_ids)
        breakpoint()
        # path for input image
        img_filename = self.coco_annotations.loadImgs(image_id)[0]['file_name']
        filepath = os.path.join(self.images_dir, img_filename)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            coco_bbox = coco_annotation[i]['bbox']
            torch_bbox = self.build_bbox(coco_bbox)
            boxes.append(torch_bbox)
            areas.append(coco_annotation[i]['area'])


        # load image
        image = read_image(filepath)

        if self.transforms:
            image = self.transforms(image)

        image = tv_tensors.Image(image)
        image = torch.reshape(image, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        annotations_target = {
            "image_id": torch.Tensor(image_id),
            # "labels": annotations_data["categories"],
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image)),
            # "masks": tv_tensors.Mask(annotations_data["masks"]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        return image, annotations_target

    # def polygon_to_mask(polygon_vertices, image_shape):
    #     # Create a blank mask
    #     mask = np.zeros(image_shape, dtype=np.uint8)

    #     # Create a Path object from the polygon vertices
    #     path = Path(polygon_vertices)

    #     # Create a grid of coordinates covering the entire image
    #     x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    #     points = np.vstack((x.flatten(), y.flatten())).T

    #     # Check if each point is inside the polygon
    #     mask_values = path.contains_points(points).reshape(image_shape)

    #     # Set the pixels inside the polygon to 1
    #     mask[mask_values] = 1

    #     return mask
    
    def build_bbox(self, bbox):
        [xmin, ymin, width, height] = bbox
        xmax = xmin + width
        ymax = ymin + height

        return [xmin, ymin, xmax, ymax]
