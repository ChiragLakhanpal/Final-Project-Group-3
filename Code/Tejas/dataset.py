from enum import Enum
import os
from typing import Callable, Mapping, Sequence

import cv2 as opencv
import torch
from torch.utils import data

DATA_DIR = os.path.join("..", "Data")

CHANNELS = 3
IMAGE_SIZE = 224

class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"


class CustomDataset(data.Dataset):
    def __init__(self, image_ids: list[str], dataset: Mapping[str, Mapping], dataset_type: DatasetType, transforms: Sequence[Callable] | None=None):
        self.image_ids = image_ids
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_filename = self.image_ids[index]
        filepath = os.path.join(DATA_DIR, self.dataset_type, "images", image_filename)
        data = self.dataset.get(image_filename, None)

        # load labels
        # load image
        img = opencv.imread(filepath)

        if self.transforms:
            img = self.transforms(img)

        x = torch.FloatTensor(img)
        x = torch.reshape(x, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

        return x