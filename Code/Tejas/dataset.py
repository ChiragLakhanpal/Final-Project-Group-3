from enum import Enum
import os
from typing import Callable, Mapping, Sequence

import cv2
import torch
from torch.utils import data

DATA_DIR = os.path.join("..", "Data")

# image config
CHANNELS = 3
IMAGE_SIZE = 224

class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"


class CustomDataset(data.Dataset):
    def __init__(self, image_ids: list[int], annotations_map: Mapping[int, Mapping], dataset_type: DatasetType, transforms: Sequence[Callable] | None=None):
        self.image_ids = image_ids
        self.annotations_map = annotations_map
        self.dataset_type = dataset_type
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        data = self.annotations_map.get(image_id, None)
        filepath = os.path.join(DATA_DIR, self.dataset_type, "images", data["file_name"])

        # load labels

        # load image
        img = cv2.imread(filepath)

        if self.transforms:
            img = self.transforms(img)

        x = torch.FloatTensor(img)
        x = torch.reshape(x, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

        return x, data