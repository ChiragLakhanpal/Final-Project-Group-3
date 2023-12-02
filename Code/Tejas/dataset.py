from enum import Enum
import os
from typing import Callable, Mapping, Sequence

from params import CHANNELS, DATA_DIR, IMAGE_SIZE

import cv2
import torch
from torch.utils import data


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
        annotations_data: dict = self.annotations_map.get(image_id, None)
        filepath = os.path.join(DATA_DIR, self.dataset_type, "images", data["file_name"])

        # load targets
        # dict containing boxes, categories, image_id, area, is_crowd, masks (Optional)
        annotations_target = torch.Tensor(annotations_data)

        # load image
        image = cv2.imread(filepath)

        if self.transforms:
            image = self.transforms(image)

        image = torch.FloatTensor(image)
        image = torch.reshape(image, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

        return image, annotations_target