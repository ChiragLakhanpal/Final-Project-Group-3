from .config import *
from .dataset import CustomCocoDataset
from .model import get_model_object_detection
from .utils import collate_fn

import torch
from torchvision.transforms import v2
from torch.utils import data


def get_test_dataset(batch_size=100):
    # image transforms
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    test_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_fn
    }

    test_dataset = CustomCocoDataset(TEST_ANNOTATIONS_PATH, TEST_IMAGES_DIR, transforms=test_transforms)
    test_generator = data.DataLoader(test_dataset, **test_params)

    return test_generator


class ModelInference:
    def __init__(self):
        self.model = get_model_object_detection()
        self.checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    
    def run(self):
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(DEVICE).eval()