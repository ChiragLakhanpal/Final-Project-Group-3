from dataset import CustomDataset, DatasetType, IMAGE_SIZE

import torch
from torch.utils import data
from torchvision.transforms import v2


def read_train_data(batch_size=100):
    image_ids = []
    annotations_by_imageId = {}

    # Data Loaders
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_params = {
        'batch_size': batch_size,
        'shuffle': True
    }

    training_set = CustomDataset(image_ids, annotations_by_imageId, DatasetType.TRAIN, transform=train_transforms)
    training_generator = data.DataLoader(training_set, **train_params)

    return training_generator
