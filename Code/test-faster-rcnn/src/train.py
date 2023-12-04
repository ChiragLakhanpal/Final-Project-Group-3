
from .config import *
from .dataset import CustomCocoDataset

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import v2
from tqdm import tqdm


def get_train_dataset(batch_size: int=100):
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": lambda batch: tuple(zip(*batch))
    }

    # training_dataset = datasets.CocoDetection(TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_PATH, transforms=train_transforms)
    training_dataset = CustomCocoDataset(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGES_DIR, transforms=train_transforms)
    #  make dataset compatible with transforms
    # training_dataset = datasets.wrap_dataset_for_transforms_v2(training_dataset, target_keys=["boxes", "labels", "masks"])
    # training_dataset = datasets.wrap_dataset_for_transforms_v2(training_dataset, target_keys=["boxes", "labels"])
    
    training_generator = data.DataLoader(training_dataset, **train_params)

    return training_generator

def train_epoch(epoch, train_data, model, optimizer, criterion):
    # set model into training mode
    model.train()

    # enable gradient calc
    with torch.set_grad_enabled(True):
        with tqdm(total=len(train_data), desc=f"Epoch {epoch}") as pbar:
            for idx, (images, targets) in enumerate(train_data):
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                # model produces category and and mask vectors
                output = model(images, targets)
                
                # calculate loss for categories and segmentation masks
                loss = criterion(output, targets)

                # zero gradients
                optimizer.zero_grad(set_to_none=True)
                # backpropogation
                loss.backward()
                # clip the gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                # update weights
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(loss / (idx + 1)))


