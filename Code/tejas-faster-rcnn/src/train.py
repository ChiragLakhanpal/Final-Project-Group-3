
from .config import *
from .dataset import CustomCocoDataset
from .utils import collate_fn

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm


def get_train_dataset(batch_size: int=100):
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        # v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": collate_fn,
        "num_workers": NUM_WORKERS
    }

    # training_dataset = datasets.CocoDetection(TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_PATH, transforms=train_transforms)

    #  make dataset compatible with transforms
    # training_dataset = datasets.wrap_dataset_for_transforms_v2(training_dataset, target_keys=["boxes", "labels", "masks"])

    training_dataset = CustomCocoDataset(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGES_DIR, transforms=train_transforms)
    training_generator = data.DataLoader(training_dataset, **train_params)

    return training_generator

def train_epoch(epoch, train_data, model, optimizer, loss_list, loss_hist):
    # set model into training mode
    model.train()

    # enable gradient calc
    with torch.set_grad_enabled(True):
        with tqdm(total=len(train_data), desc=f"Train Epoch {epoch}") as pbar:
            for idx, (images, targets) in enumerate(train_data):
                images = [image.to(DEVICE) for image in images]
                targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                # model produces a loss dictionary
                loss_dict = model(images, targets)
                
                # calculate loss for bounding boxes and labels
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                
                # add to loss history list
                loss_list.append(loss_value)
                # send to Averager 
                loss_hist.send(loss_value)

                # zero gradients
                optimizer.zero_grad(set_to_none=True)
                # backpropogation
                # loss.backward()
                losses.backward()
                # clip the gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                # update weights
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str(f"Train Loss: {loss_value:.4f}")
    
    return loss_list
