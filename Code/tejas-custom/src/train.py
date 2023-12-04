from .dataset import CustomDataset
from .params import DEVICE, IMAGE_SIZE, TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGES_DIR

import torch
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

def custom_transform(batch):
    return list(zip(*batch))

def get_train_dataset(batch_size: int=100):
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": lambda batch: list(zip(*batch))
        # "collate_fn": custom_transform
    }

    # training_dataset = CustomDataset(
    #     annotations_path=TRAIN_ANNOTATIONS_PATH, images_dir=TRAIN_IMAGES_DIR, transforms=train_transforms
    # )
    training_dataset = datasets.CocoDetection(TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_PATH, transforms=train_transforms)
    #  make dataset compatible with transforms
    training_dataset = datasets.wrap_dataset_for_transforms_v2(training_dataset, target_keys=["boxes", "labels", "masks"])

    training_generator = data.DataLoader(training_dataset, **train_params)

    return training_generator

def train_epoch(epoch, train_data, model, optimizer, criterion):
    # set model into training mode
    model.train()

    # enable gradient calc
    with torch.set_grad_enabled(True):
        with tqdm(total=len(train_data), desc=f"Epoch {epoch}") as pbar:
            for idx, (images, targets) in enumerate(train_data):
                optimizer.zero_grad()
                # category_loss.zero_grad()
                
                images = torch.stack(images).to(DEVICE)
                # targets = targets.to(DEVICE)

                # model produces category and and mask vectors
                output = model(images)
                
                # calculate loss for categories and segmentation masks
                loss = criterion(output, targets)

                # backward passs
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(loss / (idx + 1)))


