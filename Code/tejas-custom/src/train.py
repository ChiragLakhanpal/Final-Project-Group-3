from .dataset import CustomDataset
from .params import DEVICE, IMAGE_SIZE, TRAIN_IMAGES_PATH

import torch
from torch.utils import data
from torchvision.transforms import v2
from tqdm import tqdm


def get_train_dataset(annotations, batch_size: int=100):
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": lambda batch: tuple(zip(*batch))
    }

    training_dataset = CustomDataset(annotations, image_dir=TRAIN_IMAGES_PATH, transform=train_transforms)
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
                
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                # model produces category and and mask vectors
                output = model(images)                
                
                # calculate loss for categories and segmentation masks
                loss = criterion(output, targets)

                # backward passs
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(loss / (idx + 1)))


