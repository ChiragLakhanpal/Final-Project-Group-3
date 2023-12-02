from dataset import CustomDataset, DatasetType
from params import DEVICE, IMAGE_SIZE, N_EPOCH
from model import build_model

import torch
from torch.utils import data
from torchvision.transforms import v2
from tqdm import tqdm


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

def train_model(batch_size=100, pretrained=False):
    train_data = read_train_data(batch_size)
    model, optimizer, catgeory_criterion, mask_criterion = build_model(pretrained)

    for epoch in range(1, N_EPOCH + 1):
        # set model into training mode
        model.train()

        # enable gradient calc
        with torch.autograd.set_grad_enabled(True):
            with tqdm(total=len(train_data), desc=f"Epoch {epoch}") as pbar:
                for idx, (images, targets) in enumerate(train_data):
                    images = images.to(DEVICE)
                    targets = targets.to(DEVICE)

                    # need annotations for categories masks

                    optimizer.zero_grad()
                    catgeory_criterion.zero_grad()

                    # model produces category logit vectors and and mask vectors
                    class_output, mask_output = model(images)

                    # get categories and masks from targets
                    class_targets = targets
                    mask_targets = targets
                    
                    # annotations will have a category, mask, and bouding boxes
                    class_loss = catgeory_criterion(class_output, class_targets)
                    mask_loss = mask_criterion(mask_output, mask_targets)

                    # backward passs
                    class_loss.backward()
                    mask_loss.backward()
                    optimizer.step()

                    pbar.update(1)
                    pbar.set_postfix_str("Train Category Loss: {:.5f}".format(class_loss / (idx + 1)))
                    pbar.set_postfix_str("Train Mask Loss: {:.5f}".format(mask_loss / (idx + 1)))
