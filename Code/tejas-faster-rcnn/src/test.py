from .config import *
from .dataset import CustomCocoDataset
from .utils import collate_fn

import torch
from torch.utils import data
from torchvision.transforms import v2
from tqdm import tqdm


def get_test_dataset(batch_size=100):
    # image transforms
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "num_workers": NUM_WORKERS
    }

    test_dataset = CustomCocoDataset(TEST_ANNOTATIONS_PATH, TEST_IMAGES_DIR, transforms=test_transforms)
    test_generator = data.DataLoader(test_dataset, **test_params)

    return test_generator


def test_epoch(epoch, test_data, model, loss_list, loss_hist):
    with tqdm(total=len(test_data), desc=f"Test Epoch {epoch}") as pbar:
        for idx, (images, targets) in enumerate(test_data):
            # disable gradient calculation
            with torch.no_grad():
                images = [image.to(DEVICE) for image in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]


                # model produces loss dictionary
                loss_dict = model(images, targets)
                
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                # add to loss history list
                loss_list.append(loss_value)
                # send to Averager 
                loss_hist.send(loss_value)

                # calculate loss for categories and segmentation masks
                # loss = loss_func(output, targets)

                pbar.update(1)
                pbar.set_postfix_str(f"Test Loss: {loss_value:.4f}")
    
    return loss_list