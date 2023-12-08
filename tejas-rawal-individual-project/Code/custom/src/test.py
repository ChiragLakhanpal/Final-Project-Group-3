from .dataset import CustomDataset
from .params import DEVICE, IMAGE_SIZE, TEST_ANNOTATIONS_PATH, TEST_IMAGES_DIR

import torch
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2
import tqdm


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
        "collate_fn": lambda batch: tuple(zip(*batch))
    }

    # test_dataset = CustomDataset(
    #     annotations_path=TEST_ANNOTATIONS_PATH, images_dir=TEST_IMAGES_DIR, transforms=test_transforms
    # )
    test_dataset = datasets.CocoDetection(TEST_IMAGES_DIR, TEST_ANNOTATIONS_PATH, transforms=test_transforms)
    #  make dataset compatible with transforms
    test_dataset = datasets.wrap_dataset_for_transforms_v2(test_dataset, target_keys=["boxes", "labels", "masks"])
    test_generator = data.DataLoader(test_dataset, **test_params)

    return test_generator


def test_epoch(epoch, test_data, model, loss_func):
    # set model to evaluation mode
    model.eval()

    # disable gradient calculation
    with torch.set_grad_enabled(False):
        with tqdm(total=len(test_data), desc=f"Epoch {epoch}") as pbar:
            for idx, (images, targets) in enumerate(test_data):
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                # model produces category and and mask vectors
                output = model(images)

                # calculate loss for categories and segmentation masks
                loss = loss_func(output, targets)

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(loss / (idx + 1)))