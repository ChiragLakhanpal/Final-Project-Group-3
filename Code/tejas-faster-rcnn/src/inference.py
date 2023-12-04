from .config import *
from .dataset import CustomCocoDataset
from .model import get_model_object_detection
from .utils import collate_fn

import cv2
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
        self.data = get_test_dataset(BATCH_SIZE)
    
    def run(self):
        checkpoint = torch.load(f"{OUTPUT_DIR}/best_model.pth", map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(DEVICE)
        self.model.eval()

        with torch.set_grad_enabled(False):
            for images, targets in self.data:
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                # model produces bbox, labels, and scores
                outputs = self.model(images, targets)
                # detach outputs from GPU
                outputs = [{k: v.detach().cpu() for k, v in t.items()} for t in outputs]

                # evaluate outputs
                # return boxes and predictions for visualization
    
    def predict_image(self, image):
            image = cv2.imread(image)
            orig_image = image.copy()
            # BGR to RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # make the pixel range between 0 and 1
            image /= 255.0
            # bring color channels to front
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            # convert to tensor
            image = torch.tensor(image, dtype=torch.float).cuda()
            # add batch dimension
            image = torch.unsqueeze(image, 0)

            with torch.no_grad():
                outputs = self.model(image.to(DEVICE))