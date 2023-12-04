from .config import *
from .dataset import CustomCocoDataset
from .model import get_model_object_detection
from .utils import collate_fn

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torchvision.transforms import v2
from torch.utils import data


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
    }

    test_dataset = CustomCocoDataset(TEST_ANNOTATIONS_PATH, TEST_IMAGES_DIR, transforms=test_transforms)
    test_generator = data.DataLoader(test_dataset, **test_params)

    return test_generator


class ModelInference:
    def __init__(self, iouType="bbox"):    
        self.iou_type = iouType

    def run(self):
        model = get_model_object_detection()
        test_data = get_test_dataset(BATCH_SIZE)
        coco_gt = self.data.dataset.coco_annotations
        coco_eval = COCOeval(self.coco_gt, iouType=self.iouType)
        # load checkpoint
        checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        # put model into evaluation mode
        model.eval()

        for images, targets in test_data:
            images = list(image.to(DEVICE) for image in images)
            # targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                # model produces bbox, labels, and scores
                predictions = model(images)
            
            # detach and convert predictions to COCO format
            results = self.coco_predictions(predictions, targets)

            # update COCOeval with current batch
            coco_dt = COCO.loadRes(coco_gt, results) if results else COCO()
            coco_eval.cocoDt = coco_dt
            # self.coco_eval.evaluate()
            # self.coco_eval.accumulate()

        # Summarize and print results
        coco_eval.accumulate()
        stats = coco_eval.summarize()

        return coco_eval, stats

    def coco_predictions(self, predictions, targets):
        coco_predictions = []

        for i in range(len(predictions)):
            prediction = predictions[i]
            image_id = targets[i]["image_id"].item()
            boxes = prediction["boxes"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(coord) for coord in box],
                    "score": float(score),
                })
        
        return coco_predictions
    
    def predict_image(self, image, model):
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
            predictions = model(image.to(DEVICE))
    
    def convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)