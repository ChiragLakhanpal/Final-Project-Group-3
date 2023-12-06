from .config import *
from .dataset import CustomCocoDataset
from .model import get_model_object_detection
from .utils import collate_fn

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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
    def __init__(self, categories_map, iouType="bbox"):    
        self.iou_type = iouType
        self.categories_map = categories_map

    def run(self):
        image_ids = []
        model = get_model_object_detection()
        test_data_loader = get_test_dataset(BATCH_SIZE)
        coco_gt = COCO(TEST_ANNOTATIONS_PATH)
        coco_eval = COCOeval(coco_gt, iouType=self.iou_type)
        
        # load checkpoint
        checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_fasterrcnn_model.pt"), map_location=DEVICE)
        model.load_state_dict(checkpoint)
        
        # put model into evaluation mode
        model.eval()

        for images, targets in test_data_loader:
            with torch.no_grad():
                images = [image.to(DEVICE) for image in images]

                # model produces predictions
                predictions = model(images)
                
                # detach and convert predictions to COCO format
                results = self.prepare_coco_results(predictions, targets)

                image_ids.extend([result["image_id"] for result in results]) # extend with image ids

                # update COCOeval with current batch
                coco_dt = COCO.loadRes(coco_gt, results) if results else COCO()
                coco_eval.cocoDt = coco_dt
                coco_eval.params.imgIds = image_ids
                coco_eval.evaluate()

        # Summarize and print results
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval

    def prepare_coco_results(self, predictions, targets):
        coco_predictions = []

        for idx, prediction in enumerate(predictions):
            image_id = targets[idx]["image_id"].item()

            # convert bounding boxes to COCO format
            boxes = self.convert_to_xywh(prediction["boxes"])
            boxes = boxes.cpu().numpy()

            scores = prediction["scores"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                # if score >= THRESHOLD:
                category_id = self.categories_map.get(str(label))
                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(coord) for coord in box],
                    "score": float(score),
                })
        
        return coco_predictions
    
    def convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)