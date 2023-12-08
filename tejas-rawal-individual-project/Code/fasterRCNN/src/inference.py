from .config import *
from .model import Phase, get_model_object_detection
from .test import get_test_dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from tqdm import tqdm


class ModelInference:
    def __init__(self, model_name, categories_map, iouType="bbox"):    
        self.model_name = model_name
        self.categories_map = categories_map
        self.iou_type = iouType

    def run(self):
        results = []
        model = get_model_object_detection(phase=Phase.TEST, pretrained=self.model_name)
        test_data_loader = get_test_dataset(BATCH_SIZE)
        
        # load checkpoint
        checkpoint = torch.load(os.path.join(OUTPUT_DIR, f"best_{self.model_name.value}.pt"), map_location=DEVICE)

        model.load_state_dict(checkpoint)
        
        # put model into evaluation mode
        model.eval()

        with tqdm(total=len(test_data_loader), desc=f"Inference") as pbar:
            for idx, (images, targets) in enumerate(test_data_loader):
                pbar.set_description(f"Inferencing batch {idx + 1}")
                # disable gradient calc
                with torch.no_grad():
                    images = [image.to(DEVICE) for image in images]

                    # model produces predictions
                    predictions = model(images)
                    
                    # convert predictions to COCO format and save
                    results.extend(self.prepare_coco_results(predictions, targets))

                    pbar.update(1)

        # Summarize and print results
        coco_gt = COCO(TRAIN_ANNOTATIONS_PATH)
        coco_dt = COCO.loadRes(coco_gt, results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=self.iou_type)
        coco_eval.evaluate()
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
                if score >= THRESHOLD:
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