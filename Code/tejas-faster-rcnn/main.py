from enum import Enum
from functools import cache
import json

from src.config import *
from src.inference import ModelInference
from src.model import get_model_object_detection, set_optimizer, set_scheduler
from src.train import get_train_dataset, train_epoch
from src.test import get_test_dataset, test_epoch
from src.utils import Averager, SaveBestModel, save_loss_plot

from pycocotools.coco import COCO
# from torchvision.models.detection.roi_heads import maskrcnn_loss

class Phase(Enum):
    TRAIN = "train"
    TEST = "test"

@cache
def get_raw_annotations(phase):
    annotations_path = (
        TRAIN_ANNOTATIONS_PATH if phase == Phase.TRAIN
        else TEST_ANNOTATIONS_PATH
    )

    with open(os.path.join(ROOT_DIR, annotations_path), "r") as file_json:
        annotations_data = json.loads(file_json.read())
        return annotations_data

@cache
def get_categories_map(phase: Phase):
    annotations_path = (
        TRAIN_ANNOTATIONS_PATH if phase == Phase.TRAIN
        else TEST_ANNOTATIONS_PATH
    )
    coco_annotations = COCO(annotations_path)
    category_ids = sorted(coco_annotations.getCatIds())
    categories_map = {
        int(class_id): int(category_id) for class_id, category_id in enumerate(category_ids)
    }

    # save as json
    with open("categories_map.json", "w") as f:
        json.dump(categories_map, f)
    
    return categories_map


class ModelRunner:
    def __init__(self, batch_size: int=100) -> None:
        self.batch_size = batch_size
        # self.model = build_model()
   
    def train_and_test(self):
        # record loss history
        train_loss_list = []
        test_loss_list = []

        # Initialize Averagers
        train_loss_hist = Averager()
        test_loss_hist = Averager()

        # initialize SaveBestModel
        save_best_model = SaveBestModel()

        train_ds = get_train_dataset(self.batch_size)
        test_ds = get_test_dataset(self.batch_size)
        
        # self.model = get_model_instance_segmentation()
        self.model = get_model_object_detection()
        self.optimizer = set_optimizer(self.model)
        self.scheduler = set_scheduler(self.optimizer)
        # self.loss_func = maskrcnn_loss

        for epoch in range(1, EPOCHS + 1):
            # reset the training and validation loss histories for the current epoch
            train_loss_hist.reset()
            test_loss_hist.reset()

            train_loss_list = train_epoch(epoch, train_ds, self.model, self.optimizer, train_loss_list, train_loss_hist)

            print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")  
            # do metrics measurement here: IOU, AP:IOU > 0.5

            # update learning rate
            self.scheduler.step()

            test_loss_list = test_epoch(epoch, test_ds, self.model, test_loss_list, test_loss_hist)
 
            print(f"Epoch #{epoch} test loss: {test_loss_hist.value:.3f}")   
            # do metrics measurements here: : IOU, AP:IOU > 0.5

            # save best model
            save_best_model(
                test_loss_hist.value, epoch, self.model, self.optimizer
            )
            # save loss plot
            save_loss_plot(train_loss_list, test_loss_list)


# call model and return results
# runner = ModelRunner(batch_size=BATCH_SIZE)
# results = runner.train_and_test()
inference = ModelInference()
evaluator, stats = inference.run()
stats
