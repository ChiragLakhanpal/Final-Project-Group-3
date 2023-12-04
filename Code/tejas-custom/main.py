from enum import Enum
from functools import cache
import json

from src.model import build_model, set_optimizer, set_scheduler, build_custom_criterion
from src.train import get_train_dataset, train_epoch
from src.test import get_test_dataset, test_epoch
from src.params import *

import torch

class Phase(Enum):
    TRAIN = "train"
    TEST = "test"

@cache
def get_raw_annotations(phase):
    annotations_path = (
        TRAIN_ANNOTATIONS_PATH if phase == Phase.TRAIN
        else TEST_IMAGES_PATH
    )

    with open(os.path.join(DATA_DIR, annotations_path), "r") as file_json:
        annotations_data = json.loads(file_json.read())
        return annotations_data


class ModelRunner:
    def __init__(self, batch_size: int=100) -> None:
        self.batch_size = batch_size
        # self.model = build_model()
   
    def train_and_test(self):
        train_annotations = get_raw_annotations(Phase.TRAIN)
        test_annotations = get_raw_annotations(Phase.TEST)
        train_ds = get_train_dataset(train_annotations, self.batch_size)
        test_ds = get_test_dataset(test_annotations, self.batch_size)
        
        self.model = build_model()
        self.optimizer = set_optimizer(self.model)
        self.scheduler = set_scheduler(self.optimizer)
        self.loss_func = build_custom_criterion

        for epoch in range(1, EPOCHS + 1):
            train_epoch(epoch, train_ds, self.model, self.optimizer, self.loss_func)

            # do metrics measurement here

            test_epoch(epoch, test_ds, self.model, self.loss_func )

            # do metrics measurements here
            # IOU
            # hamming metric

            # save best model
            torch.save(self.model.state_dict(), "custom_model.pt")

# if __name__ == "main":
    # call model and return results
runner = ModelRunner(batch_size=BATCH_SIZE)
results = runner.train_and_test()