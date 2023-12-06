
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import pandas as pd
import cv2
import json
from tqdm.notebook import tqdm
import subprocess
import time
from pathlib import Path
from detectron2.data import transforms as T

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.structures import Boxes, BoxMode
import pycocotools.mask as mask_util
import copy
import detectron2.data.detection_utils as utils
from detectron2.data import DatasetCatalog
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import detectron2.data.detection_utils as utils
import copy
import torch
import numpy as np

import os
import json
import importlib
import numpy as np
import cv2
import torch
from detectron2.engine import DefaultPredictor

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, BoxMode
from detectron2.config import get_cfg
import pycocotools.mask as mask_util

from pycocotools.coco import COCO

from pprint import pprint 
from collections import OrderedDict
import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"
TRAIN_IMAGE_DIRECTIORY = "data/train/images/"

VAL_ANNOTATIONS_PATH = "data/val/annotations.json"
VAL_IMAGE_DIRECTIORY = "data/val/images/"

train_coco = COCO(TRAIN_ANNOTATIONS_PATH)

with open(TRAIN_ANNOTATIONS_PATH) as f:
  train_annotations_data = json.load(f)

with open(VAL_ANNOTATIONS_PATH) as f:
  val_annotations_data = json.load(f)


category_ids = train_coco.loadCats(train_coco.getCatIds())
category_names = [_["name_readable"] for _ in category_ids]

no_images_per_category = {}

for n, i in enumerate(train_coco.getCatIds()):
  imgIds = train_coco.getImgIds(catIds=i)
  label = category_names[n]
  no_images_per_category[label] = len(imgIds)

img_info = pd.DataFrame(train_coco.loadImgs(train_coco.getImgIds()))
no_images_per_category = OrderedDict(sorted(no_images_per_category.items(), key=lambda x: -1*x[1]))

from fastai.vision.core import *
from fastai.vision.utils import *
from fastai.vision.augment import *
from fastai.data.core import *
from fastai.data.transforms import *

images, lbl_bbox = get_annotations('data/train/annotations.json')

idx=14
coco_fn,bbox = 'data/train/images/'+images[idx],lbl_bbox[idx]

def _coco_bb(x):  return TensorBBox.create(bbox[0])
def _coco_lbl(x): return bbox[1]

coco_dsrc = Datasets([coco_fn]*10, [PILImage.create, [_coco_bb,], [_coco_lbl, MultiCategorize(add_na=True)]], n_inp=1)
coco_tdl = TfmdDL(coco_dsrc, bs=9, after_item=[BBoxLabeler(), PointScaler(), ToTensor(), Resize(256)],
                  after_batch=[IntToFloatTensor(), *aug_transforms()])

coco_tdl.show_batch(max_n=9)


class DataPreprocessor:
    def __init__(self, train_annotations_path, val_annotations_path, image_directory):
        self.train_annotations_path = train_annotations_path
        self.val_annotations_path = val_annotations_path
        self.image_directory = image_directory
        self.load_data()

    def load_data(self):
        with open(self.train_annotations_path) as f:
            self.train_annotations_data = json.load(f)
        with open(self.val_annotations_path) as f:
            self.val_annotations_data = json.load(f)

    def update_image_dimensions(self, annotations_data):
        for image_data in tqdm(annotations_data['images'], desc='Updating image dimensions'):
            file_path = os.path.join(self.image_directory, image_data["file_name"])
            if os.path.exists(file_path):
                img = cv2.imread(file_path)
                image_data['height'], image_data['width'] = img.shape[:2]
        return annotations_data

    def remove_rotated_annotations(self, annotations_data):
        annotations_data['annotations'] = [annotation for annotation in annotations_data['annotations'] if
                                           annotation['bbox_mode'] != BoxMode.XYWHA_ABS]
        return annotations_data
      
    def adjust_bounding_boxes(self, annotations_data):
        for annotation in tqdm(annotations_data['annotations'], desc='Adjusting bounding boxes'):
            if annotation['bbox_mode'] == BoxMode.XYWH_ABS:
                annotation['bbox_mode'] = BoxMode.XYXY_ABS
                x, y, w, h = annotation['bbox']
                annotation['bbox'] = [x, y, x + w, y + h]
        return annotations_data
    
    def remove_non_food_items(self, annotations_data):
        annotations_data['annotations'] = [annotation for annotation in annotations_data['annotations'] if
                                           annotation['category_id'] in self.food_category_ids]
        return annotations_data
      

    def save_data(self, annotations_data, path):
        with open(path, 'w') as f:
            json.dump(annotations_data, f)

    def process_data(self):
        self.train_annotations_data = self.update_image_dimensions(self.train_annotations_data)
        self.train_annotations_data = self.remove_rotated_annotations(self.train_annotations_data)
        self.train_annotations_data = self.adjust_bounding_boxes(self.train_annotations_data)
        
        self.val_annotations_data = self.update_image_dimensions(self.val_annotations_data)
        self.val_annotations_data = self.remove_rotated_annotations(self.val_annotations_data)
        self.val_annotations_data = self.adjust_bounding_boxes(self.val_annotations_data)

        self.save_data(self.train_annotations_data, 'new_ann_train.json')
        self.save_data(self.val_annotations_data, 'new_ann_val.json')


from fastai.vision.augment import aug_transforms, RandomErasing


transforms = aug_transforms(
    max_rotate=30, 
    max_zoom=1.5,  
    max_lighting=0.4,
    max_warp=0.4,  
    p_affine=0.75, 
    p_lighting=0.75, 
    xtra_tfms=[RandomErasing(p=0.5, max_count=2)] 
)

coco_tdl = TfmdDL(
    coco_dsrc, 
    bs=9, 
    after_item=[BBoxLabeler(), PointScaler(), ToTensor(), Resize(256)],
    after_batch=[IntToFloatTensor(), *transforms]
)

coco_tdl.show_batch(max_n=9)

  
np.array(train_annotations_data['annotations'][2]['segmentation']).shape , np.array(train_annotations_data['annotations'][2]['bbox']).shape

def fix_data(annotations, directiory, VERBOSE = False):
  for n, i in enumerate(tqdm((annotations['images']))):

      img = cv2.imread(directiory+i["file_name"])

      if img.shape[0] != i['height']:
          annotations['images'][n]['height'] = img.shape[0]
          if VERBOSE:
            print(i["file_name"])
            print(annotations['images'][n], img.shape)

      if img.shape[1] != i['width']:
          annotations['images'][n]['width'] = img.shape[1]
          if VERBOSE:
            print(i["file_name"])
            print(annotations['images'][n], img.shape)

  return annotations

train_annotations_data = fix_data(train_annotations_data, TRAIN_IMAGE_DIRECTIORY)

with open('data/train/new_ann.json', 'w') as f:
    json.dump(train_annotations_data, f)

val_annotations_data = fix_data(val_annotations_data, VAL_IMAGE_DIRECTIORY)

with open('data/val/new_ann.json', 'w') as f:
    json.dump(val_annotations_data, f)



train_annotations_path = 'data/train/new_ann.json'
train_images_path = 'data/train/images'

val_annotations_path = 'data/val/new_ann.json'
val_images_path = 'data/val/images'

if "training_dataset" not in DatasetCatalog:

  register_coco_instances("training_dataset", {},train_annotations_path, train_images_path)
  register_coco_instances("validation_dataset", {},val_annotations_path, VAL_IMAGE_DIRECTIORY)


MODEL_ARCH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(MODEL_ARCH))

cfg.DATASETS.TRAIN = ("training_dataset",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 8

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ARCH)


cfg.SOLVER.IMS_PER_BATCH = 8     

cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.MAX_ITER = 50000


cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

cfg.SOLVER.CHECKPOINT_PERIOD = 20000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 498


cfg.OUTPUT_DIR = "logs_detectron2_r50"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

RESUME = False


trainer = DefaultTrainer(cfg)

if RESUME:
  trainer.resume_or_load(resume=True)
else:
  trainer.resume_or_load(resume=False)


trainer.train()


os.system("cp logs_detectron2_r50/model_final.pth model_final.pth")

cfg.MODEL.WEIGHTS = 'model_final.pth'

evaluator = COCOEvaluator("validation_dataset", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "validation_dataset")
valResults = inference_on_dataset(trainer.model, val_loader, evaluator)

print("Validation mAP: ", valResults["bbox"]["AP"])

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 498

cfg.DATASETS.TEST = ("validation_dataset", )
predictor = DefaultPredictor(cfg)

val_metadata = MetadataCatalog.get("val_dataset")

image_id = '183370'
im = cv2.imread(f"data/val/images/{image_id}.jpg")
outputs = predictor(im)

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
classes = outputs["instances"].pred_classes.cpu().numpy()
scores = outputs["instances"].scores.cpu().numpy()

im_pil = Image.fromarray(im)

fig = go.Figure()

fig.add_trace(go.Image(z=im_pil))

for box, cls, score in zip(boxes, classes, scores):
    y1, x1, y2, x2 = box
    fig.add_shape(
        type="rect",
        x0=x1, y0=y1, x1=x2, y1=y2,
        line=dict(color="Red"),
    )
    fig.add_trace(go.Scatter(
        x=[x1], y=[y1],
        text=[f"{cls} ({score:.2f})"],
        mode="text",
        textposition="bottom center"
    ))

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    width=im.shape[1],
    height=im.shape[0],
)

fig.show()

category_ids = sorted(train_coco.getCatIds())
categories = train_coco.loadCats(category_ids)

class_to_category = { int(class_id): int(category_id) for class_id, category_id in enumerate(category_ids) }

with open("class_to_category.json", "w") as fp:
  json.dump(class_to_category, fp)


test_images_dir = "data/test/images"
output_filepath = "predictions_detectron2.json"

model_path = 'model_final.pth'

threshold = 0.1


class_to_category = {}
with open("class_to_category.json") as fp:
    class_to_category = json.load(fp)
