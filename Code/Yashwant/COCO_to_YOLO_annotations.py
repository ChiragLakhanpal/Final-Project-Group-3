from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import json
import urllib
import PIL.Image as Image
import cv2
import torch
import torchvision
from IPython.display import display
from sklearn.model_selection import train_test_split

# !pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# !pip install numpy==1.17
# !pip install PyYAML==5.3.1
# !pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
os.system('pip install pycocotools')
from pycocotools.coco import COCO

import pandas as pd

# %matplotlib inline
# %config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 16, 10

#np.random.seed(42)

# Reading annotations.json
TRAIN_ANNOTATIONS_PATH = "./data/train/annotations.json"
TRAIN_IMAGE_DIRECTIORY = "./data/train/images/"

VAL_ANNOTATIONS_PATH = "./data/val/annotations.json"
VAL_IMAGE_DIRECTIORY = "./data/val/images/"

train_coco = COCO(TRAIN_ANNOTATIONS_PATH)
val_coco = COCO(VAL_ANNOTATIONS_PATH)

# Reading the annotation files
with open(TRAIN_ANNOTATIONS_PATH) as f:
    train_annotations_data = json.load(f)

train_annotations_data['annotations'][0]

training_info = []

for a in train_annotations_data['annotations']:
  training_info.append(a)
training_info[0]

categories = []

for a in train_annotations_data['annotations']:
  categories.append(a['category_id'])
categories = list(set(categories))
categories.sort()
categories

img_info = training_info[20]
img_id = str(img_info['image_id'])
img = Image.open(TRAIN_IMAGE_DIRECTIORY + img_id + '.jpg')
img = img.convert('RGB')
#plt.imshow(img)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

####checking annotation for solo image

img_info = pd.DataFrame(train_coco.loadImgs(train_coco.getImgIds()))
i = 70
annIds = train_coco.getAnnIds(imgIds=img_info['id'][i])
anns = train_coco.loadAnns(annIds)
print(anns)
img = plt.imread(TRAIN_IMAGE_DIRECTIORY+img_info['file_name'][i])
[x,y,w,h] = anns[0]['bbox']
#create rectagle bbox of size given in dataset
cv2.rectangle(img, (int(x), int(y)), (int(x+h), int(y+w)), (255,0,0), 2)
plt.imshow(img)

# !pip install pathlib
from pathlib import Path


def create_Yolo_labels(coco, data_type):
    labels_path = Path(f"./data/{data_type}/Yolo_labels")

    labels_path.mkdir(parents=True, exist_ok=True)

    img_info = pd.DataFrame(coco.loadImgs(coco.getImgIds()))
    print(labels_path)
    for i in tqdm(range(0, len(img_info))):
        annIds = coco.getAnnIds(imgIds=img_info['id'][i])
        anns = coco.loadAnns(annIds)
        img_id = anns[0]['image_id']
        label_name = f"{img_id}.txt"

        with (labels_path / label_name).open(mode="w") as label_file:

            for j in range(0, len(anns)):
                category_idx = anns[j]['category_id']
                [x, y, bbox_width, bbox_height] = anns[j]['bbox']

                label_file.write(
                    f"{category_idx} {x + bbox_width / 2} {y + bbox_height / 2} {bbox_width} {bbox_height}\n"
                )

create_Yolo_labels(train_coco,'train')
create_Yolo_labels(val_coco,'val')




