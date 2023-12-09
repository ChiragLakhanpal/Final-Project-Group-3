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
from pathlib import Path
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

#train_annotations_data['annotations'][0]

id = []
name = []
for i in train_annotations_data['categories']:
  id.append(i['id'])
  name.append(i['name'])
df_train = pd.DataFrame(list(zip(id, name)),
               columns =['ID', 'name'])

cat_name_mapping = df_train.set_index('ID')['name'].to_dict()

sorted_cat_name_mapping = dict(sorted(cat_name_mapping.items()))
sorted_cat_name_mapping

sorted_train_df = pd.DataFrame(sorted_cat_name_mapping.items(), columns=['ID', 'Name'])
sorted_train_df['class'] = sorted_train_df.index
sorted_train_df.tail()

sorted_train_df['class'] = sorted_train_df.index
sorted_train_df.head()


train_cat_class_mapping = sorted_train_df.set_index('ID')['class'].to_dict()


training_info = []

for a in train_annotations_data['annotations']:
  training_info.append(a)


categories = []

for a in train_annotations_data['annotations']:
  categories.append(a['category_id'])
categories = list(set(categories))
categories.sort()




def create_Yolo_segment_labels(coco, data_type, mapping_list):
    labels_path = Path(f"./food/labels/{data_type}")
    # labels_path = Path(f"./food/train_test")
    labels_path.mkdir(parents=True, exist_ok=True)
    # counter = 0
    img_info = pd.DataFrame(coco.loadImgs(coco.getImgIds()))
    print(labels_path)
    for i in tqdm(range(0, len(img_info))):
        annIds = coco.getAnnIds(imgIds=img_info['id'][i])
        anns = coco.loadAnns(annIds)
        img_id = anns[0]['image_id']
        label_name = f"{img_id}.txt"
        width = img_info.query('id=={}'.format(img_id))["width"].values
        width = width[0]
        height = img_info.query('id=={}'.format(img_id))["height"].values
        height = height[0]
        # print(anns[0]['segmentation'][0])
        with (labels_path / label_name).open(mode="w") as label_file:

            for j in range(0, len(anns)):  # len(anns)
                cat_ID = anns[j]['category_id']
                category_idx = str(mapping_list[cat_ID])
                list_seg = anns[j]['segmentation'][0]
                # print(anns[j]['segmentation'][0])

                for i in range(0, len(list_seg)):
                    if i % 2 == 0:
                        val = list_seg[i] / width
                    else:
                        val = list_seg[i] / height
                    category_idx = category_idx + ' ' + str(val)
                category_idx = category_idx + "\n"

                label_file.write(
                    category_idx
                )

create_Yolo_segment_labels(train_coco,'train',train_cat_class_mapping)

create_Yolo_segment_labels(val_coco,'val',train_cat_class_mapping)

img_info = pd.DataFrame(train_coco.loadImgs(train_coco.getImgIds()))
img_info.head()






