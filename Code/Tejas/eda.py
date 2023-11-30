#%%
# Set data directory
import os

IMAGES_DIR = os.path.join("../../../train/train/images/")
ANNOTATIONS = os.path.join("../../../train/train/annotations.json")

#%%
# load and parse annotations file
import json

with open(ANNOTATIONS, "r") as file_json:
    annotations = json.loads(file_json.read())

# %%
print("Annotations JSON keys: ", [key for key in annotations.keys()])

# %%
categories: list[dict] = annotations["categories"].copy()
print("Number of categories: ", len(categories))


# %%
from collections import defaultdict
print("Distribution of images by categories")
categories_by_id = { cat["id"]: cat["name_readable"] for cat in categories }
images_by_cat = defaultdict(int)
for ann in annotations["annotations"]:
    category_id = ann["category_id"]
    category = categories_by_id.get(category_id, None)
    if not category:
        raise KeyError(f"No category found for id {category_id}")
    images_by_cat[category] += 1

images_by_cat
# %%
import pandas as pd
images_by_cat_df = pd.DataFrame(images_by_cat.items()).sort_values(1)
images_by_cat_df.rename(columns={0: "category", 1: "images"}, inplace=True)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 48))  # Adjust the figsize as needed
ax.barh(images_by_cat_df["category"], images_by_cat_df["images"], align="edge", height=0.8, color="skyblue")
ax.set_xlabel('Number of Images')
ax.set_ylabel('Category')
ax.set_title('Number of Images by Category')

plt.xticks(rotation=45, ha='right')
plt.show()

# %%
import numpy as np
import matplotlib.patches as patches
import random

def get_random_color():
    return (np.random.random((1, 3))*0.6+0.4).tolist()[0]

def build_segmentation_masks(annotation, color):
    polygons = []
    segmentation = annotation["segmentation"]
    
    for seg in segmentation:
        mask = np.array(seg).reshape((int(len(seg)/2), 2))
        polygon = patches.Polygon(mask, closed=True, facecolor=color, linewidth=0, alpha=0.5)
        polygons.append(polygon)
    
    return polygons

def build_bbox(annotation, color):
    [bbox_x, bbox_y, bbox_w, bbox_h] = annotation["bbox"]
    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], 
            [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
    np_poly = np.array(poly).reshape((4,2))
    return patches.Polygon(np_poly, closed=True, facecolor='none', edgecolor=color, linewidth=2)

images = annotations["images"].copy()
image_annotations = annotations["annotations"].copy()
images_by_id = { img["id"]: img["file_name"] for img in images }

annotations_by_img_id = defaultdict(list)
for ann in image_annotations:
    image_id = ann["image_id"]
    annotations_by_img_id[image_id].append(ann)

fig, axs = plt.subplots(4, 3, figsize=(10, 10))
axes = axs.flatten()

for i, image_data in enumerate(random.sample(images, k=12)):
    image_id = image_data["id"]
    
    axes[i].set_title(f"{image_id}")
    axes[i].axis('off')

    bbox = ann["bbox"]
    image_file = images_by_id.get(image_id, None)
    if not image_file:
        raise KeyError(f"No image found for image ID {image_id}.")
    
    image = plt.imread(IMAGES_DIR + image_file)
    axes[i].imshow(image)

    img_annotations = annotations_by_img_id.get(image_id, None)
    if not img_annotations:
        raise KeyError(f"No annotations found for image ID {image_id}")
    
    for ann in img_annotations:
        color = get_random_color()
        
        # add segmentation mask
        masks = build_segmentation_masks(ann, color)
        for mask in masks:
            axes[i].add_patch(mask)
        
        # add bounding box
        bbox = build_bbox(ann, color)
        axes[i].add_patch(bbox)

plt.axis('off')
plt.show()

# %%
