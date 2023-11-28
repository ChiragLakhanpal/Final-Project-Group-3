#%%
# Set data directory
import os

IMAGES_DIR = os.path.join("../../train/train/images/")
ANNOTATIONS = os.path.join("../../train/train/annotations.json")

#%%
# load and parse annotations file
import json

with open(ANNOTATIONS, "r") as file_json:
    annotations = json.loads(file_json.read())

# %%
print("Annotations JSON keys: ", [key for key in annotations.keys()])

# %%
print("Image categories")
categories: list[dict] = annotations["categories"].copy()
print("Number of categories: ", len(categories))


# %%
print("Distribution of images by categories")