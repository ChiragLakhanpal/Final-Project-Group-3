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
