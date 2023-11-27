from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import cv2
import matplotlib.pyplot as plt
import random
import os

########################### Hyperparameters ###########################

path_to_test_data = "data/test/images"
path_to_train_data = "data/train/images"
path_to_val_data = "data/val/images"

path_to_train_annotations = "data/train/annotations.json"
path_to_val_annotations = "data/val/annotations.json"

n_epoch = 2
BATCH_SIZE = 128
LR = 0.001

CHANNELS = 1
IMAGE_SIZE = 100

NICKNAME = "adamex"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

########################### Explanatory Data Analysis ###########################

##### Class Distribution #####

pio.templates.default = "plotly_dark"

with open(path_to_train_annotations, 'r') as file:
    data = json.load(file)

category_counts = {category['name']: 0 for category in data['categories']}
for annotation in data['annotations']:
    category_name = next(item for item in data['categories'] if item["id"] == annotation["category_id"])["name"]
    category_counts[category_name] += 1

category_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))
fig = px.bar(x=list(category_counts.keys()), y=list(category_counts.values()), labels={'x': 'Food Category', 'y': 'Count'}, title='Distribution of Food Categories')
fig.layout.template = 'plotly_dark'
fig.update_layout(xaxis_tickangle=-45)
fig.update_layout(title={'x': 0.5,
                            'xanchor': 'center',
                            'y': 0.95,
                            'yanchor': 'top'},
                    font_family='Avenir',
                    font_size=12)

fig.show()

##### Annotation Area Distribution #####

areas = [annotation['area'] for annotation in data['annotations']]

areas.sort()
fig = px.histogram(areas, nbins=50, log_y=True, labels={'value': 'Area (log scale)'}, title='Distribution of Annotation Areas (Logarithmic Scale)')
fig.update_layout(yaxis_title='Count (log scale)')

fig.show()

##### Bounding Box Aspect Ratios #####

aspect_ratios = [bbox[2] / bbox[3] for bbox in (annotation['bbox'] for annotation in data['annotations'])]

fig = px.histogram(aspect_ratios, nbins=50, labels={'value': 'Aspect Ratio'}, title='Distribution of Bounding Box Aspect Ratios')
fig.show()

##### Images with Segmentation #####

from plotly.subplots import make_subplots

def select_random_images(directory, num_images=12):
    all_files = os.listdir(directory)

    jpg_files = [file for file in all_files if file.endswith('.jpg')]

    selected_files = random.sample(jpg_files, min(num_images, len(jpg_files)))

    image_ids = [file.replace('.jpg', '') for file in selected_files]
    return image_ids

def plotly_images_with_segmentation(image_ids, data, root_dir, rows=2, cols=6):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Image {id}" for id in image_ids])

    for i, image_id in enumerate(image_ids, start=1):
        img_path = os.path.join(root_dir, f"{image_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image with ID {image_id} not found.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = [ann for ann in data['annotations'] if ann['image_id'] == int(image_id)]
        shapes = []
        for ann in annotations:
            for segmentation in ann['segmentation']:
                points = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
                shapes.append({
                    'type': 'path',
                    'path': ' M ' + ' L '.join([f'{x} {y}' for x, y in points]) + ' Z',
                    'line': {
                        'color': 'blue',
                        'width': 3,
                    },
                })

        row, col = divmod(i-1, cols)
        fig.add_trace(go.Image(z=image), row=row+1, col=col+1)
        for shape in shapes:
            fig.add_shape(shape, row=row+1, col=col+1)

    fig.update_layout(height=1000, width=1500, showlegend=False)
    fig.show()


root_dir = path_to_train_data
image_ids = select_random_images(root_dir)
plotly_images_with_segmentation(image_ids, data, root_dir)


from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        with open(annotations_file) as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['annotations'])

    def __getitem__(self, idx):
        img_id = self.annotations['annotations'][idx]['image_id']
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert("RGB")

        annotation = self.annotations['annotations'][idx]

        if self.transform:
            image = self.transform(image)

        return image, annotation

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

df_train = FoodDataset('data/train/annotations.json', 'data/train/images', transform)

from torch.utils.data import DataLoader

# Number of workers
num_workers = 8  # Adjust this number based on your machine's capabilities

# DataLoader
data_loader = DataLoader(df_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

