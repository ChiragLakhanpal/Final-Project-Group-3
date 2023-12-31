# Wellness Wingman

## Introduction
Recognizing food from images is an extremely useful tool for a variety of use cases. In particular, it would allow people to track their food intake by simply taking a picture of what they consume. Food tracking can be of personal interest, and can often be of medical relevance as well. Medical studies have for some time been interested in the food intake of study participants but had to rely on food frequency questionnaires that are known to be imprecise.

## Problem Statement
The goal of this challenge is to train models that can look at images of food items and detect the individual food items present in them. 

## Dataset
### Training
Set of 54,392 (as RGB images) food images, along with their corresponding 100,256 annotations in MS-COCO format
### Test
Set of 946 (as RGB images) food images, along with their corresponding 1708 annotations in MS-COCO format

## Models Trained
- [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)
- [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [YOLOv8](https://github.com/ultralytics/ultralytics)


## Training the Mask R-CNN Model

Follow these steps to train the model using your dataset or to download the dataset automatically.

1. Install the required dependencies:
`pip install -r requirements.txt`

2. Running the Training Script
You can run the training script using one of the two methods below.

  - If you have a dataset:
      Ensure your dataset is structured as follows before proceeding:
    ````
    data/
    ├── train/
    │   ├── annotations.json
    │   └── images/
    ├── val/
    │   ├── annotations.json
    │   └── images/
    └── test/
        └── images/
    ````
    If your dataset is ready, use the following command to start training:
    `python train.py --data-dir <path/to/your/data>`
    Replace /path/to/your/data with the actual path to your dataset.
    
  - If you need to download the dataset:
    If you do not have a dataset and need to download it, simply run the training script without specifying the --data-dir argument. The script will handle the download and preparation of the dataset:
    
## Training the Yolov8 model

The Yolo model is built for 2 different tasks.
* Object Detection
* Instance segmentation

for training
1. Create a directory '/home/ubuntu/term_project/data/' and place the food.yaml file in the data folder.
   
2. For instance segmentation task, Run the Yolov8_instance_segmentation_annotation.py file to update the coco labels to the Yolo Labels
   
3. for the Object detection task, run COCO_to_YOLO_annotations.py file
   
4. based on the task selected, pass the model for training. Comment on the other models and execute the code just by keeping the required model.
   
5. It will generate the runs/segment/train directory, in which, you will get the trained model(it will give the best and the general model)
  
6. keep predict.py in the same directory and run the file to see the output.


## Stremlit App

To run the Streamlit app, use the following command:

```streamlit run streamlit.py --server.port <port number> --server.fileWatcherType none -- --class_to_category <path/to/the/class_to_category.json> --annotations_json <path/to/the/annotations_json.json>```

This command starts the Streamlit server. The application automatically downloads the necessary models from Hugging Face and stores them in the Models directory.

Usage
Navigate to http://localhost:<port> in your web browser to access the app.
Follow the on-screen instructions to upload an image and perform food detection.
