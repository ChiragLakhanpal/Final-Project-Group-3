import streamlit as st
import requests
import tempfile
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pycocotools.coco import COCO
from detectron2.data.datasets import register_coco_instances

# Load model configuration and weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "model_final.pth"  # Change to the path of your saved weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 498  # Change this to the number of classes in your dataset

# Registering datasets (if not already registered)
train_annotations_path = 'data/train/new_ann.json'
train_images_path = 'data/train/images'
val_annotations_path = 'data/val/new_ann.json'
val_images_path = 'data/val/images'

if "training_dataset" not in DatasetCatalog:
    register_coco_instances("training_dataset", {}, train_annotations_path, train_images_path)
    register_coco_instances("validation_dataset", {}, val_annotations_path, val_images_path)

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get("training_dataset")

def predict_and_visualize(image_path, predictor, metadata):
    img = cv2.imread(image_path)
    if img is None:
        st.write(f"Error: {image_path} could not be read")
        return None
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    annotated_img = out.get_image()[:, :, ::-1]
    return annotated_img, outputs["instances"]

# Main App
def main():
    st.title("Food Item Detector and Calorie Estimator")
    st.write("This app detects food items in an image and provides an estimated calorie count.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Detecting...")

        # Convert the file to an image and predict
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            annotated_img, instances = predict_and_visualize(tmp_file.name, predictor, metadata)
            if annotated_img is not None:
                st.image(annotated_img, caption='Processed Image.', use_column_width=True)
                for i in instances:
                    st.write(f"Class: {metadata.thing_classes[i.pred_classes]}, Confidence: {i.scores:.2f}")
            else:
                st.write("No items detected or there was an error in processing the image.")

if __name__ == "__main__":
    main()
