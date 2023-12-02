import streamlit as st
import requests
import tempfile
import os
import cv2
from detectron2.utils.visualizer import Visualizer
from PIL import Image
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
import json
import re

if os.path.isfile('class_to_category.json') and os.path.isfile("data/train/annotations.json"):
    with open('class_to_category.json') as f:
        class_to_category = json.load(f)
    with open("data/train/annotations.json") as f:
        annotations = json.load(f)
else:
    class_to_category = {}    

cfg = get_cfg()
config_path = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = 'model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 498
predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get("training_dataset")
  
def get_calorie_info(food_item):
    API_URL = f'https://api.api-ninjas.com/v1/nutrition?query={food_item}'
    API_KEY = 't10Uf0U2FaPHov7a1++GSw==1iTzjwBudd4SFs8n' 
    headers = {'X-Api-Key': API_KEY}
    response = requests.get(API_URL, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def format_names(name):
    name = re.sub(r'[^\w\s]', '', name)  
    name = re.sub(r'\s+', ' ', name) 
    name = re.sub(r'_', ' ', name)  
    name = name.strip()  
    name = name.lower()  
    name = re.sub(r'-+', ' ', name)  
    name = ' '.join(name.split()[::-1])  
    return name

def predict_and_visualize(image_path, predictor, metadata):
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return []

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read the image file {image_path}")
        return []

    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    annotated_img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    
    st.image(annotated_img, caption='Detected Image.', use_column_width=True)    

    classes = outputs["instances"].pred_classes.cpu().numpy()

    category_id = list(set(class_to_category.get(str(i)) for i in classes))
    class_names = [category["name"] for category_id in category_id for category in annotations["categories"] if category["id"] == category_id]
    formatted_names = [format_names(name) for name in class_names]

    return formatted_names 

def save_uploaded_file(uploaded_file):
    try:
        temp_dir = tempfile.mkdtemp()  
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def main():
    inject_custom_css()

    tab1, tab2, tab3 = st.tabs(["Demo", "Presentation", "Connect with Us"])


    with tab1:
        st.title("Food Item Detector and Calorie Estimator")

        st.write("## Description")
        st.write("This app detects food items in an image and provides an estimated calorie count.")

        st.write("## Steps")
        st.write("1. Upload an image of the food.")
        st.write("2. Choose a detection model.")
        st.write("3. Wait for the app to detect the food items.")
        st.write("4. View the detected items and their estimated calorie content.")

        model_choice = st.sidebar.selectbox("Choose a Detection Model", ["YOLO", "Detectron", "Custom"])

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("Detecting...")
            image_path = save_uploaded_file(uploaded_file)

            if image_path:
                if model_choice == "YOLO":
                    detected_items = yolo_predict(image_path)
                elif model_choice == "Detectron":
                    detected_items = predict_and_visualize(image_path, predictor, metadata)
                elif model_choice == "Custom":
                    detected_items = custom_predict(image_path)

                st.write("## Detected Items")
                st.write(f"Found {len(detected_items)} items:")

                for item in detected_items:
                    nutrition_data = get_calorie_info(item)
                    if nutrition_data:
                        with st.expander(f"{item.capitalize()} Nutrition Facts"):
                            col1, col2 = st.columns(2)
                            for food_info in nutrition_data:
                                with col1:
                                    st.markdown("**Nutrient**")
                                    for key in food_info:
                                        st.markdown(f"*{key.capitalize().replace('_', ' ')}:*")
                                with col2:
                                    st.markdown("**Value**")
                                    for value in food_info.values():
                                        st.markdown(f"{value}")
                    else:
                        st.write(f"No nutrition data available for {item}")


    with tab2:
        st.write("## Presentation")

    with tab3:
        st.write("## Connect with Us")
        with st.form("contact_form"):
            st.write("Feel free to reach out to us!")
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message")
            submit_button = st.form_submit_button("Submit")

        st.write("### Socials")

        st.markdown('<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.8.1/css/all.css">', unsafe_allow_html=True)

        linkedin_icon = "<i class='fab fa-linkedin'></i>"
        github_icon = "<i class='fab fa-github'></i>"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Chirag Lakhanpal")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/chiraglakhanpal/)", unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/ChiragLakhanpal)", unsafe_allow_html=True)
        with col2:
            st.markdown("#### Yashwant Bhaidkar")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/yashwant-bhaidkar/)", unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/yashwant2304)", unsafe_allow_html=True)
        with col3:
            st.markdown("#### Tejas Rawal")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/tejasrawal)", unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/tejas-rawal)", unsafe_allow_html=True)

def inject_custom_css():
    custom_css = """
        <style>
            /* General styles */
            html, body {
                font-family: 'Avenir', sans-serif;
            }

            /* Specific styles for titles and headings */
            h1, h2, h3, h4, h5, h6, .title-class  {
                color: #C72C41; 
            }
            a {
                color: #FFFFFF;  
            } 
            /* Styles to make tabs equidistant */
            .stTabs [data-baseweb="tab-list"] {
                display: flex;
                justify-content: space-around; 
                width: 100%; 
            }

            /* Styles for individual tabs */
            .stTabs [data-baseweb="tab"] {
                flex-grow: 1; 
                display: flex;
                justify-content: center; 
                align-items: center; 
                height: 50px;
                white-space: pre-wrap;
                background-color: #C72C41; 
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
                font-size: 90px; 
            }

            /* Styles for the active tab to make it stand out */
            .stTabs [aria-selected="true"] {
                background-color: #EE4540 !important; 
                color: #0E1117 !important; 
                font-weight: bold !important; 
            }
            /* Styles for the tab hover*/
            .stTabs [data-baseweb="tab"]:hover {
                color: #0E1117 !important; 
                font-weight: bold !important; 
            }
               
        </style>    
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
