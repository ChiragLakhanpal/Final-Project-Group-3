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
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from PIL import Image
from ultralytics import YOLO
from fastercnn_predictor import Predictor
import torch

cfg = get_cfg()

metadata = MetadataCatalog.get("training_dataset")

def plotly_images_with_segmentation(image_ids, annotations_data, root_dir, rows=2, cols=6):
    category_id_to_name = {category['id']: category['name_readable'] for category in annotations_data['categories']}
    fig = make_subplots(rows=rows, cols=cols)

    for i, image_id in enumerate(image_ids, start=1):
        img_path = os.path.join(root_dir, f"{image_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = [ann for ann in annotations_data['annotations'] if ann['image_id'] == int(image_id)]
        shapes = []
        category_names = set()
        for ann in annotations:
            category_name = category_id_to_name.get(ann['category_id'], 'Unknown')
            category_names.add(category_name)
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

        fig.layout.annotations[i-1].update(text=f"Image {image_id} ({', '.join(category_names)})")

    fig.update_layout(height=rows * 400, width=cols * 400, showlegend=False)
    return fig
  
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

def predict_and_visualize(image_path, predictor, metadata,class_to_category, annotations):
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

def faster_rcnn_predict(img_path, annotations, categories):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(
        os.path.join('Models', "Model_Faster_RCNN.pt"), map_location=device
    )
    fasterrcnn_predictor = Predictor(categories, annotations)
    image, names = fasterrcnn_predictor(img_path, checkpoint)
    
    st.image(image, caption='Detected Image.', use_column_width=True)

    return [format_names(name) for name in names]
    
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

def yolo_predict(image_path): 
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return []

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read the image file {image_path}")
        return []


    model = YOLO('Models/Model_Yolo.pt')

    results = model(image_path,show = True,save = True,hide_labels = True, hide_conf = True,conf = 0.1,save_txt = True,save_crop = False,line_thickness = 2)  # results list

    # Show the results
    for r in results:
        im_array = r.plot() 
        im = Image.fromarray(im_array[..., ::-1])  

    st.image(im, caption='Detected Image.', use_column_width=True)
    name_list  =[]
    for r in results:

        boxes = r.boxes
        for box in boxes:
            c = box.cls
            val = model.names[int(c)]
            if val not in name_list:
                name_list.append(val)

    return name_list

def main():
    inject_custom_css()
    
    if os.path.isfile('class_to_category.json') and os.path.isfile("data/train/annotations.json"):
        with open('class_to_category.json') as f:
            class_to_category = json.load(f)
        with open("data/train/annotations.json") as f:
            annotations_data = json.load(f)
    else:
        class_to_category = {}
        annotations_data = {"images": [], "annotations": [], "categories": []}
            
    if os.path.isfile('class_to_category.json') and os.path.isfile("data/train/annotations.json"):
        with open('class_to_category.json') as f:
            class_to_category = json.load(f)
        with open("data/train/annotations.json") as f:
            annotations = json.load(f)
    else:
        class_to_category = {}    
        
    tab1, tab2, tab3, tab4 = st.tabs(["Demo", "Presentation", "Explanatory Data Analysis", "Connect with Us"])


    with tab1:
        st.title("Food Item Detector and Calorie Estimator")
        st.write("## Description")
        st.write("This app detects food items in an image and provides an estimated calorie count.")
        st.write("## Steps")
        st.write("1. Upload an image of the food.")
        st.write("2. Choose a detection model.")
        st.write("3. Wait for the app to detect the food items.")
        st.write("4. View the detected items and their estimated calorie content.")

        model_choice = st.sidebar.selectbox("Choose a Detection Model", ["YOLO", "Mask R-CNN", "Faster R-CNN"])

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("Detecting...")
            image_path = save_uploaded_file(uploaded_file)

            if image_path:
                if model_choice == "YOLO":
                    detected_items = yolo_predict(image_path)
                elif model_choice == "Mask R-CNN":
                    config_path = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
                    cfg.merge_from_file(config_path)
                    cfg.MODEL.WEIGHTS = 'Models/Model_Mask_RCNN.pth'
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 498
                    predictor = DefaultPredictor(cfg)                    
                    detected_items = predict_and_visualize(image_path, predictor, metadata, class_to_category, annotations)
                elif model_choice == "Faster R-CNN":
                    detected_items = faster_rcnn_predict(image_path, annotations, class_to_category)

                st.write("## Detected Items")
                # display the detected items through st.write.
                st.write(f"Detected {len(detected_items)} item(s): {', '.join(detected_items)}")


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
        st.write("## Explanatory Data Analysis")
        st.write("### Dataset Glimpse")

        categories_df = pd.DataFrame(annotations_data['categories'])
        images_df = pd.DataFrame(annotations_data['images'])
        annotations_df = pd.DataFrame(annotations_data['annotations'])

        category_id_to_name = categories_df.set_index('id')['name_readable'].to_dict()
        annotations_df['category_name'] = annotations_df['category_id'].apply(lambda x: category_id_to_name.get(x, ''))
        annotations_df['category_name'] = annotations_df['category_name'].apply(lambda x: x.replace('_', ' '))
        annotations_df['category_name'] = annotations_df['category_name'].apply(lambda x: x.capitalize())

        with st.container():

            # Display first few rows from each DataFrame
            st.write("#### Categories")
            st.dataframe(categories_df.head(), width=800)

            st.write("#### Images")
            st.dataframe(images_df.head(), width=800)

            st.write("#### Annotations")
            st.dataframe(annotations_df.head(), width=800)

        # Chart 1: Distribution of images per category
        
        st.write("### Category Distribution")
        top_n = st.selectbox("Select number of top categories to display:", [10, 20, 50, 100, 'All'])

        category_counts = annotations_df['category_name'].value_counts().reset_index(name='counts')
        category_counts.columns = ['Category Name', 'Counts']
        category_counts = category_counts.sort_values('Counts', ascending=False)
        import altair as alt

        if top_n != 'All':
            category_counts = category_counts.head(top_n)
        chart_data = category_counts

        c = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Category Name:N', sort='-y'),
            y='Counts:Q'
        ).properties(
            width=alt.Step(40) 
        )

        st.altair_chart(c, use_container_width=True)

        # Chart 2: Distribution of images per category
        
        st.write("### Distribution of heights and widths of images")

        scatter_chart = alt.Chart(images_df).mark_circle(size=60).encode(
            x=alt.X('width:Q', title='Image Width'),
            y=alt.Y('height:Q', title='Image Height'),
            tooltip=['file_name', 'width', 'height']
        ).interactive().properties(
            width=600,
            height=400
        )

        st.altair_chart(scatter_chart, use_container_width=True)
        
        # Chart 3: Distribution of aspect ratios of images

        st.write("### Distribution of aspect ratios of images")
        
        # Calculate Aspect Ratios
        images_df['aspect_ratio'] = images_df['width'] / images_df['height']

        # Bin aspect ratios into categories
        aspect_ratio_bins = pd.cut(images_df['aspect_ratio'], bins=[0, 0.5, 1, 1.5, 2, np.inf], labels=['<0.5', '0.5-1', '1-1.5', '1.5-2', '>2'])
        aspect_ratio_counts = aspect_ratio_bins.value_counts().reset_index()
        aspect_ratio_counts.columns = ['aspect_ratio', 'count']

        # Adjusted Density Plot for Aspect Ratios with increased bandwidth
        aspect_ratio_density = alt.Chart(images_df).transform_density(
            'aspect_ratio', 
            as_=['aspect_ratio', 'density'],
            bandwidth=0.1,  # Increase the bandwidth for smoother curves
            extent=[0, max(images_df['aspect_ratio'])],  # Adjust the extent if necessary
            groupby=[]
        ).mark_area().encode(
            x='aspect_ratio:Q',
            y='density:Q'
        ).properties(
            width=400,
            height=400
        )

        st.altair_chart(aspect_ratio_density, use_container_width=True)


        # Chart 4: Distribution of annotation counts per image

        st.write("### Distribution of annotation counts per image (log scale)")
                    
        # Calculation of annotation counts
        annotation_counts = annotations_df['image_id'].value_counts().reset_index()
        annotation_counts.columns = ['image_id', 'annotation_count']

        # Adjusting the bin size and using a log scale
        bar_chart = alt.Chart(annotation_counts).mark_bar().encode(
            x=alt.X('annotation_count:Q', bin=alt.Bin(maxbins=50), title='Annotation Count Bins'),
            y=alt.Y('count()', title='Number of Images', scale=alt.Scale(type='log')),
            tooltip=[alt.Tooltip('annotation_count:Q', title='Annotation Count'), alt.Tooltip('count()', title='Number of Images')]
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(bar_chart, use_container_width=True)

        # Chart 5: Distribution of area of annotations per category
        
        st.write("### Distribution of area of annotations per category")
        
        # Calculate the sum of areas per category
        category_area = annotations_df.groupby('category_name')['area'].sum().reset_index()
        category_area.columns = ['category_name', 'total_area']

        fig = px.treemap(category_area, path=['category_name'], values='total_area')
        st.plotly_chart(fig, use_container_width=True)

    
    with tab4:
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
