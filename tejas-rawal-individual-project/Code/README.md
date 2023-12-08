### Author: Tejas Rawal

# About
These files contain the implementation details for the custom and Faster R-CNN models that I worked on for this project.

*Note*: Since my model isn't actually working (yet), this demo will not display any annotations.

# Training and testing
1. Clone the repo
2. `cd` into the `tejas-rawal-individual-project` folder
3. From the current directory, run `pip install -r requirements.txt`
4. In `Code/fasterRCNN/main.py`, uncomment the type of Faster R-CNN model you'd like to train. The variable `model_name` can be one of either `Pretrained.MOBILE_NET` or `Pretrained.RESNET`.
5. From the current directory, run the following command:
```
python3 Code/fasterRCNN/main.py
```

# Inference 
1. Clone the repo
2. `cd` into the `tejas-rawal-individual-project` folder
3. From the current directory, run `pip install -r requirements.txt`
3. From the current directory, run the following command:
```
python3 Code/fasterRCNN/predictor.py --image_path path/to/img.jpg
```
