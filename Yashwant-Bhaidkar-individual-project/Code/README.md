## Steps to train the YOLOv8 model

1)create a file structure as per below screenshot

 !mkdir -p data/ data/train data/val data/test
 !cp test data/test && cd data/test && echo "Extracting test dataset" && tar -xvf test > /dev/null
 !cp val data/val && cd data/val && echo "Extracting val dataset" &&  tar -xvf val > /dev/null
 !cp train data/train && cd data/train && echo "Extracting train dataset" &&  tar -xvf train > /dev/null

2)Execute the coco-to-YOLO annotations code to convert the labels into YOLO compatible format

3)Create a path '/home/ubuntu/term_project/data/' and keep food.yaml file in the same directtory

4)Choose model from the Yolov8.py file and execute the code

5)It will generate the best.pt file on /runs/segment/train/weights directory.

6)Keep the predict.py in the same directory and pass the image path in predict funtion.

7) Execute python3 predict.py to get the final results(image 'predict.jpg' will get saved with the detected object output)
