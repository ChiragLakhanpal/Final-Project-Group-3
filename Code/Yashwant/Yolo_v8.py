from ultralytics import YOLO

#with yolov8m
model = YOLO('yolov8m.pt')


# Training.
results = model.train(
   data='/home/ubuntu/term_project/data/food.yaml',
   imgsz=640,
   epochs=1,
   batch=40,
   save=True,
   pretrained = True,
   name='yolov8m_custom',
   plots = True,
   device = 0,
   workers = 8
   )


##with yolov8n
model = YOLO('yolov8n.pt')


# Training.
results = model.train(
   data='/home/ubuntu/term_project/data/food.yaml',
   imgsz=640,
   epochs=1,
   batch=40,
   save=True,
   pretrained = True,
   name='yolov8n_custom',
   plots = True,
   device = 0,
   workers = 8
   )

###without pre trained
# Training.
results = model.train(
   data='/home/ubuntu/term_project/data/food.yaml',
   imgsz=640,
   epochs=1,
   batch=40,
   save=True,
   pretrained = False,
   name='yolov8n_custom',
   plots = True,
   device = 0,
   workers = 8
   )



##########m is training better
##with yolov8m
model = YOLO('yolov8m-seg.pt')


# Training.
results = model.train(
   data='/home/ubuntu/term_project/data/food.yaml',
   imgsz=640,
   epochs=1,
   batch=40,
   save=True,
   pretrained = True,
   name='yolov8m_segmentation',
   plots = True,
   device = 0,
   workers = 8
   )


