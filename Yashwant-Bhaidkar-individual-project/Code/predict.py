from ultralytics import YOLO
model = YOLO("best.pt")
model.predict(source = '/home/ubuntu/term_project/food/images/train/131145.jpg',show = True,save = True,hide_labels = True, hide_conf = True,conf = 0.1,save_txt = True,save_crop = False,line_thickness = 2)
