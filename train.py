import os
import datetime
name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
results = model.train(data="path-to-data-yaml", epochs=300, imgsz=256, pretrained=True, name=name, project="log-path")