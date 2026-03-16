from ultralytics import YOLO
import os
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

ckpt = "path-to-your-ckpt/best.pt"
model = YOLO(ckpt)

test_folder = "path-to-your-test-folder/"
test_img_path = test_folder + "/images"
test_label_path = test_folder + "/labels"

output_folder = "path-to-your-output-folder"
BEST_THRES=None  ## Use the best threshold from validation set BOX_PR curve
BS = 16

import shutil
shutil.rmtree(output_folder) if os.path.exists(output_folder) else None
os.makedirs(output_folder, exist_ok=True)

# 加载模型
model = YOLO(ckpt)
model.eval()
img_fp = output_folder + "/images/"
label_fp = output_folder + "/labels/"
if not os.path.exists(img_fp):
    os.makedirs(img_fp)
if not os.path.exists(label_fp):
    os.makedirs(label_fp)

t0 = time.time()
results = model(test_img_path, stream=True, conf=BEST_THRES, batch=BS, verbose=False) 
for result in results:
    txt_path = label_fp + result.path.split("/")[-1].replace(".jpg", ".txt")
    with open(txt_path, "w") as f:
        for box in result.boxes: 
            f.write(f"{box.cls.item()} {box.xywhn[0][0].item()} {box.xywhn[0][1].item()} {box.xywhn[0][2].item()} {box.xywhn[0][3].item()}\n")
t1 = time.time()
print(f"Finish inference：{t1 - t0}s")
print("len of results:", len(os.listdir(label_fp)))
## ~40ms/pic @ 3090ti
