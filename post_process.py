# province = os.listdir("/home/lisibo/YOLO/res")
import os
from sklearn.cluster import DBSCAN
import numpy as np

def lonlat_2_tile(lon,lat,zoom):
    n = 2 ** zoom
    x = (lon + 180) / 360 * n
    y = (1 - (np.log(np.tan(lat * np.pi / 180) + 1 / np.cos(lat * np.pi / 180)) / np.pi)) / 2 * n
    return x,y

def tile_2_lonlat(x,y,zoom):
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat = np.arctan(np.sinh(np.pi * (1 - 2 * y / n))) * 180 / np.pi
    return lon,lat

province_path = "path-to-detection-results"
# labels: "x_y.png", where x and y are the tile coordinates in zoom level 17
labels = []
# walk through the folder
for root, dirs, files in os.walk(province_path):
    for file in files:
        if file.endswith(".txt"):
            labels.append(file)
print("N labels", len(labels))
xy_lists = []
for l in labels:
    x,y = l.split(".")[0].split("_")
    xy_lists.append([float(x), float(y)])
xy_lists = np.array(xy_lists)

cluster_result = DBSCAN(eps=20, min_samples=10).fit_predict(xy_lists)
noisy_index = np.where(cluster_result == -1)[0]
valid_idx = np.where(cluster_result != -1)[0]

for l in valid_idx:
    label = labels[l]
    cluster = cluster_result[l]
    x,y = label.split(".")[0].split("_")
    lon, lat = tile_2_lonlat(float(y), float(x), 17)
    lonlat_square = [
        tile_2_lonlat(float(y), float(x), 17),
        tile_2_lonlat(float(y), float(x)+1, 17),
        tile_2_lonlat(float(y)+1, float(x)+1, 17),
        tile_2_lonlat(float(y)+1, float(x), 17)
    ]
