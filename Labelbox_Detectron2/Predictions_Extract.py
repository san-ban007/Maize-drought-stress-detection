from skimage.io import imread, imshow
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import cv2
import requests
import numpy as np
from PIL import Image
import os


import os
import json
import uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm 
from shapely.geometry import MultiPolygon
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

from labelbox import Client, LabelingFrontend, OntologyBuilder
from labelbox.data.serialization import COCOConverter, NDJsonConverter
from labelbox.schema.model import Model
from labelbox.data.metrics.group import get_label_pairs
from labelbox.data.annotation_types import (
    Mask, 
    MaskData, 
    ObjectAnnotation, 
    LabelList, 
    Point, 
    Rectangle, 
    Polygon, 
    ImageData, 
    Label,
    ScalarMetric
)


def image_processing(image, top, left, right, bottom):
    image[:top] = [255, 255, 255]
    image[:, :left] = [255, 255, 255]
    image[:,right:] = [255, 255, 255]
    image[bottom:] = [255, 255, 255]
    return image

def all_white(image):
    image[:] = [255, 255, 255]
    return image

def color_segmentation(image):
    im = rgb2hsv(image)

#     fig, ax = plt.subplots(1, 3, figsize=(12,4))
#     ax[0].imshow(im[:,:,0], cmap='gray')
#     ax[0].set_title('Hue')
#     ax[1].imshow(im[:,:,1], cmap='gray')
#     ax[1].set_title('Saturation')
#     ax[2].imshow(im[:,:,2], cmap='gray')
#     ax[2].set_title('Value')

    lower_mask = im[:,:,0] > 0.5
    #refer to hue channel (in the colorbar)
    upper_mask = im[:,:,0] < 0.9
    #refer to transparency channel (in the colorbar)
    saturation_mask = im[:,:,1] > 0.5

    mask = upper_mask*lower_mask*saturation_mask
    red = image[:,:,0]*mask
    green = image[:,:,1]*mask
    blue = image[:,:,2]*mask
    bags_masked = np.dstack((red,green,blue))
    # print(type(bags_masked))
    #imshow(bags_masked)
    return bags_masked





# Last 3 digits of trial number
trial_num = '005'
# Filer address, where all data is located
root = r'/mnt/research-projects/e/ejlobato/assist1data/cyber_plant/SideCam/Trial'+ str(trial_num) + '/Patches'
Dates = ['y20m09d27','y20m09d28','y20m09d29','y20m09d30','y20m10d01','y20m10d02','y20m10d03','y20m10d04','y20m10d05','y20m10d06','y20m10d07','y20m10d08','y20m10d09','y20m10d10','y20m10d11','y20m10d12','y20m10d13']
Scan = '1'
# Plant ID
Plants = ['Plant1','Plant2','Plant3','Plant4']
for Plant in Plants:
    for Date in Dates:
        directory = os.path.join(root, Plant, Date,  'Scans', Scan)
        print("Working on:",directory)
        yel_boxes = directory + '/YELys_NEW/'
        if not os.path.exists(yel_boxes):
            os.mkdir(yel_boxes)
        pred_boxes = directory + '/PREDys_NEW/'
        pred_path = os.path.join(pred_boxes+"predictions.txt")
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                data = f.readlines()
            for d in data:
                d = d.split(',')
                d.remove('\n')
                if len(d) == 0:
                    continue
                elif len(d) >= 1:
                    image = cv2.imread(d[0])
                    if len(d) == 1:
                        image[:] = [255, 255, 255]
                    elif len(d) > 1:
                        flag = False
                        for i in range(1,len(d)):
                            temp = d[i].split(':')
                            print(temp)
                            if temp[0] == '0':
                                flag = True
                                t = temp[1]
                                t = t[1:-1].split(' ')
                                if '' in t:
                                    t = list(filter(('').__ne__, t))
                                top = int(float(t[1]))
                                left = int(float(t[0]))
                                bottom = int(float(t[3]))
                                right = int(float(t[2]))
                                image[:top] = [255, 255, 255]
                                image[:,:left] = [255, 255, 255]
                                image[:,right:] = [255, 255, 255]
                                image[bottom:] = [255, 255, 255]
                                break
                        if flag == False:
                            image[:] = [255, 255, 255]
                
                    final_image = color_segmentation(image)
                    print(d[0][-8:-4])
                    cv2.imwrite(os.path.join(yel_boxes, str(d[0][-8:-4]) +'.jpg'), final_image)

for Plant in Plants:
    for Date in Dates:
        directory = os.path.join(root, Plant, Date,  'Scans', Scan)
        print("Working on:",directory)
        stem_boxes = directory + '/STEMys_NEW/'
        if not os.path.exists(stem_boxes):
            os.mkdir(stem_boxes)
        pred_boxes = directory + '/PREDys_NEW/'
        pred_path = os.path.join(pred_boxes+"predictions.txt")
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                data = f.readlines()
            for d in data:
                d = d.split(',')
                d.remove('\n')
                if len(d) == 0:
                    continue
                elif len(d) >= 1:
                    image = cv2.imread(d[0])
                    if len(d) == 1:
                        image[:] = [255, 255, 255]
                    elif len(d) > 1:
                        flag = False
                        for i in range(1,len(d)):
                            temp = d[i].split(':')
                            print(temp)
                            if temp[0] == '1':
                                flag = True
                                t = temp[1]
                                t = t[1:-1].split(' ')
                                if '' in t:
                                    t = list(filter(('').__ne__, t))
                                top = int(float(t[1]))
                                left = int(float(t[0]))
                                bottom = int(float(t[3]))
                                right = int(float(t[2]))
                                image[:top] = [255, 255, 255]
                                image[:,:left] = [255, 255, 255]
                                image[:,right:] = [255, 255, 255]
                                image[bottom:] = [255, 255, 255]
                                break
                        if flag == False:
                            image[:] = [255, 255, 255]
                
                    final_image = color_segmentation(image)
                    print(d[0][-8:-4])
                    cv2.imwrite(os.path.join(stem_boxes, str(d[0][-8:-4]) +'.jpg'), final_image)