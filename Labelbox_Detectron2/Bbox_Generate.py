import os
import json
import uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob   

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


from labelbox.data.metrics import (
    feature_miou_metric, 
    feature_confusion_matrix_metric
)

#with open('./coco_utils.py', 'w' ) as file:
    #helper = requests.get("https://raw.githubusercontent.com/Labelbox/labelbox-python/coco/examples/integrations/detectron2/coco_utils.py").text
    #file.write(helper)
from coco_utils import visualize_coco_examples, visualize_object_inferences, partition_coco


#API_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja2E0ZnJzNjAzcTBoMDg1MWhmcWp4ZWNtIiwib3JnYW5pemF0aW9uSWQiOiJja2IxYnphaXUzMHV2MDc0NHRmb3I3MGZhIiwiYXBpS2V5SWQiOiJja3Zmb2VvMDExenEyMHo3ajUzNzIwdTEzIiwic2VjcmV0IjoiNzYxZmM4OWQ2ZmU1YTU5NDUyMWEyMTUwOGEyZGYzOGYiLCJpYXQiOjE2MzU3MTE1MjIsImV4cCI6MjI2Njg2MzUyMn0.w5XGxygsSnVBt-yDdDKhnSYmsuNbCL0s_LrmaSrCuYQ'
API_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja2FkNGlleDA0cjJ2MDkxOGdneHh4b3VxIiwib3JnYW5pemF0aW9uSWQiOiJja3JtMXN0cG8zdnU2MHo4ZTAxNng2ejRtIiwiYXBpS2V5SWQiOiJjbDhucWE0OGUxc3lrMDd6bTg0N2w3OXgwIiwic2VjcmV0IjoiYjMwYzhlOGM0MzFlYzg5NjdlZGVjMGJjYmQ1NGU4MjgiLCJpYXQiOjE2NjQ0OTY1NjgsImV4cCI6MjI5NTY0ODU2OH0.gf90Q7dsDUdWIOv49UixwZwujtS9TNGCzY1z6YChEvw'
client = Client(api_key = API_KEY)

project_id='cl8s1gueu0l0p070fgrv0chhr'

root='/home/sbaner24/Maize/Data/Trial008_YELStem'
image_root = root+"/temp/images/"
mask_root = root+"/temp/masks/"
train_json_path =root+ '/temp/json_train_annotations.json'
test_json_path = root+'/temp/json_test_annotations.json'
#train_test_split = [0.90, 0.10]
model_zoo_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# These can be set to anything. As long as this process doesn't have 
# another dataset with these name
train_ds_name = "yelstem_train_maize008"
test_ds_name = "yelstem_test_maize008"

#model_name = "detectron_maize_model"


proj = client.get_project(project_id)
#for path in [image_root, mask_root]:
    #if not os.path.exists(path):
        #os.mkdir(path)
        
labels = proj.label_generator()

coco = COCOConverter.serialize_instances(
    labels = labels, 
    image_root = image_root,
    ignore_existing_data=True
)


MetadataCatalog.get(train_ds_name).thing_classes = {r['id'] : r['name'] for r in coco['categories']}

#Clear metadata so detectron recomputes.
if hasattr(MetadataCatalog.get(train_ds_name), 'thing_classes'):
    del MetadataCatalog.get(train_ds_name).thing_classes
if hasattr(MetadataCatalog.get(test_ds_name), 'thing_classes'):
    del MetadataCatalog.get(test_ds_name).thing_classes   

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # Set threshold for this model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_yelstem_008.pth")
cfg.DATASETS.TRAIN = (train_ds_name, )
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
predictor = DefaultPredictor(cfg)


# Last 3 digits of trial number
trial_num = '008'
# Filer address, where all data is located
root = r'/mnt/research-projects/e/ejlobato/assist1data/cyber_plant/SideCam/Trial'+ str(trial_num) + '/Patches'
# Scan name to work with
Dates = ['y20m12d14','y20m12d15','y20m12d16','y20m12d17','y20m12d18','y20m12d19','y20m12d20','y20m12d21','y20m12d22']
# Scan identifier: 1 = Morning Scan, 2 = Afternoon Scan
Scan = '1'
# Plant ID
Plants = ['Plant1','Plant2','Plant3','Plant4']

for Plant in Plants:
    for Date in Dates:
        directory = os.path.join(root, Plant, Date,  'Scans', Scan)
        if not os.path.exists(os.path.join(directory, 'BBys_T'+str(trial_num))):
            print('Creating: {}'.format(os.path.join(directory, 'BBys_T'+str(trial_num))))
            os.makedirs(os.path.join(directory, 'BBys_T'+str(trial_num)))
        out_dir = os.path.join(directory, 'BBys_T'+str(trial_num))
                                                                  # segment leaves from trial images
        x = 0

        finalPredictions = []

        for filename in glob.glob(os.path.join(directory, '*.jpg')):
            if 'Thumbs' in filename:
                print("corrupted file")
            else:
                im = cv2.imread(filename)
                print("Working on image:",filename)
                outputs = predictor(im)
            #     print(type(outputs["instances"].pred_boxes))
                data = []
                data.append(filename)
                data.append(outputs["instances"].pred_classes)
                data.append(outputs["instances"].pred_boxes)
                finalPredictions.append(data)
                
                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite(os.path.join(out_dir, filename[:-4].split('/')[-1] + '.jpg'), out.get_image()[:, :, ::-1])
                
                pred_boxes = directory + '/PREDys_NEW/'
                if not os.path.exists(pred_boxes):
                    os.mkdir(pred_boxes)

                pred_path = os.path.join(pred_boxes+"predictions.txt")

                textfile = open(pred_path, "w")

                for element in finalPredictions:
                    string = str(element[0]) + ','    
                    boxes = element[2].tensor.numpy()
                    #print(len(boxes))
                    if len(boxes) != 0:
                        for i in range(len(boxes)):
                            string += str(element[1].numpy()[i]) + ':' + str(element[2].tensor.numpy()[i]) + ','
                    #print(string) 
                    textfile.write(string + "\n")
                textfile.close()
