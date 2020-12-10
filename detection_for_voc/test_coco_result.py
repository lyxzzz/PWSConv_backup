import datetime
import os
import sys
import time
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import json

sys.path.append('python')
sys.path.append('python/dataloader_coco')

import cfg_loader
import preprocessing_loader
import data_loader
import model_loader
import eval_func
import dataloader_coco.data_loader as COCO_LOADER

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import eval_tools
from utils import tf_utils
from utils.epoch_info_record import EpochRecorder
from utils import file_tools
from utils import progress_bar
import random

coco_root = "/home/lyx/dataset/COCO"
dataType = "val2017"

def process_load_to_numpy(dataset, catids):
    result = []
    random.shuffle(dataset)
    for data in dataset:
        for i in range(len(data[3])):
            result.append([data[-1], data[3][i][0], data[3][i][1], data[3][i][2] - data[3][i][0], data[3][i][3] - data[3][i][1], random.random(), catids[data[4][i]-1]])
    return np.array(result)
anno_file = "{}/annotations/instances_{}.json".format(coco_root, dataType)
cocoGT = COCO(anno_file)
catids = sorted(cocoGT.getCatIds())
dataset = COCO_LOADER.data_loader([coco_root, dataType])
cocoDT = cocoGT.loadRes(process_load_to_numpy(dataset, catids))

# dtcoco = COCO(anno_file)
E = COCOeval(cocoGT,cocoDT, iouType='bbox'); # initialize CocoEval object
E.evaluate();                # run per image evaluation
E.accumulate();              # accumulate per image results
E.summarize();               # display summary metrics of results
print(E.stats)
