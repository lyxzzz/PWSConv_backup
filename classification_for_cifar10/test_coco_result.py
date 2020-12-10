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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import eval_tools
from utils import tf_utils
from utils.epoch_info_record import EpochRecorder
from utils import file_tools
from utils import progress_bar


    gtcoco = COCO(gt_ann_file)
    dtcoco = COCO(FLAGS.res_file)
    E = COCOeval(gtcoco,dtcoco, iouType='bbox'); # initialize CocoEval object
    E.evaluate();                # run per image evaluation
    E.accumulate();              # accumulate per image results
    E.summarize();               # display summary metrics of results

    if FLAGS.save_file is not None:
        now = datetime.datetime.now()
        StyleTime = now.strftime("%Y-%m-%d")
        with open(FLAGS.save_file, 'a') as fsave:
            fsave.write('{}\n'.format(StyleTime))
            fsave.write('{}\n'.format(E.__str__()))
