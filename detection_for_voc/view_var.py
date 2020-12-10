import datetime
import os
import sys
import time
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import json
import math

sys.path.append('python/dataloader_coco')
sys.path.append('python')

import cfg_loader
import preprocessing_loader
import data_loader
import model_loader
import eval_func
import optimizer_config

from utils import eval_tools
from utils import tf_utils
from utils.epoch_info_record import EpochRecorder
from utils import file_tools
from utils import progress_bar

tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('load_difficult', 1, '')

tf.app.flags.DEFINE_string('ckpt_name', None, '')

tf.app.flags.DEFINE_string('last_layer_norm', None, '')

tf.app.flags.DEFINE_boolean('stop_gradient', None, '')

tf.app.flags.DEFINE_string('model', 'tpn', '')

tf.app.flags.DEFINE_float('learning_rate', None, '')

tf.app.flags.DEFINE_boolean('ba_fn', None, '')
tf.app.flags.DEFINE_boolean('ha_fn', None, '')

FLAGS = tf.app.flags.FLAGS

def _summary_mean_var(input, axes, name):
    # print(input)
    mean, var = tf.nn.moments(input, axes=axes)
    
    mean = tf.reduce_mean(mean)
    var = tf.reduce_mean(var)

    mean_name = name + '_mean'
    var_name = name + '_var'
    
    tf.summary.scalar(mean_name, mean)
    tf.summary.scalar(var_name, var)

def make_var_mean_summary(para_list):
    raw_pred, raw_loc = para_list
    layer_len = len(raw_pred)
    raw_pred = tf.concat(raw_pred, axis=1)
    raw_loc = tf.concat(raw_loc, axis=1)
    neg_pred = tf.reshape(raw_pred[:,:,0], [-1])
    pos_pred = tf.reshape(raw_pred[:,:,1:], [-1, 20])
    loc_pred = tf.reshape(raw_loc, [-1, 4])

    _summary_mean_var(neg_pred, 0, 'batch/neg')
    _summary_mean_var(pos_pred, 0, 'batch/pos')
    _summary_mean_var(loc_pred, 0, 'batch/loc')

    _summary_mean_var(raw_pred, [1,2], 'batch/total')
    
def __parser_cmd_to_json(var, json_dict, name):
    if var is not None:
        json_dict[name] = var

def main(argv=None):
    config_path = os.path.join('train_cfgs', FLAGS.model+'.json')
    with open(config_path, 'r') as json_file:
        start_cfg_dict = json.load(json_file)
    TRAIN_PARAMETERS = start_cfg_dict['train_parameters']
    TEST_PARAMETERS = start_cfg_dict['test_parameters']
    RESTORE_PARAMETERS = start_cfg_dict['restore_parameters']
    DATASET_PARAMETERS = start_cfg_dict['dataset']
    BACKBONE_PARAMETERS = start_cfg_dict['backbone']
    NETWORK_PARAMETERS = start_cfg_dict['network']
    HEADER_PARAMETERS = start_cfg_dict['header']
    LOSSES_PARAMETERS = start_cfg_dict['losses']
    POSTPROCESSING_PARAMETERS = start_cfg_dict['postprocessing']

    __parser_cmd_to_json(FLAGS.ckpt_name, TRAIN_PARAMETERS, 'ckpt_name')
    __parser_cmd_to_json(FLAGS.last_layer_norm, HEADER_PARAMETERS, 'last_layer_norm')
    __parser_cmd_to_json(FLAGS.stop_gradient, NETWORK_PARAMETERS, 'stop_gradient')
    __parser_cmd_to_json(FLAGS.ba_fn, NETWORK_PARAMETERS, 'backbone_activation')
    __parser_cmd_to_json(FLAGS.ha_fn, NETWORK_PARAMETERS, 'header_activation')
    if FLAGS.learning_rate is not None:
        for i in range(len(TRAIN_PARAMETERS['learning_rate'])):
            TRAIN_PARAMETERS['learning_rate'][i] = TRAIN_PARAMETERS['learning_rate'][i] * FLAGS.learning_rate
    

    ROOT_CFG = cfg_loader.get_cfgs(start_cfg_dict.get('default_network_cfgs','emptyCFG'), start_cfg_dict)
    
    gpu_id = int(start_cfg_dict['gpuid'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    now = datetime.datetime.now()
    # StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    StyleTime = now.strftime("%Y-%m-%d")

    file_tools.touch_dir(TRAIN_PARAMETERS['logs_path'] + StyleTime)
    file_tools.touch_dir(TRAIN_PARAMETERS['ckpt_path'])

    preload_train_dataset, obj_type_nums = data_loader.get_train_dataset(DATASET_PARAMETERS['train'], FLAGS.load_difficult)
    train_data_size = len(preload_train_dataset)

    prepare_data = preprocessing_loader.prepare_before_model_construct('train', ROOT_CFG)
    anchor_list = prepare_data[-1]

    with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model') as scope:
            model_outputs = model_loader.forward(ROOT_CFG,
                                                obj_type_nums,
                                                prepare_data[0],
                                                prepare_data[1],
                                                backbone_name=BACKBONE_PARAMETERS['type'],
                                                header_name=HEADER_PARAMETERS['type'])
            


            loss_list = model_loader.losses(ROOT_CFG,
                                            model_outputs,
                                            prepare_data,
                                            loss_name=LOSSES_PARAMETERS['type'])

            post_outputs = model_loader.postprocessing(ROOT_CFG,
                                                        obj_type_nums,
                                                        anchor_list,
                                                        model_outputs,
                                                        postprocessing_name=POSTPROCESSING_PARAMETERS['type'])

    var_list = tf.trainable_variables()
    while True:
        name = input()
        if name == "exit":
            break
        else:
            for var in var_list:
                if name in var.name:
                    print(var.name)

    # for name in var_to_shape_map2.keys():
    # if "block7/conv1/rate_map" in name and "Momentum" not in name and "ExponentialMovingAverage" not in name:
    #     print(name)
    #     val = reader.get_tensor(name)
    #     print(val.shape)
    #     if len(val.shape) == 4:
    #         normalization(val)
    #         # print(val[0, 0])
    #     else:
    #         print(val)

if __name__ == '__main__':
    tf.app.run()
