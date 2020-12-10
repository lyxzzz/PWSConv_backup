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


tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('load_difficult', 1, '')

tf.app.flags.DEFINE_string('ckpt_path', None, '')
tf.app.flags.DEFINE_string('ckpt_name', None, '')
tf.app.flags.DEFINE_string('postprocessing', None, '')

tf.app.flags.DEFINE_string('save_file', None, '')

tf.app.flags.DEFINE_string('last_layer_norm', None, '')
tf.app.flags.DEFINE_string('conv_type', None, '')
tf.app.flags.DEFINE_string('norm_type', None, '')

tf.app.flags.DEFINE_string('exp_name', None, '')
tf.app.flags.DEFINE_string('model', 'ssd', '')
tf.app.flags.DEFINE_integer('gpuid', 1, '')

FLAGS = tf.app.flags.FLAGS

def __parser_cmd_to_json(var, json_dict, name):
    if var is not None:
        json_dict[name] = var

def build_coco_annFile(ID_TO_TYPE, boxs, labels, scores, basename, shape):
    img_id = int(basename[:-4])
    img_h, img_w = shape

    result = []
    for index, b in enumerate(boxs):
        ann_id = ID_TO_TYPE[labels[index]]
        x = b[0] * img_w
        y = b[1] * img_h
        w = (b[2] - b[0]) * img_w
        h = (b[3] - b[1]) * img_h
        ann_id = TYPE_TO_ID[labels[index]]
        x = b[0] * img_w
        y = b[1] * img_h
        w = (b[2] - b[0]) * img_w
        h = (b[3] - b[1]) * img_h
        result.append(img_id, x, y, w, h, scores[index], ann_id)
    
    return result


def main(argv=None):
    config_path = os.path.join('test_cfgs', FLAGS.model+'.json')
    with open(config_path, 'r') as json_file:
        start_cfg_dict = json.load(json_file)

    TEST_PARAMETERS = start_cfg_dict['test_parameters']
    DATASET_PARAMETERS = start_cfg_dict['dataset']
    BACKBONE_PARAMETERS = start_cfg_dict['backbone']
    HEADER_PARAMETERS = start_cfg_dict['header']
    NETWORK_PARAMETERS = start_cfg_dict['network']
    POSTPROCESSING_PARAMETERS = start_cfg_dict['postprocessing']

    if FLAGS.ckpt_path is not None:
        TEST_PARAMETERS['ckpt_path'] = FLAGS.ckpt_path
    
    if FLAGS.ckpt_name is not None:
        TEST_PARAMETERS['ckpt_name'] = FLAGS.ckpt_name

    if FLAGS.postprocessing is not None:
        POSTPROCESSING_PARAMETERS['type'] = FLAGS.postprocessing

    if FLAGS.last_layer_norm is not None:
        HEADER_PARAMETERS['last_layer_norm'] = FLAGS.last_layer_norm

    __parser_cmd_to_json(FLAGS.last_layer_norm, HEADER_PARAMETERS, 'last_layer_norm')
    __parser_cmd_to_json(FLAGS.conv_type, NETWORK_PARAMETERS, 'conv_type')
    __parser_cmd_to_json(FLAGS.norm_type, NETWORK_PARAMETERS, 'norm_func')

    file_tools.touch_dir(TEST_PARAMETERS['anno_path'])

    ROOT_CFG = start_cfg_dict
    cfg_loader.get_cfgs(ROOT_CFG)

    
    gpu_id = int(start_cfg_dict['gpuid'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with tf.get_default_graph().as_default():

        with tf.device('/gpu:%d' % gpu_id):
            cocoGT, preload_test_dataset, obj_type_nums, ID_TO_TYPE = data_loader.get_coco_minval()
            cocoDT = []

            test_data_size = len(preload_test_dataset)

            prepare_data = preprocessing_loader.prepare_before_model_construct('test', ROOT_CFG)
            anchors = prepare_data[-1]

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            with tf.name_scope('model') as scope:
                model_outputs = model_loader.forward(ROOT_CFG,
                                                    obj_type_nums,
                                                    prepare_data[0],
                                                    prepare_data[1],
                                                    backbone_name=BACKBONE_PARAMETERS['type'],
                                                    header_name=HEADER_PARAMETERS['type'])

                post_outputs = model_loader.postprocessing(ROOT_CFG,
                                                            obj_type_nums,
                                                            anchors,
                                                            model_outputs,
                                                            postprocessing_name=POSTPROCESSING_PARAMETERS['type'])

            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)

            variable_averages.apply(tf.trainable_variables())

            saver = tf.train.Saver(variable_averages.variables_to_restore())
            
            # saver = tf.train.Saver(tf.trainable_variables())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True

    printBar = progress_bar.ProgressBar(50, test_data_size)

    datastart = time.time()
    with tf.Session(config=config) as sess:
        dataset = data_loader.load_test_dataset(preload_test_dataset, ROOT_CFG)
        data = next(dataset)

        # ckpt_state = tf.train.get_checkpoint_state(TEST_PARAMETERS['ckpt_path'])

        # path = ckpt_state.all_model_checkpoint_paths[-1]
        # path = "ssd_300_vgg/ssd_300_vgg.ckpt"
        model_path = os.path.join(TEST_PARAMETERS['ckpt_path'], TEST_PARAMETERS['ckpt_name'])
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

            
        now_batch_index = 0
        while data != 0:
            batch_num = len(data[1])
            b_scores, b_bboxes, b_labels = sess.run(post_outputs,
                                feed_dict={prepare_data[0]: data[0],
                                            prepare_data[1]: False})
            for batch_index in range(batch_num):
                cocoDT = cocoDT + build_coco_annFile(ID_TO_TYPE, b_bboxes[batch_index], b_labels[batch_index], b_scores[batch_index], data[4][batch_index], data[5][batch_index])
                
            now_batch_index += batch_num
            data = next(dataset)
            printBar.print(now_batch_index)

        cocoDT = np.array(cocoDT)
        result_file = open(FLAGS.save_file, 'a')
        cocoDT = cocoGT.loadRes(cocoDT)

        # dtcoco = COCO(anno_file)
        E = COCOeval(cocoGT,cocoDT, iouType='bbox'); # initialize CocoEval object
        E.evaluate();                # run per image evaluation
        E.accumulate();              # accumulate per image results
        E.summarize(result_file);               # display summary metrics of results
        result_file.close()

        eval_model_res(coco_ann_file)
    print('total time: {}'.format(time.time() - datastart))
    sess.close()

if __name__ == '__main__':
    tf.app.run()
