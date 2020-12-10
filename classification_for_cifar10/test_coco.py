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

tf.app.flags.DEFINE_string('res_file', 'result/coco_res.json', '')

tf.app.flags.DEFINE_string('last_layer_norm', None, '')

tf.app.flags.DEFINE_boolean('ba_fn', None, '')
tf.app.flags.DEFINE_boolean('ha_fn', None, '')

tf.app.flags.DEFINE_string('model', 'ssd', '')
FLAGS = tf.app.flags.FLAGS

def __parser_cmd_to_json(var, json_dict, name):
    if var is not None:
        json_dict[name] = var

def build_coco_annFile(dstcoco, TYPE_TO_ID, boxs, labels, basename, shape):
    img_id = int(basename[:-4])
    img_h, img_w = shape
    for index, b in enumerate(boxs):
        ann_id = TYPE_TO_ID[labels[index]]
        x = b[0] * img_w
        y = b[1] * img_h
        w = (b[2] - b[0]) * img_w
        h = (b[3] - b[1]) * img_h
        dstcoco['annotations'].append({
            "iscrowd":0,
            "image_id": img_id,
            "bbox": [x, y, w, h],
            "category_id": ann_id,
            "id": len(dstcoco['annotations'])
        })

def eval_model_res(gt_ann_file):
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

    __parser_cmd_to_json(FLAGS.ba_fn, NETWORK_PARAMETERS, 'backbone_activation')
    __parser_cmd_to_json(FLAGS.ha_fn, NETWORK_PARAMETERS, 'header_activation')

    file_tools.touch_dir(TEST_PARAMETERS['anno_path'])

    ROOT_CFG = cfg_loader.get_cfgs(start_cfg_dict['default_network_cfgs'], start_cfg_dict)

    
    gpu_id = int(start_cfg_dict['gpuid'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with tf.get_default_graph().as_default():

        with tf.device('/gpu:%d' % gpu_id):
            coco_ann_file, preload_test_dataset, obj_type_nums, ID_TO_NAME, TYPE_TO_ID = data_loader.get_coco_minval()

            gtcoco = json.load(open(coco_ann_file, 'r'))
            dstcoco = dict()

            dstcoco['images'] = gtcoco['images']
            dstcoco['categories'] = gtcoco['categories']
            dstcoco['info'] = gtcoco['info']
            dstcoco['licenses'] = gtcoco['licenses']
            dstcoco['annotations'] = []
            test_data_size = len(preload_test_dataset)

            prepare_data = preprocessing_loader.prepare_before_model_construct('test', ROOT_CFG)
            anchor_list = prepare_data[-1]

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
                                                            anchor_list,
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
    eval_average_f1 = np.zeros((3), dtype=np.int32)
    eval_map = eval_func.EvalmAP(obj_type_nums)

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
                
                build_coco_annFile(dstcoco, TYPE_TO_ID, b_bboxes[batch_index], b_labels[batch_index], data[4][batch_index], data[5][batch_index])
                
            now_batch_index += batch_num
            data = next(dataset)
            printBar.print(now_batch_index)

        with open(FLAGS.res_file, 'w') as f:
            res_json_data = json.dumps(dstcoco)
            json.dump(res_json_data, f)

        eval_model_res(coco_ann_file)
    print('total time: {}'.format(time.time() - datastart))
    sess.close()

if __name__ == '__main__':
    tf.app.run()
