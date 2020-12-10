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

from utils import eval_tools
from utils import tf_utils
from utils.epoch_info_record import EpochRecorder
from utils import file_tools
from utils import progress_bar


tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')

tf.app.flags.DEFINE_string('ckpt_path', None, '')
tf.app.flags.DEFINE_string('ckpt_name', None, '')

tf.app.flags.DEFINE_string('save_file', None, '')

tf.app.flags.DEFINE_string('conv_type', None, '')
tf.app.flags.DEFINE_string('norm_type', None, '')
tf.app.flags.DEFINE_float('pwsepsilon', None, '')

tf.app.flags.DEFINE_string('exp_name', None, '')
tf.app.flags.DEFINE_string('model', 'VGG', '')
tf.app.flags.DEFINE_integer('gpuid', 1, '')
FLAGS = tf.app.flags.FLAGS

def __parser_cmd_to_json(var, json_dict, name):
    if var is not None:
        json_dict[name] = var

def main(argv=None):
    config_path = os.path.join('test_cfgs', FLAGS.model+'.json')
    with open(config_path, 'r') as json_file:
        start_cfg_dict = json.load(json_file)

    TEST_PARAMETERS = start_cfg_dict['test_parameters']
    DATASET_PARAMETERS = start_cfg_dict['dataset']
    BACKBONE_PARAMETERS = start_cfg_dict['backbone']
    NETWORK_PARAMETERS = start_cfg_dict['network']
    AUGMENT_PARAMETERS = start_cfg_dict['augmentation']

    if FLAGS.ckpt_path is not None:
        TEST_PARAMETERS['ckpt_path'] = FLAGS.ckpt_path
    
    if FLAGS.ckpt_name is not None:
        TEST_PARAMETERS['ckpt_name'] = FLAGS.ckpt_name

    __parser_cmd_to_json(FLAGS.conv_type, NETWORK_PARAMETERS, 'conv_type')
    __parser_cmd_to_json(FLAGS.norm_type, NETWORK_PARAMETERS, 'norm_func')
    __parser_cmd_to_json(FLAGS.pwsepsilon, NETWORK_PARAMETERS, 'pwsepsilon')


    file_tools.touch_dir(TEST_PARAMETERS['anno_path'])

    ROOT_CFG = cfg_loader.get_cfgs(start_cfg_dict['default_network_cfgs'], start_cfg_dict)

    
    gpu_id = FLAGS.gpuid
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with tf.get_default_graph().as_default():

        with tf.device('/gpu:%d' % gpu_id):
            preload_test_dataset, obj_type_nums, ID_TO_NAME = data_loader.get_test_dataset(DATASET_PARAMETERS['test'])
            test_data_size = len(preload_test_dataset)

            prepare_data = preprocessing_loader.prepare_before_model_construct('test', ROOT_CFG)

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            with tf.name_scope('model') as scope:
                model_outputs = model_loader.forward(ROOT_CFG,
                                                    obj_type_nums,
                                                    prepare_data[0],
                                                    prepare_data[1],
                                                    backbone_name=BACKBONE_PARAMETERS['type'])

            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)

            variable_averages.apply(tf.trainable_variables())

            saver = tf.train.Saver(variable_averages.variables_to_restore())
            
            # saver = tf.train.Saver(tf.trainable_variables())
    para_nums = tf_utils.count_parameters()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True

    printBar = progress_bar.ProgressBar(50, test_data_size)
    eval_average_f1 = np.zeros((3), dtype=np.int32)
    eval_map = eval_func.EvalmAP(obj_type_nums)

    datastart = time.time()
    with tf.Session(config=config) as sess:
        dataset = data_loader.load_test_dataset(preload_test_dataset, ROOT_CFG, AUGMENT_PARAMETERS)
        data = next(dataset)

        # ckpt_state = tf.train.get_checkpoint_state(TEST_PARAMETERS['ckpt_path'])

        # path = ckpt_state.all_model_checkpoint_paths[-1]
        # path = "ssd_300_vgg/ssd_300_vgg.ckpt"
        model_path = os.path.join(TEST_PARAMETERS['ckpt_path'], TEST_PARAMETERS['ckpt_name'])
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

            
        now_batch_index = 0

        top1err = np.zeros((2), dtype=np.int32)
        
        while data != 0:
            batch_num = len(data[1])
            start = time.time()
            logits, pred = sess.run(model_outputs,
                                feed_dict={prepare_data[0]: data[0],
                                            prepare_data[1]: False})

            pred_label = np.argmax(pred, axis=1)

            top1err[0] += np.sum(pred_label == data[1]) 
            top1err[1] += float(batch_num)
            
            now_batch_index += batch_num
            data = next(dataset)
            printBar.print(now_batch_index)
        
        top1err = eval_tools.top_error(top1err)
        print('err:{:.2f}'.format(top1err))

        if FLAGS.save_file is not None:
            now = datetime.datetime.now()
            StyleTime = now.strftime("%Y-%m-%d %H:%m")
            with open(FLAGS.save_file, 'a') as fsave:
                fsave.write('{}:{} {:.4f}\n'.format(FLAGS.exp_name, para_nums[0], para_nums[1]))
                fsave.write('err:{:.2f}\n'.format(top1err)) 
    print('total time: {}'.format(time.time() - datastart))
    sess.close()

if __name__ == '__main__':
    tf.app.run()
