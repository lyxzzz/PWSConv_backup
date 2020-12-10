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

    __parser_cmd_to_json(FLAGS.last_layer_norm, HEADER_PARAMETERS, 'last_layer_norm')
    __parser_cmd_to_json(FLAGS.conv_type, NETWORK_PARAMETERS, 'conv_type')
    __parser_cmd_to_json(FLAGS.norm_type, NETWORK_PARAMETERS, 'norm_func')


    file_tools.touch_dir(TEST_PARAMETERS['anno_path'])

    ROOT_CFG = start_cfg_dict
    cfg_loader.get_cfgs(ROOT_CFG)

    
    gpu_id = FLAGS.gpuid
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with tf.get_default_graph().as_default():

        with tf.device('/gpu:%d' % gpu_id):
            preload_test_dataset, obj_type_nums, ID_TO_NAME = data_loader.get_test_dataset(DATASET_PARAMETERS['test'], FLAGS.load_difficult)
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
        dataset = data_loader.load_test_dataset(preload_test_dataset, ROOT_CFG)
        data = next(dataset)

        # ckpt_state = tf.train.get_checkpoint_state(TEST_PARAMETERS['ckpt_path'])

        # path = ckpt_state.all_model_checkpoint_paths[-1]
        # path = "ssd_300_vgg/ssd_300_vgg.ckpt"
        model_path = os.path.join(TEST_PARAMETERS['ckpt_path'], TEST_PARAMETERS['ckpt_name'])
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

            
        now_batch_index = 0
        if TEST_PARAMETERS['write_to_file']:
            scores_dict = {}
            basename_dict = {}
            bboxes_dict = {}
            for c in range(1, obj_type_nums):
                scores_dict[c] = []
                basename_dict[c] = []
                bboxes_dict[c] = []
        while data != 0:
            batch_num = len(data[1])
            start = time.time()
            b_scores, b_bboxes, b_labels = sess.run(post_outputs,
                                feed_dict={prepare_data[0]: data[0],
                                            prepare_data[1]: False})
            
            # print("{:.2f}s".format(time.time()-start))
            # start = time.time()
            eval_map.addsample(batch_num, b_scores, b_bboxes, b_labels, data[1], data[2], data[3], TEST_PARAMETERS['matching_threhold'])
            # print("{:.2f}s".format(time.time()-start))
            # for batch_index in range(batch_num):
            #     eval_average_f1 += eval_func.get_tp_and_fp(b_scores[batch_index], b_bboxes[batch_index], b_labels[batch_index],
            #                                         data[1][batch_index], data[2][batch_index], 
            #                                         TEST_PARAMETERS['matching_threhold'])
            #     if TEST_PARAMETERS['write_to_file']:
            #         for c in range(1, obj_type_nums):
            #             idxes = np.where(b_labels[batch_index] == c)[0]
            #             detected_scores = b_scores[batch_index][idxes]
            #             detected_boxes = b_bboxes[batch_index][idxes]
            #             idxes = np.where(detected_scores > 1e-4)[0]
            #             detected_scores = detected_scores[idxes]
            #             detected_boxes = detected_boxes[idxes]
            #             for i in range(detected_scores.shape[0]):
            #                 scores_dict[c].append(detected_scores[i])
            #                 basename_dict[c].append(data[4][batch_index])
            #                 bboxes_dict[c].append(detected_boxes[i])
            now_batch_index += batch_num
            data = next(dataset)
            printBar.print(now_batch_index)
        # eval_average_f1 = eval_tools.f_measure(eval_average_f1)
        # print('f:{:.2f} P:{:.2f} R:{:.2f}'.format(eval_average_f1[0], eval_average_f1[1], eval_average_f1[2]))

        # if TEST_PARAMETERS['write_to_file']:
        #     anno_root = TEST_PARAMETERS['anno_path']
        #     for c in scores_dict.keys():
        #         anno_file_path = os.path.join(anno_root, "{}.anno".format(ID_TO_NAME[c]))
        #         with open(anno_file_path, 'w') as anno_file:
        #             idxes = np.argsort(scores_dict[c])[::-1]
        #             for singel_rec in range(len(idxes)):
        #                 write_score = scores_dict[c][idxes[singel_rec]]
        #                 write_bbox = bboxes_dict[c][idxes[singel_rec]]
        #                 write_basename = basename_dict[c][idxes[singel_rec]]
        #                 write_str = '{},{},{},{},{},{}\n'.format(write_basename, write_score,
        #                                                             write_bbox[0],write_bbox[1],
        #                                                             write_bbox[2],write_bbox[3])
        #                 anno_file.write(write_str)
        ap_dict, total_ap, info_dict, pr_dict = eval_map.calmAP()
        print('mAP:{:.2f}'.format(100*total_ap))
        total_tp_fp_gt = np.array([0, 0, 0], dtype=np.int32)
        for c in ap_dict.keys():
            total_tp_fp_gt += np.array(info_dict[c], dtype=np.int32)
            print('{}:{:.2f}\t\ttp:{} predict:{} gt:{}]'.format(ID_TO_NAME[c], ap_dict[c]*100, info_dict[c][0], info_dict[c][1], info_dict[c][2]))
            print(pr_dict[c])
        # print(ap_dict)
        print(total_tp_fp_gt)

        if FLAGS.save_file is not None:
            now = datetime.datetime.now()
            StyleTime = now.strftime("%Y-%m-%d %H:%m")
            with open(FLAGS.save_file, 'a') as fsave:
                fsave.write('{}:{} {:.4f}\n'.format(FLAGS.exp_name, para_nums[0], para_nums[1]))
                fsave.write('mAP:{:.2f}\t'.format(100*total_ap))
                fsave.write('[{}, {}, {}]\n'.format(total_tp_fp_gt[0], total_tp_fp_gt[1], total_tp_fp_gt[2]))
                for c in ap_dict.keys():
                    fsave.write('{}:{:.2f}\t\t[tp:{} predict:{} gt:{}]\n\t'.format(ID_TO_NAME[c], ap_dict[c]*100, info_dict[c][0], info_dict[c][1], info_dict[c][2]))
                    for pr in pr_dict[c]:
                        fsave.write('{:.2f}, '.format(pr))
                    fsave.write('\n')    
    print('total time: {}'.format(time.time() - datastart))
    sess.close()

if __name__ == '__main__':
    tf.app.run()
