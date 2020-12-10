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
tf.app.flags.DEFINE_string('conv_type', None, '')
tf.app.flags.DEFINE_string('norm_type', None, '')
tf.app.flags.DEFINE_integer('batch_size', None, '')
tf.app.flags.DEFINE_float('memory_fraction', 0.5, '')
tf.app.flags.DEFINE_string('model', 'tpn', '')

tf.app.flags.DEFINE_float('learning_rate', None, '')

FLAGS = tf.app.flags.FLAGS
class Fetch():
    def __init__(self):
        self.fetchlist = []
        self.index = {}
    def add(self, val, name):
        self.index[name] = [len(self.fetchlist), len(val)]
        self.fetchlist += val
    def get(self, real_val, name):
        return real_val[self.index[name][0]:self.index[name][0]+self.index[name][1]]

def _summary_mean_var(input, axes, name):
    # print(input)
    mean, var = tf.nn.moments(input, axes=axes)
    
    mean = tf.reduce_mean(mean)
    var = tf.reduce_mean(var)

    mean_name = name + '_mean'
    var_name = name + '_var'
    
    tf.summary.scalar(mean_name, mean)
    tf.summary.scalar(var_name, var)

def info(var, mul=1.0):
    if len(var.shape) > 1:
        mul = 1.0
        for v in var.shape[:-1]:
            mul *= v
    print("shape:{}, var:{}, mean:{}, mul:{}, after mul:{}".format(var.shape, np.var(var), np.mean(var), mul, np.var(var) * mul))

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
    __parser_cmd_to_json(FLAGS.conv_type, NETWORK_PARAMETERS, 'conv_type')
    __parser_cmd_to_json(FLAGS.norm_type, NETWORK_PARAMETERS, 'norm_func')
    __parser_cmd_to_json(FLAGS.batch_size, TRAIN_PARAMETERS, 'train_batch_nums')
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

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    if RESTORE_PARAMETERS['restore']:
        lr_init = RESTORE_PARAMETERS['learning_rate']
    else:
        lr_init = TRAIN_PARAMETERS['learning_rate'][0]        

    if 'warmup_epoch' in TRAIN_PARAMETERS and TRAIN_PARAMETERS['warmup_epoch'] != 0:
        warmup_epoch = TRAIN_PARAMETERS['warmup_epoch']
        warmup_init = TRAIN_PARAMETERS['warmup_init']
        learning_rate = tf.Variable(warmup_init, trainable=False)
        warmup_ratios = (lr_init - warmup_init) / warmup_epoch
    else:
        warmup_epoch = 0
        warmup_ratios = 0.0
        learning_rate = tf.Variable(lr_init, trainable=False)
    
    tf.summary.scalar('learning_rate', learning_rate)

    opt = optimizer_config.get_optimizer_from_cfg(learning_rate, TRAIN_PARAMETERS.get('optimizer', None))

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
    # make_var_mean_summary(model_outputs[0:2])
    
    # root_filter = tf.get_collection('filter')
    # conv_output = tf.get_collection('conv_output')
    # conv_input = tf.get_collection('conv_input')
    # person = tf.get_collection('per')
    conv_output = tf.get_collection('conv_output')
    conv_input = tf.get_collection('conv_input')
    gamma_list = tf.get_collection('gamma')
    per_list = tf.get_collection('per')
    kernel_var_list = tf.get_collection('kernel_var')
    kernel_real_var_list = tf.get_collection('kernel_real_var')
    out_list = []
    in_list = []
    for i, c in enumerate(conv_output):
        if "header" not in c.name:
            logname = c.name[len("model/tpn_backbone/"):-len("/Conv2D:0")]
            m, v = tf.nn.moments(c, [0, 1, 2, 3], name='moments')
            tf.summary.scalar("{}/output".format(logname), v)

            m, v = tf.nn.moments(conv_input[i], [0, 1, 2, 3], name='moments')
            tf.summary.scalar("{}/input".format(logname), v)

            tf.summary.scalar("{}/gamma".format(logname), tf.reduce_mean(gamma_list[i]))

            tf.summary.scalar("{}/per".format(logname), per_list[i])

            tf.summary.scalar("{}/kernelvar".format(logname), kernel_var_list[i])

            tf.summary.scalar("{}/realvar".format(logname), kernel_real_var_list[i])

    ema_var_list = None

    losses_description = model_loader.losses_description(loss_name=LOSSES_PARAMETERS['type'])
    total_loss_index = losses_description[0]
    print_loss_dict = losses_description[1]
    print_loss_index = losses_description[2]

    freezen_list = TRAIN_PARAMETERS.get('freezen_list', None)
    train_op, summary_op, grads = tf_utils.create_train_op(loss_list[total_loss_index], opt, FLAGS.moving_average_decay, global_step, ema_var_list, freezen_list)

    grad_val = []
    grad_name = []
    grad_real = []
    for g in grads:
        if "root_filter" in g[1].name:
            grad_val.append(g[0])
            grad_name.append((g[1].name[::-1][(g[1].name[::-1].index('/'))+1:])[::-1])
            grad_real.append(g[1])

    summary_writer = tf.summary.FileWriter(TRAIN_PARAMETERS['logs_path'] + StyleTime, tf.get_default_graph())

    init_op = tf.global_variables_initializer()

    saver, variable_restore_op = tf_utils.create_save_op(RESTORE_PARAMETERS['restore'], TRAIN_PARAMETERS['pretrained_model_path'], TRAIN_PARAMETERS.get('pretrained_model_scope',"None")
                                    , TRAIN_PARAMETERS['max_to_keep'], ema_var_list, TRAIN_PARAMETERS.get('checkpoint_exclude_scopes', None))
    
    sess, restore_step = tf_utils.create_session(TRAIN_PARAMETERS['ckpt_path'], init_op, learning_rate, RESTORE_PARAMETERS['learning_rate']
                                    , saver, RESTORE_PARAMETERS['restore'], RESTORE_PARAMETERS['reset_learning_rate']
                                    , variable_restore_op, gpu_memory_fraction = FLAGS.memory_fraction)

    fetch = Fetch()
    fetch.add(list(loss_list), "loss")
    fetch.add(list(post_outputs), "post")
    fetch.add([train_op, summary_op], "trainop")

    max_epochs = TRAIN_PARAMETERS['max_epochs']
    if RESTORE_PARAMETERS['restore']:
        ckpt_path = tf.train.latest_checkpoint(TRAIN_PARAMETERS['ckpt_path'])
        restore_epoch = int(ckpt_path.split('.')[-2].split('_')[-1])
    else:
        restore_epoch = 0
    
    print_each_epoch = TRAIN_PARAMETERS['print_each_epoch']
    decay_epoch = TRAIN_PARAMETERS['decay_epoch']
    decay_learning_rate = TRAIN_PARAMETERS['learning_rate']

    decay_point = 1
    for _ in decay_epoch:
        if restore_epoch >= _:
            decay_point += 1

    save_epochs = TRAIN_PARAMETERS['save_epochs']

    train_dataset = data_loader.load_train_dataset(max_epochs + warmup_epoch - restore_epoch, preload_train_dataset, ROOT_CFG, anchor_list, aug_epochs=TRAIN_PARAMETERS["aug_epochs"])
    train_data = next(train_dataset)

    for warm_up_step in range(warmup_epoch):
        LR = sess.run(learning_rate)
        print("---------warmup[{}/{} LR:{:.6f}]--------".format(warm_up_step+1, warmup_epoch, LR))
        warmupBar = progress_bar.ProgressBar(50, train_data_size)
        warmup_index = 0
        while train_data != 0:
            batch_num = len(train_data[1])
            fetch_real_value = sess.run(fetch.fetchlist,
                                feed_dict={prepare_data[0]: train_data[0],
                                            prepare_data[2]: train_data[1],
                                            prepare_data[3]: train_data[2],
                                            prepare_data[4]: train_data[3],
                                            prepare_data[1]: True})
            warmup_index += batch_num
            warmupBar.print(warmup_index)
            train_data = next(train_dataset)
        train_data = next(train_dataset)
        sess.run(tf.assign(learning_rate, LR + warmup_ratios))
    if not RESTORE_PARAMETERS['restore']:
        # lr_init = RESTORE_PARAMETERS['learning_rate']
        sess.run(tf.assign(learning_rate, lr_init))

    epochRecorder = EpochRecorder(print_loss_dict, summary_writer, restore_epoch, max_epochs, isMAP=False)
    start = time.time()
    step = restore_step

    for epoch in range(restore_epoch + 1, max_epochs + 1):
        train_acc = np.zeros((3), dtype=np.int32)
        mean_loss = np.zeros((len(print_loss_dict)), dtype=np.float32)
        steps_per_epoch = 0
        now_batch_nums = 0
        epochRecorder.start_epoch()
        LR = sess.run(learning_rate)

        # eval_map = eval_func.EvalmAP(obj_type_nums)

        while train_data != 0:
            batch_num = len(train_data[1])
            fetch_real_value = sess.run(fetch.fetchlist,
                                feed_dict={prepare_data[0]: train_data[0],
                                            prepare_data[2]: train_data[1],
                                            prepare_data[3]: train_data[2],
                                            prepare_data[4]: train_data[3],
                                            prepare_data[1]: True})

            b_scores, b_bboxes, b_labels = fetch.get(fetch_real_value, "post")

            # eval_map.addsample(batch_num, b_scores, b_bboxes, b_labels, train_data[4], train_data[5], matching_threshold=TEST_PARAMETERS['matching_threhold'])

            for batch_index in range(batch_num):
                train_acc += eval_func.get_tp_and_fp(b_scores[batch_index], b_bboxes[batch_index], b_labels[batch_index],
                                                    train_data[4][batch_index], train_data[5][batch_index], 
                                                    TEST_PARAMETERS['matching_threhold'])

            mean_loss = mean_loss + fetch.get(fetch_real_value, "loss")[print_loss_index[0]:print_loss_index[1]]

            step = step + 1
            steps_per_epoch = steps_per_epoch + 1
            now_batch_nums += batch_num

            summary_str = fetch.get(fetch_real_value, "trainop")[-1]
            summary_writer.add_summary(summary_str, global_step=step)

            if print_each_epoch is not None and steps_per_epoch % print_each_epoch == 0:
                total_time = time.time() - start
                avg_time_per_step = total_time / print_each_epoch
                start = time.time()
                print_loss_value = mean_loss / steps_per_epoch
                print('Epoch[{}/{}] Data[{}/{}]'.format(epoch-1, max_epochs, now_batch_nums, train_data_size), end='')
                # ap_dict, total_ap, info_dict, pr_dict = eval_map.calmAP()
                # print(',map{:.2f}'.format(100 * total_ap), end='')
                tmp_acc = eval_tools.f_measure(train_acc)
                print(',f:{:.2f} P:{:.2f} R:{:.2f}'.format(tmp_acc[0], tmp_acc[1], tmp_acc[2]), end='')
                for name, value in zip(print_loss_dict, print_loss_value):
                    print(', {} {:.4f}'.format(name, value), end='')
                
                rest_time = total_time * (train_data_size - now_batch_nums) / batch_num / print_each_epoch
                print(', {:.2f} seconds, remain {:.2f} seconds, LR {:.6f}'.format(total_time, rest_time, LR))

            train_data = next(train_dataset)

        # record train condition
        # ap_dict, total_ap, info_dict, pr_dict = eval_map.calmAP()
        # epochRecorder.summary_epoch(mean_loss, 100 * total_ap, LR, epoch, step, steps_per_epoch, 'train')
        epochRecorder.summary_epoch(mean_loss, train_acc, LR, epoch, step, steps_per_epoch, 'train')
        train_data = next(train_dataset)
        if epoch in decay_epoch:
            sess.run(tf.assign(learning_rate, decay_learning_rate[decay_point]))
            decay_point = decay_point + 1

        if save_epochs[0] < 0:
            save_epoch = -save_epochs[0]
            if epoch % save_epoch == 0:
                filename = (TRAIN_PARAMETERS['ckpt_name'] + '_{:d}'.format(epoch) + '.ckpt')
                filename = os.path.join(TRAIN_PARAMETERS['ckpt_path'], filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))
        else:
            if epoch in save_epochs:
                filename = (TRAIN_PARAMETERS['ckpt_name'] + '_{:d}'.format(epoch) + '.ckpt')
                filename = os.path.join(TRAIN_PARAMETERS['ckpt_path'], filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))

    filename = (TRAIN_PARAMETERS['ckpt_name'] + '_last' + '.ckpt')
    filename = os.path.join(TRAIN_PARAMETERS['ckpt_path'], filename)
    saver.save(sess, filename)
    print('Write model to: {:s}'.format(filename))

    sess.close()

if __name__ == '__main__':
    tf.app.run()


            # real_grad = fetch_real_value[0:len(grads)]
            # while True:
            #     key = input("input:")
            #     if key == "exit":
            #         break
            #     else:
            #         g_var = [] 
            #         n_var = []
            #         p_var = []
            #         for i, name in enumerate(grad_name):
            #             if key in name:
            #                 print(name)
            #                 g_var.append(real_grad[i])
            #                 n_var.append(name)
            #                 p_var.append(grad_real[i])
            #         p_var = sess.run(p_var)
            #         while True:
            #             key = input("query:")
            #             if key == "q":
            #                 break
            #             if key == "p":
            #                 for i, v in enumerate(n_var):
            #                     print(n_var[i])
            #             else:
            #                 for i, name in enumerate(n_var):
            #                     if key in name:
            #                         print("{}".format(name))
            #                         info(g_var[i])
            #                         info(p_var[i])
                    # real_var = sess.run(p_var)
            # exit()