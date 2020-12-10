import datetime
import os
import sys
import time
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import json
import math

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

tf.app.flags.DEFINE_float('percent', 0.1, '')
tf.app.flags.DEFINE_string('ckpt_name', None, '')

tf.app.flags.DEFINE_string('last_layer_norm', None, '')
tf.app.flags.DEFINE_string('log_name', None, '')

tf.app.flags.DEFINE_string('model', 'ssd', '')
FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    config_path = os.path.join('train_cfgs', FLAGS.model+'.json')
    with open(config_path, 'r') as json_file:
        start_cfg_dict = json.load(json_file)
    TRAIN_PARAMETERS = start_cfg_dict['train_parameters']
    TEST_PARAMETERS = start_cfg_dict['test_parameters']
    RESTORE_PARAMETERS = start_cfg_dict['restore_parameters']
    DATASET_PARAMETERS = start_cfg_dict['dataset']
    BACKBONE_PARAMETERS = start_cfg_dict['backbone']
    HEADER_PARAMETERS = start_cfg_dict['header']
    LOSSES_PARAMETERS = start_cfg_dict['losses']
    POSTPROCESSING_PARAMETERS = start_cfg_dict['postprocessing']

    if FLAGS.ckpt_name is not None:
        TRAIN_PARAMETERS['ckpt_name'] = FLAGS.ckpt_name
    
    if FLAGS.last_layer_norm is not None:
        HEADER_PARAMETERS['last_layer_norm'] = FLAGS.last_layer_norm

    ROOT_CFG = cfg_loader.get_cfgs(start_cfg_dict.get('default_network_cfgs','emptyCFG'), start_cfg_dict)
    
    gpu_id = int(start_cfg_dict['gpuid'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")

    if FLAGS.log_name is not None:
        log_name = os.path.join(TRAIN_PARAMETERS['logs_path'], FLAGS.log_name)
    else:
        log_name = TRAIN_PARAMETERS['logs_path'] + StyleTime
    file_tools.touch_dir(log_name)
    file_tools.touch_dir(TRAIN_PARAMETERS['ckpt_path'])

    preload_train_dataset, obj_type_nums = data_loader.get_train_dataset(DATASET_PARAMETERS['train'], FLAGS.load_difficult, FLAGS.percent)
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

    ema_var_list = None

    losses_description = model_loader.losses_description(loss_name=LOSSES_PARAMETERS['type'])
    total_loss_index = losses_description[0]
    print_loss_dict = losses_description[1]
    print_loss_index = losses_description[2]

    freezen_list = TRAIN_PARAMETERS.get('freezen_list', None)
    train_op, summary_op = tf_utils.create_train_op(loss_list[total_loss_index], opt, FLAGS.moving_average_decay, global_step, ema_var_list, freezen_list)

    summary_writer = tf.summary.FileWriter(log_name, tf.get_default_graph())

    init_op = tf.global_variables_initializer()

    saver, variable_restore_op = tf_utils.create_save_op(RESTORE_PARAMETERS['restore'], TRAIN_PARAMETERS['pretrained_model_path'], TRAIN_PARAMETERS.get('pretrained_model_scope',"None")
                                    , TRAIN_PARAMETERS['max_to_keep'], ema_var_list, TRAIN_PARAMETERS.get('checkpoint_exclude_scopes', None))
    
    sess, restore_step = tf_utils.create_session(TRAIN_PARAMETERS['ckpt_path'], init_op, learning_rate, RESTORE_PARAMETERS['learning_rate']
                                    , saver, RESTORE_PARAMETERS['restore'], RESTORE_PARAMETERS['reset_learning_rate']
                                    , variable_restore_op, gpu_memory_fraction = 0.99)

    train_fetch_list = list(loss_list + post_outputs) + [train_op, summary_op]
    # train_fetch_list = list(loss_list) + [train_op, summary_op]

    max_epochs = TRAIN_PARAMETERS['max_epochs']
    if RESTORE_PARAMETERS['restore']:
        restore_epoch = RESTORE_PARAMETERS['restore_epoch']
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

    train_dataset = data_loader.load_train_dataset(max_epochs + warmup_epoch - restore_epoch, preload_train_dataset, ROOT_CFG, anchor_list)
    train_data = next(train_dataset)

    for warm_up_step in range(warmup_epoch):
        LR = sess.run(learning_rate)
        print("---------warmup[{}/{} LR:{:.6f}]--------".format(warm_up_step+1, warmup_epoch, LR))
        warmupBar = progress_bar.ProgressBar(50, train_data_size)
        warmup_index = 0
        while train_data != 0:
            batch_num = len(train_data[1])
            fetch_real_value = sess.run(train_fetch_list,
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
    sess.run(tf.assign(learning_rate, lr_init))

    epochRecorder = EpochRecorder(print_loss_dict, summary_writer, restore_epoch, max_epochs)
    start = time.time()
    step = restore_step

    for epoch in range(restore_epoch + 1, max_epochs + 1):
        train_acc = np.zeros((3), dtype=np.int32)
        mean_loss = np.zeros((len(print_loss_dict)), dtype=np.float32)
        steps_per_epoch = 0
        now_batch_nums = 0
        epochRecorder.start_epoch()
        LR = sess.run(learning_rate)
        while train_data != 0:
            batch_num = len(train_data[1])
            fetch_real_value = sess.run(train_fetch_list,
                                feed_dict={prepare_data[0]: train_data[0],
                                            prepare_data[2]: train_data[1],
                                            prepare_data[3]: train_data[2],
                                            prepare_data[4]: train_data[3],
                                            prepare_data[1]: True})

            b_scores, b_bboxes, b_labels = fetch_real_value[len(loss_list):-2]

            for batch_index in range(batch_num):
                train_acc += eval_func.get_tp_and_fp(b_scores[batch_index], b_bboxes[batch_index], b_labels[batch_index],
                                                    train_data[4][batch_index], train_data[5][batch_index], 
                                                    TEST_PARAMETERS['matching_threhold'])

            mean_loss = mean_loss + fetch_real_value[print_loss_index[0]:print_loss_index[1]]

            step = step + 1
            steps_per_epoch = steps_per_epoch + 1
            now_batch_nums += batch_num

            summary_str = fetch_real_value[-1]
            summary_writer.add_summary(summary_str, global_step=step)

            if print_each_epoch is not None and steps_per_epoch % print_each_epoch == 0:
                total_time = time.time() - start
                avg_time_per_step = total_time / print_each_epoch
                start = time.time()
                print_loss_value = mean_loss / steps_per_epoch
                print('Epoch[{}/{}] Data[{}/{}]'.format(epoch-1, max_epochs, now_batch_nums, train_data_size), end='')
                tmp_acc = eval_tools.f_measure(train_acc)
                print(',f:{:.2f} P:{:.2f} R:{:.2f}'.format(tmp_acc[0], tmp_acc[1], tmp_acc[2]), end='')
                for name, value in zip(print_loss_dict, print_loss_value):
                    print(', {} {:.4f}'.format(name, value), end='')
                
                rest_time = total_time * (train_data_size - now_batch_nums) / batch_num / print_each_epoch
                print(', {:.2f} seconds, remain {:.2f} seconds, LR {:.6f}'.format(total_time, rest_time, LR))

            train_data = next(train_dataset)

        # record train condition
        
        epochRecorder.summary_epoch(mean_loss, train_acc, LR, epoch, step, steps_per_epoch, 'train')
        train_data = next(train_dataset)
        if epoch in decay_epoch:
            sess.run(tf.assign(learning_rate, decay_learning_rate[decay_point]))
            decay_point = decay_point + 1

        if epoch in save_epochs:
            filename = (TRAIN_PARAMETERS['ckpt_name'] + '_{:d}'.format(step) + '.ckpt')
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
