import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import math

def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print('total_parameters: {}, {:.4f}m'.format(total_parameters, total_parameters/1000/1000))
    return total_parameters, total_parameters/1000.0/1000.0

def _filter_var_in_scopelist(var_list, scope_list):
    if scope_list is None:
        return var_list

    var_nums = len(var_list)
    var_mask = np.ones((var_nums), dtype=np.bool)

    for s_scope in scope_list:
        for var_index in range(var_nums):
            if var_list[var_index].op.name.startswith(s_scope) == True:
                var_mask[var_index] = False
    new_var_list = []
    for new_var_index in range(var_nums):
        if var_mask[new_var_index] == True:
            new_var_list.append(var_list[new_var_index])
    return new_var_list

def create_train_op(total_loss, optimizer, moving_average_decay, global_step, ema_var_list=None, freezen_list=None):
    count_parameters()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    if update_ops: 
        batch_norm_updates_op = tf.group(*update_ops)

    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    grad_val_list = _filter_var_in_scopelist(trainable_var_list, freezen_list)
    grads = optimizer.compute_gradients(total_loss, var_list=grad_val_list)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    
    if ema_var_list is None:
        ema_var_list = tf.trainable_variables()

    variables_averages_op = variable_averages.apply(ema_var_list)

    if update_ops:
        with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
            train_op = tf.no_op(name='train_op')
    else:
        with tf.control_dependencies([variables_averages_op, apply_gradient_op]):
            train_op = tf.no_op(name='train_op')        

    summary_op = tf.summary.merge_all()
    return train_op, summary_op, grads

def create_save_op(restore, pretrained_path, pretrained_scope, max_to_keep, pretrained_variables = None, checkpoint_exclude_scopes = None):
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
    variable_restore_op = None
    if pretrained_path is not None and restore != True:
        if pretrained_scope == 'None':
            raise Exception("None Pretrained Scope")
        if pretrained_variables is None:
            load_para_list = slim.get_trainable_variables()
        else:
            load_para_list = pretrained_variables
        
        load_para_list = _filter_var_in_scopelist(load_para_list, checkpoint_exclude_scopes)

        if pretrained_scope is not None:
            load_para_dict = {}
            for load_var in load_para_list:
                tmp_name = load_var.op.name
                tmp_index = tmp_name.find('/')
                if tmp_index == -1:
                    load_para_dict[tmp_name] = load_var
                else:
                    dst_name = pretrained_scope + tmp_name[tmp_index:]
                    load_para_dict[dst_name] = load_var
        else:
            load_para_dict = load_para_list
        variable_restore_op = slim.assign_from_checkpoint_fn(pretrained_path,
                                                        load_para_dict,
                                                        ignore_missing_vars=True)
        print('use pretrained model: {}'.format(pretrained_path))
        # variable_restore_op = slim.assign_from_checkpoint_fn(pretrained_path,
        #                                                      slim.get_trainable_variables(),
        #                                                      ignore_missing_vars=True)
    return saver, variable_restore_op

def create_session(ckpt_path, init_op, learning_rate, LR_value, saver, restore=True, reset_learning_rate=True, pretrain_restore_op=None,
                    gpu_memory_fraction = 0.99):
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_memory_fraction)
    # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options)
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    if restore:
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        restore_step = int(ckpt.split('.')[-2].split('_')[-1])
        print("continue from previous checkpoint {}".format(restore_step))
        saver.restore(sess, ckpt)
        if reset_learning_rate:
            sess.run(tf.assign(learning_rate, LR_value))
    else:
        sess.run(init_op)
        restore_step = 0
        if pretrain_restore_op is not None:
            pretrain_restore_op(sess)
    return sess, restore_step

def get_variable(root, name):
    with tf.variable_scope(root, reuse=True):
        var = tf.get_variable(name)
        return var

def tensor_shape(x, rank=3):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]