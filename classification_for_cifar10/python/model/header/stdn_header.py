import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops

from utils import cfg_utils
from utils import tf_layer
from utils import tf_utils

def arg_scope(network_cfg):
    weight_decay = cfg_utils.get_cfg_attr(network_cfg, 'weight_decay', 0.0005)
    
    norm_function = cfg_utils.get_norm_func(network_cfg)
    if norm_function == tf_layer.layer_norm:
        batch_norm_params={}
    elif norm_function == slim.batch_norm:
        batch_norm_cfg = network_cfg['batch_norm']
        batch_norm_params = {
            'decay': cfg_utils.get_cfg_attr(batch_norm_cfg, 'decay', 0.997),
            'epsilon': cfg_utils.get_cfg_attr(batch_norm_cfg, 'epsilon', 0.001),
            'updates_collections': ops.GraphKeys.UPDATE_OPS,
            'fused': True,
            'variables_collections': {
                    'beta': None,
                    'gamma': None,
                    'moving_mean': ['moving_vars'],
                    'moving_variance': ['moving_vars'],
            }
        }
    else:
        raise Exception("Need Netowrk Normlization!")

    with slim.arg_scope([slim.conv2d],
                    normalizer_fn=norm_function,
                    normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                padding='SAME'):
                with slim.arg_scope([norm_function], **batch_norm_params) as sc:
                    return sc

def _subnet(inputs, channel, output_channel, last_layer_norm_func, num_anchors):
    net = slim.conv2d(inputs, channel, [1, 1], scope='conv1x1_1')
    net = slim.conv2d(net, channel, [3, 3], scope='conv3x3_1')
    net = slim.conv2d(net, output_channel, [3, 3], activation_fn=None, normalizer_fn=None,scope='conv3x3_2')
    if last_layer_norm_func is not None:
        print('use last layer norm')
        if last_layer_norm_func == tf_layer.group_norm:
            net = last_layer_norm_func(net, num_anchors, activation_fn=None)
        else:
            net = last_layer_norm_func(net, activation_fn=None)
    return net

def _multibox_layer(inputs,
                    head_channels,
                    is_training,
                    num_classes,
                    last_layer_norm_func=None,
                    ratios=[1]):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    # Number of anchors.
    num_anchors = 1 + len(ratios)

    N, H, W, C = net.shape.as_list()

    num_loc_pred = num_anchors * 4

    with tf.variable_scope('loc'):
        loc_pred = _subnet(net, head_channels, num_loc_pred, last_layer_norm_func, num_anchors)
        loc_pred = tf.reshape(loc_pred,
                            tf_utils.tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    
    with tf.variable_scope('cls'):
        cls_pred = _subnet(net, head_channels, num_cls_pred, last_layer_norm_func, num_anchors)
        cls_pred = tf.reshape(cls_pred,
                            tf_utils.tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    
    middle_nums = cls_pred.shape[1] * cls_pred.shape[2] * cls_pred.shape[3]

    loc_pred = tf.reshape(loc_pred,[-1, middle_nums, 4])
    cls_pred = tf.reshape(cls_pred,[-1, middle_nums, num_classes])

    return cls_pred, loc_pred

def get_last_layer_func(func_name):
    if func_name == 'ChannelNorm':
        return tf_layer.channel_norm
    elif func_name == 'LayerNorm':
        return tf_layer.layer_norm
    elif func_name == 'GroupNorm':
        return tf_layer.group_norm
    elif func_name == 'BatchNorm':
        return slim.batch_norm
    else:
        return None

def header(end_points, num_classes, header_cfg, network_cfg, anchors_cfg, is_training=True):
    head_channels = cfg_utils.get_cfg_attr(header_cfg, 'head_channels', 256)

    predictions = []
    logits = []
    localisations = []

    last_layer_norm = cfg_utils.get_cfg_attr(header_cfg, 'last_layer_norm', 'None')
    last_layer_norm_func = get_last_layer_func(last_layer_norm)

    norm_func = cfg_utils.get_norm_func(network_cfg)
    with slim.arg_scope([norm_func], is_training=is_training):
        for i, layer in enumerate(anchors_cfg.feat_layers):
            with tf.variable_scope(layer + '_box'):
                scales = anchors_cfg.feat_scales[i]
                end_points[layer] = norm_func(end_points[layer])
                if scales > 0:
                    nets = tf_layer.scale_transfer_layer(end_points[layer], scales)
                else:
                    pool_strides = -scales
                    nets = slim.avg_pool2d(end_points[layer], [pool_strides, pool_strides], stride=pool_strides, scope='avgpool')
                print('{}:{}'.format(i, nets.shape))
                p, l = _multibox_layer(nets,
                                        head_channels,
                                        is_training,
                                        num_classes,
                                        last_layer_norm_func,
                                        anchors_cfg.anchor_ratios[i])
            predictions.append(slim.softmax(p))
            logits.append(p)
            localisations.append(l)

    return predictions, localisations, logits