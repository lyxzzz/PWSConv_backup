#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib import slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope as layer_arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from utils import tf_layer
from utils import cfg_utils

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

    with layer_arg_scope(
        [layers_lib.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=norm_function,
        normalizer_params=batch_norm_params):
        with layer_arg_scope([layers_lib.conv2d, layers.max_pool2d, layers_lib.avg_pool2d], padding='SAME'):
            with slim.arg_scope([norm_function], **batch_norm_params) as arg_sc:
                return arg_sc

def _dense_block(inputs, growth_num, bottleneck_ratio, dense_num, end_points, stop_gradient, norm_func, scope):
    net = inputs
    with tf.variable_scope(scope):
        for i in range(dense_num):
            with tf.variable_scope('block{}'.format(i)) as sc:
                dense_conv = norm_func(net)
                dense_conv = layers_lib.conv2d(net, growth_num * bottleneck_ratio, [1, 1], stride=1, scope='conv1x1')
                dense_conv = layers_lib.conv2d(dense_conv, growth_num, [3, 3], stride=1, 
                                normalizer_fn=None, scope='conv3x3')
                if stop_gradient:
                    dense_conv = tf.stop_gradient(dense_conv)
                net = tf.concat([net, dense_conv], axis=-1)
                end_points[sc.name] = net
    return net

def _transition_block(inputs, transition_ratio, norm_func, scope):
    with tf.variable_scope(scope) as sc:
        depth = inputs.get_shape().as_list()[-1] * transition_ratio
        net = norm_func(inputs)
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, 
                    normalizer_fn=None, scope='conv1x1')
        net = layers_lib.avg_pool2d(net, [2, 2], stride=2, scope='avgpool')
    return net

def backbone(inputs, backbone_cfg, network_cfg, is_training=True, scope=None):
    bottleneck_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'bottleneck_ratio', 4)
    growth_num = cfg_utils.get_cfg_attr(backbone_cfg, 'growth_num', 32)
    transition_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'transition_ratio', [0.5, 0.5, 0.5])
    if not isinstance(transition_ratio, list):
        transition_ratio = [transition_ratio] * 3
    dense_block = cfg_utils.get_cfg_attr(backbone_cfg, 'dense_block', [6, 12, 32, 32])
    use_stem = cfg_utils.get_cfg_attr(backbone_cfg, 'use_stem', True)

    stop_gradient = cfg_utils.get_cfg_attr(network_cfg, 'stop_gradient', False)

    norm_func = cfg_utils.get_norm_func(network_cfg)
    with variable_scope.variable_scope(scope, 'densenet', [inputs]) as sc:
        end_points = {}
        with layer_arg_scope([norm_func], is_training=is_training):

            net = inputs
            net = norm_func(net)
            
            if use_stem:
                #stem
                net = layers_lib.conv2d(net, 64, [3, 3], stride=2, scope='conv1_3x3')
                net = layers_lib.conv2d(net, 64, [3, 3], stride=1, scope='conv2_3x3')
                net = layers_lib.conv2d(net, 64, [3, 3], stride=1, scope='conv3_3x3')                
            else:
                net = layers_lib.conv2d(net, 64, [7, 7], stride=2, scope='conv1_7x7')
            
            net = layers_lib.avg_pool2d(net, [2, 2], stride=2, scope='avgpool1')
            for i in range(3):
                net = _dense_block(net, growth_num, bottleneck_ratio, dense_block[i], end_points, stop_gradient, norm_func, 'dense_module_{}'.format(i))
                net = _transition_block(net , transition_ratio[i], norm_func, 'transition_module_{}'.format(i))
                end_points['transition_{}'.format(i)] = net
            net = _dense_block(net, growth_num, bottleneck_ratio, dense_block[3], end_points, stop_gradient, norm_func, 'dense_module_{}'.format(3))
        return end_points