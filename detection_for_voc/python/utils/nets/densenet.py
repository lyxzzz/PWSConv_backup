#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib import slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
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

def densenet_arg_scope(weight_decay=0.0001):
    batch_norm_params = {
    }
    with arg_scope(
        [layers_lib.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=tf_layer.batch_norm,
        normalizer_params=batch_norm_params
        ):
        with arg_scope([layers_lib.conv2d, layers.max_pool2d, layers_lib.avg_pool2d], padding='SAME') as arg_sc:
            return arg_sc

def _dense_block(inputs, growth_num, bottleneck_ratio, dense_num, end_points, scope):
    net = inputs
    with tf.variable_scope(scope):
        for i in range(dense_num):
            with tf.variable_scope('block{}'.format(i)) as sc:
                dense_conv = layers_lib.conv2d(net, growth_num * bottleneck_ratio, [1, 1], stride=1, scope='conv1x1')
                dense_conv = layers_lib.conv2d(dense_conv, growth_num, [3, 3], stride=1, scope='conv3x3')
                net = tf.concat([net, dense_conv], axis=-1)
                end_points[sc.name] = net
    return net

def _transition_block(inputs, transition_depth, scope):
    with tf.variable_scope(scope) as sc:
        net = layers_lib.conv2d(inputs, transition_depth, [1, 1], stride=1, scope='conv1x1')
        net = layers_lib.avg_pool2d(net, [2, 2], stride=2, scope='avgpool')
    return net

def densenet(inputs, bottleneck_ratio=4.0,
                 growth_num = 32,
                 transition_depth=[128, 256, 640],
                 dense_block=[6, 12, 32, 32],
                 is_training=True,
                 num_classes=None,
                 scope=None):
    with variable_scope.variable_scope(scope, 'densenet', [inputs]) as sc:
        end_points = {}
        with arg_scope([tf_layer.batch_norm], is_training=is_training):
            net = tf_layer.image_subtraction(inputs)
            net = tf_layer.batch_norm(net)
            
            #stem
            net = layers_lib.conv2d(net, 64, [3, 3], stride=2, scope='conv1_3x3')
            net = layers_lib.conv2d(net, 64, [3, 3], stride=1, scope='conv2_3x3')
            net = layers_lib.conv2d(net, 64, [3, 3], stride=1, scope='conv3_3x3')
            net = layers_lib.avg_pool2d(net, [2, 2], stride=2, scope='avgpool1')

            for i in range(3):
                net = _dense_block(net, growth_num, bottleneck_ratio, dense_block[i], end_points, 'dense_module_{}'.format(i))
                net = _transition_block(net , transition_depth[i], 'transition_module_{}'.format(i))
                end_points['transition_{}'.format(i)] = net
            net = _dense_block(net, growth_num, bottleneck_ratio, dense_block[3], end_points, 'dense_module_{}'.format(3))
        
        if num_classes is not None:
            kernel_size = net.get_shape()[1:3]
            net = layers_lib.avg_pool2d(
                    net,
                    kernel_size,
                    padding='VALID',
                    scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # 1 x 1 x 2048
                # 2048
            logits = layers.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logits')
            end_points['Logits'] = logits
            end_points['Predictions'] = layers_lib.softmax(logits, scope='Predictions')
        return net, end_points
