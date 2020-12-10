# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
import tensorflow as tf
from utils import tf_layer

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def inception_v3_base(inputs,
                variables_collection = None,
                tpn_module_trainable = True,
                cls_module_trainable = True,
                final_endpoint='Mixed_6e',
                min_depth=16,
                depth_multiplier=1.0,
                scope=None):
    """Inception model from http://arxiv.org/abs/1512.00567.

    Constructs an Inception v3 network from inputs to the given final endpoint.
    This method can construct the network up to the final inception block
    Mixed_7c.

    Note that the names of the layers in the paper do not correspond to the names
    of the endpoints registered by this function although they build the same
    network.

    Here is a mapping from the old_names to the new names:
    Old name          | New name
    =======================================
    conv0             | Conv2d_1a_3x3
    conv1             | Conv2d_2a_3x3
    conv2             | Conv2d_2b_3x3
    pool1             | MaxPool_3a_3x3
    conv3             | Conv2d_3b_1x1
    conv4             | Conv2d_4a_3x3
    pool2             | MaxPool_5a_3x3
    mixed_40x40x256a  | Mixed_5b
    mixed_40x40x288a  | Mixed_5c
    mixed_40x40x288b  | Mixed_5d
    mixed_20x20x768a  | Mixed_6a
    mixed_20x20x768b  | Mixed_6b
    mixed_20x20x768c  | Mixed_6c
    mixed_20x20x768d  | Mixed_6d
    mixed_20x20x768e  | Mixed_6e
    mixed_8x8x1280a   | Mixed_7a
    mixed_8x8x2048a   | Mixed_7b
    mixed_8x8x2048b   | Mixed_7c"""

    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with variable_scope.variable_scope(scope, 'InceptionV3', [inputs]):
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
                stride=1,
                padding='SAME'):
            with arg_scope(
                [layers.conv2d, tf_layer.batch_norm], trainable=tpn_module_trainable):
                # 299 x 299 x 3
                end_point = 'Conv2d_1a_3x3'
                net = layers.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # 149 x 149 x 32
                end_point = 'Conv2d_2a_3x3'
                net = layers.conv2d(net, depth(32), [3, 3], scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # 147 x 147 x 32
                end_point = 'Conv2d_2b_3x3'
                net = layers.conv2d(
                        net, depth(64), [3, 3], padding='SAME', scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # 147 x 147 x 64
                end_point = 'MaxPool_3a_3x3'
                net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # 73 x 73 x 64
                end_point = 'Conv2d_3b_1x1'
                net = layers.conv2d(net, depth(80), [1, 1], scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # 73 x 73 x 80.
                end_point = 'Conv2d_4a_3x3'
                net = layers.conv2d(net, depth(192), [3, 3], scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # 71 x 71 x 192.
                end_point = 'MaxPool_5a_3x3'
                net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # 40 x 40 x 192.

                # Inception blocks
                # mixed: 40 x 40 x 256.
                end_point = 'Mixed_5b'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = layers.conv2d(
                                branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

                # mixed_1: 40 x 40 x 288.
                end_point = 'Mixed_5c'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = layers.conv2d(
                                branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

                # mixed_2: 40 x 40 x 288.
                end_point = 'Mixed_5d'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = layers.conv2d(
                                branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            with arg_scope(
                [layers.conv2d, tf_layer.batch_norm], trainable=cls_module_trainable):
                # mixed_3: 20 x 20 x 768.
                end_point = 'Mixed_6a'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net,
                                depth(384), [3, 3],
                                stride=2,
                                padding='SAME',
                                scope='Conv2d_1a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                        branch_1 = layers.conv2d(
                                branch_1,
                                depth(96), [3, 3],
                                stride=2,
                                padding='SAME',
                                scope='Conv2d_1a_1x1')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers_lib.max_pool2d(
                                net, [3, 3], stride=2, padding='SAME', scope='MaxPool_1a_3x3')
                    net = array_ops.concat([branch_0, branch_1, branch_2], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

                # mixed4: 20 x 20 x 768.
                end_point = 'Mixed_6b'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = layers.conv2d(
                                branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = layers.conv2d(
                                branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

                # mixed_5: 20 x 20 x 768.
                end_point = 'Mixed_6c'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = layers.conv2d(
                                branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = layers.conv2d(
                                branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # mixed_6: 20 x 20 x 768.
                end_point = 'Mixed_6d'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = layers.conv2d(
                                branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = layers.conv2d(
                                branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

                # mixed_7: 20 x 20 x 768.
                end_point = 'Mixed_6e'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = layers.conv2d(
                                branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = layers.conv2d(
                                branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

                # mixed_8: 8 x 8 x 1280.
                end_point = 'Mixed_7a'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = layers.conv2d(
                                branch_0,
                                depth(320), [3, 3],
                                stride=2,
                                padding='SAME',
                                scope='Conv2d_1a_3x3')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = layers.conv2d(
                                branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = layers.conv2d(
                                branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                        branch_1 = layers.conv2d(
                                branch_1,
                                depth(192), [3, 3],
                                stride=2,
                                padding='SAME',
                                scope='Conv2d_1a_3x3')
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers_lib.max_pool2d(
                                net, [3, 3], stride=2, padding='SAME', scope='MaxPool_1a_3x3')
                    net = array_ops.concat([branch_0, branch_1, branch_2], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
                # mixed_9: 8 x 8 x 2048.
                end_point = 'Mixed_7b'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = array_ops.concat(
                                [
                                        layers.conv2d(
                                                branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                                        layers.conv2d(
                                                branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')
                                ],
                                3)
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = array_ops.concat(
                                [
                                        layers.conv2d(
                                                branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                                        layers.conv2d(
                                                branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                                ],
                                3)
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

                # mixed_10: 8 x 8 x 2048.
                end_point = 'Mixed_7c'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(
                                net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                    with variable_scope.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(
                                net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = array_ops.concat(
                                [
                                        layers.conv2d(
                                                branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                                        layers.conv2d(
                                                branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')
                                ],
                                3)
                    with variable_scope.variable_scope('Branch_2'):
                        branch_2 = layers.conv2d(
                                net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = layers.conv2d(
                                branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = array_ops.concat(
                                [
                                        layers.conv2d(
                                                branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                                        layers.conv2d(
                                                branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                                ],
                                3)
                    with variable_scope.variable_scope('Branch_3'):
                        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = layers.conv2d(
                                branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
        raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v3(inputs,
        variables_collection = None,
        tpn_module_trainable = True,
        cls_module_trainable = True,
        final_endpoint='Mixed_6e',
        num_classes=25,
        is_training=True,
        dropout_keep_prob=0.5,
        min_depth=16,
        depth_multiplier=1.0,
        prediction_fn=layers_lib.softmax,
        spatial_squeeze=False,
        reuse=None,
        scope='InceptionV3'):
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with variable_scope.variable_scope(
            scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with arg_scope(
                [layers_lib.dropout], is_training=is_training):
            with arg_scope(
                    [tf_layer.batch_norm], is_training=is_training):
                print(variables_collection)
                net, end_points = inception_v3_base(
                            inputs,
                            tpn_module_trainable = tpn_module_trainable,
                            cls_module_trainable = cls_module_trainable,
                            variables_collection=variables_collection,
                            final_endpoint=final_endpoint,
                            scope=scope,
                            min_depth=min_depth,
                            depth_multiplier=depth_multiplier)

            # Final pooling and prediction
            with variable_scope.variable_scope('Logits'):
                # kernel_size = _reduced_kernel_size_for_small_input(net, [10, 10])
                kernel_size = net.get_shape()[1:3]
                net = layers_lib.avg_pool2d(
                        net,
                        kernel_size,
                        padding='VALID',
                        scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # 1 x 1 x 2048
                net = layers_lib.dropout(
                        net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                # 2048
                logits = layers.conv2d(
                        net,
                        num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        variables_collections=variables_collection,
                        trainable=cls_module_trainable,
                        scope='myConv2d_1c_1x1')
                if spatial_squeeze:
                    logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')
                # 1000
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points



def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):

    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
                min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
        ]
    return kernel_size_out


def inception_v3_arg_scope(weight_decay=0.0005):
    batch_norm_params = {
    }
    with arg_scope(
            [layers.conv2d, layers_lib.fully_connected],
            weights_regularizer=regularizers.l2_regularizer(weight_decay)):
        with arg_scope(
                [layers.conv2d],
                weights_initializer=initializers.variance_scaling_initializer(),
                activation_fn=nn_ops.relu
                ,normalizer_fn=tf_layer.batch_norm
                ,normalizer_params=batch_norm_params
                ) as sc:
            return sc
