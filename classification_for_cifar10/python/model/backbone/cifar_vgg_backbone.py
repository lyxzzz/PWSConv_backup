import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from utils import tf_layer
from utils import tf_utils
from utils import custom_layers
from utils import cfg_utils
import math

def arg_scope(network_cfg):
    weight_decay = network_cfg.get('weight_decay', 0.0005)

    norm_function = cfg_utils.get_norm_func(network_cfg)
    if norm_function == tf_layer.layer_norm or norm_function == tf_layer.instance_norm:
        batch_norm_params={}
    elif norm_function == slim.batch_norm or norm_function == tf_layer.switch_norm:
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
    elif norm_function == tf_layer.group_norm:
        batch_norm_params={
            'group_size':32
        }
    else:
        batch_norm_params = None
        norm_function = None

    if not network_cfg.get('backbone_norm', False):
        batch_norm_params = None
        norm_function = None

    with slim.arg_scope([tf_layer.pws_conv, tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d, tf_layer.myconv],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([tf_layer.pws_conv],
                            epsilon=cfg_utils.get_cfg_attr(network_cfg, 'pwsepsilon', 0.001),
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True)):
            with slim.arg_scope([tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d, tf_layer.myconv],
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True)):
                with slim.arg_scope([tf_layer.pws_conv, tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d, tf_layer.myconv, slim.max_pool2d],
                                    padding='SAME') as sc:
                    if norm_function == None:
                        print('!!!not use normalization!!!')
                        return sc
                    else:
                        with slim.arg_scope([tf_layer.pws_conv, tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d, tf_layer.myconv],normalizer_fn=norm_function,
                                            normalizer_params=batch_norm_params):
                            with slim.arg_scope([norm_function], **batch_norm_params) as norm_sc:
                                print('!!!use normalization!!!')
                                return norm_sc


def backbone(inputs, backbone_cfg, network_cfg, num_classes, is_training=True, scope='vgg_backbone'):

    net = inputs

    norm_func = cfg_utils.get_norm_func(network_cfg)
    conv_fn = cfg_utils.get_conv_func(network_cfg)

    tmp_points = {}
    with slim.arg_scope([norm_func], is_training=is_training):
        with tf.variable_scope(scope, 'vgg_backbone'):

            #block 1
            net = slim.repeat(net, 3, conv_fn, 96, [3, 3], scope='block1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            # Block 2.
            net = slim.repeat(net, 3, conv_fn, 192, [3, 3], scope='block2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            # Block 3.
            end_point = 'block3'
            with tf.variable_scope(end_point):
                net = conv_fn(net, 192, [3, 3], scope='conv3x3', padding='VALID')
                net = conv_fn(net, 192, [1, 1], scope='conv1x1_1')
                net = conv_fn(net, 192, [1, 1], scope='conv1x1_2')

            net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)

            # if conv_fn == tf_layer.pws_conv:
            #     net = conv_fn(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, single_beta=True, scope='mylogits')
            # else:
            net = conv_fn(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='mylogits')
            print(net)
            net = tf.reshape(net, [-1, num_classes])

            prediction = slim.softmax(net)
            print(prediction)
    return net, prediction