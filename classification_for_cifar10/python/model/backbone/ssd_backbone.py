import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from utils import tf_layer
from utils import tf_utils
from utils import custom_layers
from utils import cfg_utils
import math

# with slim.arg_scope([tf_layer.unbalance_conv],
#                 rate_initializer=tf.random_uniform_initializer(minval=1.0-math.sqrt(3.0), maxval=1.0+math.sqrt(3.0)),
#                 # rate_initializer=tf.constant_initializer(-100.0),
#                 multiplier=unbalance_multiplier,
#                 normalization_rate=normalization_rate,
#                 normalization_point=normalization_point):

def arg_scope(network_cfg):
    weight_decay = network_cfg.get('weight_decay', 0.0005)
    unbalance_multiplier = network_cfg.get('unbalance_multiplier', 1.0)
    normalization_rate = network_cfg.get('unbalance_norm', True)
    normalization_point = network_cfg.get('unbalance_point', 1.0)

    norm_function = cfg_utils.get_norm_func(network_cfg)
    if norm_function == tf_layer.layer_norm or norm_function == tf_layer.instance_norm or norm_function == tf_layer.switch_norm:
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

    with slim.arg_scope([tf_layer.pws_conv, tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([tf_layer.pws_conv],
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True)):
            with slim.arg_scope([tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d],
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([tf_layer.pws_conv, tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    if norm_function == None:
                        print('!!!not use normalization!!!')
                        return sc
                    else:
                        with slim.arg_scope([tf_layer.pws_conv, tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d],normalizer_fn=norm_function,
                                            normalizer_params=batch_norm_params):
                            with slim.arg_scope([norm_function], **batch_norm_params) as norm_sc:
                                print('!!!use normalization!!!')
                                return norm_sc


def backbone(inputs, backbone_cfg, network_cfg, is_training=True, scope='tpn_backbone'):
    end_points = {}
    output_points = []
    net = tf_layer.image_subtraction(inputs)

    dense_block = cfg_utils.get_cfg_attr(backbone_cfg, 'dense_block', 2)
    bottleneck_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'bottleneck_ratio', 4)
    growth_num = cfg_utils.get_cfg_attr(backbone_cfg, 'growth_num', 64)
    transition_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'transition_ratio', 0.5)
    # transition_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'transition_ratio', 1.0)

    norm_func = cfg_utils.get_norm_func(network_cfg)
    conv_fn = cfg_utils.get_conv_func(network_cfg)

    tmp_points = {}
    with slim.arg_scope([norm_func], is_training=is_training):
        with tf.variable_scope(scope, 'tpn_backbone'):

            # conv_fn = slim.conv2d
            #block 1
            net = slim.repeat(net, 2, conv_fn, 64, [3, 3], scope='conv1')
            #160
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            # Block 2.
            net = slim.repeat(net, 2, conv_fn, 128, [3, 3], scope='conv2')
            #80
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            # Block 3.
            net = slim.repeat(net, 3, conv_fn, 256, [3, 3], scope='conv3')
            #40
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            # Block 4.
            net = slim.repeat(net, 3, conv_fn, 512, [3, 3], scope='conv4')
            end_points['block4'] = net
            #20
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            # Block 5.
            net = slim.repeat(net, 3, conv_fn, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

            # Block 6.
            net = conv_fn(net, 1024, [3, 3], scope='conv6')
            net = conv_fn(net, 1024, [1, 1], scope='conv7')
            end_points['block5'] = net

            end_point = 'block6'
            with tf.variable_scope(end_point):
                net = conv_fn(net, 256, [1, 1], scope='conv1x1')
                net = conv_fn(net, 512, [3, 3], stride=2, scope='conv3x3')
            end_points[end_point] = net


            end_point = 'block7'
            with tf.variable_scope(end_point):
                net = conv_fn(net, 128, [1, 1], scope='conv1x1')
                net = conv_fn(net, 256, [3, 3], stride=2, scope='conv3x3')
            end_points[end_point] = net
            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = conv_fn(net, 128, [1, 1], scope='conv1x1')
                net = conv_fn(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net

            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = conv_fn(net, 128, [1, 1], scope='conv1x1')
                net = conv_fn(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net

    return end_points