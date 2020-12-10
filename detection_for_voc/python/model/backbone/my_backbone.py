import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from utils import tf_layer
from utils import tf_utils
from utils import custom_layers
from utils import cfg_utils
import math

def lrelu(inputs):
    N, H, W, C = inputs.shape.as_list()
    para_shape = [C]
    alpha = slim.variable("linear_para", shape=para_shape,
                        initializer=tf.constant_initializer(0.0),
                        # device=None,
                        trainable=True)
    beta = slim.variable("rectify_para", shape=para_shape,
                    initializer=tf.constant_initializer(1.0),
                    # device=None,
                    trainable=True)
    alpha = tf.exp(alpha)
    beta = tf.exp(beta)
    sum = alpha + beta
    alpha = alpha / sum
    alpha = tf.reshape(alpha, [1, 1, 1, -1])
    outputs = tf.maximum(inputs, alpha * inputs)

    return outputs

def arg_scope(network_cfg):
    weight_decay = network_cfg.get('weight_decay', 0.0005)
    unbalance_multiplier = network_cfg.get('unbalance_multiplier', 2.0)
    normalization_rate = network_cfg.get('unbalance_norm', True)
    normalization_point = network_cfg.get('unbalance_point', 1.0)

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
        batch_norm_params = None
        norm_function = None

    if not network_cfg.get('backbone_norm', False):
        batch_norm_params = None
        norm_function = None

    with slim.arg_scope([tf_layer.unbalance_conv, slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        # weights_initializer=tf.constant_initializer(0.0),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([tf_layer.unbalance_conv],
                        rate_initializer=tf.random_uniform_initializer(minval=1.0-math.sqrt(3.0), maxval=1.0+math.sqrt(3.0)),
                        # rate_initializer=tf.constant_initializer(1.0),
                        multiplier=unbalance_multiplier,
                        normalization_rate=normalization_rate,
                        normalization_point=normalization_point):
            with slim.arg_scope([tf_layer.unbalance_conv, slim.conv2d, slim.max_pool2d],
                                padding='SAME') as sc:
                if norm_function == None:
                    return sc
                else:
                    with slim.arg_scope([tf_layer.unbalance_conv, slim.conv2d],normalizer_fn=norm_function,
                                        normalizer_params=batch_norm_params):
                        with slim.arg_scope([norm_function], **batch_norm_params) as norm_sc:
                            print('!!!use normalization!!!')
                            return norm_sc

def dense_module(inputs, growth_num, bottleneck_ratio, dense_num, scope, rate=1):
    net = inputs
    with tf.variable_scope(scope):
        for i in range(dense_num):
            with tf.variable_scope('block{}'.format(i)) as sc:
                dense_conv = slim.conv2d(net, growth_num * bottleneck_ratio, [1, 1], stride=1, scope='conv1x1')
                dense_conv = tf_layer.unbalance_conv(dense_conv, growth_num, [3, 3], stride=1, scope='conv3x3')
                net = tf.concat([net, dense_conv], axis=-1)
    return net

def transition_module(inputs, transition_ratio, scope, pool=True):
    with tf.variable_scope(scope) as sc:
        depth = inputs.get_shape().as_list()[-1] * transition_ratio
        net = slim.conv2d(inputs, depth, [1, 1], stride=1, scope='conv1x1')
        if pool:
            net = slim.avg_pool2d(net, [2, 2], stride=2, padding='SAME', scope='avgpool')
    return net


def backbone(inputs, backbone_cfg, network_cfg, is_training=True, scope='tpn_backbone'):
    end_points = {}
    output_points = []
    net = inputs

    dense_block = cfg_utils.get_cfg_attr(backbone_cfg, 'dense_block', 2)
    bottleneck_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'bottleneck_ratio', 4)
    growth_num = cfg_utils.get_cfg_attr(backbone_cfg, 'growth_num', 64)
    transition_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'transition_ratio', 0.5)
    # transition_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'transition_ratio', 1.0)

    norm_func = cfg_utils.get_norm_func(network_cfg)

    tmp_points = {}
    with slim.arg_scope([norm_func], is_training=is_training):
        with tf.variable_scope(scope, 'tpn_backbone'):

            depth = 64
            with tf.variable_scope('block_1'):
                net = tf_layer.unbalance_conv(net, depth, [3, 3], stride=1, scope='conv1')#[3,3]
                net = tf_layer.unbalance_conv(net, depth, [3, 3], stride=1, scope='conv2')#[5,5]
                net = tf_layer.unbalance_conv(net, depth, [3, 3], stride=2, scope='conv3')#[7,7]
            tmp_points['block1'] = net

            depth = 128
            with tf.variable_scope('block_2'):
                net = tf_layer.unbalance_conv(net, depth, [3, 3], stride=1, scope='conv1')#[11,11]
                net = tf_layer.unbalance_conv(net, depth, [3, 3], stride=1, scope='conv2')#[15,15]
                # net = slim.conv2d(net, depth, [3, 3], stride=2, scope='conv3')#[19,19]
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxpool')
            tmp_points['block2'] = net

            net = dense_module(net, growth_num, bottleneck_ratio, dense_block, 'denseblock1')
            net = transition_module(net, transition_ratio, 'transitblock1')
            tmp_points['block3_0'] = net

            net = dense_module(net, growth_num, bottleneck_ratio, dense_block, 'denseblock2')
            net = transition_module(net, transition_ratio, 'transitblock2', pool=False)
            tmp_points['block3_1'] = net

            depth = 256

            net = tf.concat([tmp_points['block3_0'], tmp_points['block3_1']], axis=-1)
            # net = slim.conv2d(net, depth, [1, 1], stride=1, scope='conv1x1_1')
            end_points['bf_map'] = net            

            output_depth = 512
            
            with tf.variable_scope('block_4'):
                start = tf_layer.unbalance_conv(net, output_depth, [3, 3], stride=1, scope='conv1')#[59,59]
                net = tf_layer.unbalance_conv(start, output_depth/2, [3, 3], stride=1, scope='conv2')#[75,75]
                net = tf_layer.unbalance_conv(net, output_depth/2, [3, 3], stride=1, scope='conv3')

                net = tf.concat([start, net], axis = -1)
                net = tf_layer.unbalance_conv(net, output_depth, [3, 3], stride=1, scope='conv4')

            # end_points['block3'] = [start]
            end_points['block4'] = [net]

            # end_points = xy_module(net, end_points, 'block4')

            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            with tf.variable_scope('block_5'):
                net = tf_layer.unbalance_conv(net, output_depth, [3, 3], stride=1, scope='conv1')#[59,59]
                start = tf_layer.unbalance_conv(net, output_depth, [3, 3], stride=1, scope='conv2')#[75,75]
                net = tf_layer.unbalance_conv(start, output_depth, [5, 5], scope='conv3')#[75,75]

                net = tf.concat([start, net], axis = -1)
                net = tf_layer.unbalance_conv(net, output_depth * 2, [3, 3], stride=1, scope='conv5')
                net = tf_layer.unbalance_conv(net, output_depth * 2, [3, 3], stride=1, scope='conv6')
                # net = slim.conv2d(start, output_depth/2, [3, 3], rate=2, scope='conv3')#[75,75]
                # net = slim.conv2d(net, output_depth/2, [3, 3], rate=2, scope='conv4')#[75,75]

                # net = tf.concat([start, net], axis = -1)
                # net = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv5')

            # with tf.variable_scope('block_5'):
            #     net = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv1')#[59,59]
            #     net = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv2')#[75,75]
            #     net = slim.conv2d(net, output_depth, [3, 3], rate=2, scope='conv4')#[75,75]
                
            end_points['block5'] = [net]

            # end_points = xy_module(net, end_points, 'block5')

            end_point = 'block6'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, output_depth, [1, 1], scope='conv1x1')
                net = tf_layer.unbalance_conv(net, output_depth, [3, 3], stride=1, scope='conv1')
                net = tf_layer.unbalance_conv(net, output_depth, [3, 3], stride=1, scope='conv2')
                net = tf_layer.unbalance_conv(net, output_depth, [3, 3], stride=2, scope='conv3x3')
            end_points[end_point] = [net]
            # end_points = xy_module(net, end_points, 'block6')

            end_point = 'block7'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = tf_layer.unbalance_conv(net, output_depth/2, [3, 3], stride=1, scope='conv1')
                net = tf_layer.unbalance_conv(net, output_depth/2, [3, 3], stride=2, scope='conv3x3')
            end_points[end_point] = [net]
            # end_points = xy_module(net, end_points, 'block7')

            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = tf_layer.unbalance_conv(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = [net]
            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = tf_layer.unbalance_conv(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = [net]


    return end_points