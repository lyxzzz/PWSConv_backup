import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from utils import tf_layer
from utils import tf_utils
from utils import custom_layers
from utils import cfg_utils

def arg_scope(network_cfg):
    weight_decay = network_cfg.get('weight_decay', 0.0005)

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

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME'):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last]) as sc:
                if norm_function == None:
                    return sc
                else:
                    with slim.arg_scope([slim.conv2d],normalizer_fn=norm_function,
                                        normalizer_params=batch_norm_params):
                        with slim.arg_scope([norm_function], **batch_norm_params) as norm_sc:
                            print('!!!use normalization!!!')
                            return norm_sc

def xy_module(net, end_points, scope_name):
    x_scope = '{}_x'.format(scope_name)
    y_scope = '{}_y'.format(scope_name)

    net_depth = net.shape.as_list()[-1] / 4

    with tf.variable_scope(scope_name):
        bottleneck = slim.conv2d(net, net_depth, [1, 1], stride=1, scope='conv_bottleneck')

    with tf.variable_scope(x_scope):
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(bottleneck, net_depth, [1, 3], stride=1, scope='conv1x3_1')
            branch_1 = slim.conv2d(branch_1, net_depth, [1, 3], stride=1, scope='conv1x3_2')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(bottleneck, net_depth, [1, 3], rate=2, scope='aconv1x3_1')
            branch_2 = slim.conv2d(branch_2, net_depth, [1, 3], rate=2, scope='aconv1x3_2')
        x_net = tf.concat([branch_1, branch_2], axis = -1)
        x_net = slim.conv2d(x_net, net_depth, [1, 1], stride=1, scope='{}_concat'.format(x_scope))
        
    end_points[scope_name].append(x_net)

    with tf.variable_scope(y_scope):
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(bottleneck, net_depth, [3, 1], stride=1, scope='conv3x1_1')
            branch_1 = slim.conv2d(branch_1, net_depth, [3, 1], stride=1, scope='conv3x1_2')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(bottleneck, net_depth, [3, 1], rate=2, scope='aconv3x1_1')
            branch_2 = slim.conv2d(branch_2, net_depth, [3, 1], rate=2, scope='aconv3x1_2')
        
        y_net = tf.concat([branch_1, branch_2], axis = -1)
        y_net = slim.conv2d(y_net, net_depth, [1, 1], stride=1, scope='{}_concat'.format(y_scope))
    end_points[scope_name].append(y_net)
    return end_points

def dense_module(inputs, growth_num, bottleneck_ratio, dense_num, scope, rate=1):
    net = inputs
    with tf.variable_scope(scope):
        for i in range(dense_num):
            with tf.variable_scope('block{}'.format(i)) as sc:
                dense_conv = slim.conv2d(net, growth_num * bottleneck_ratio, [1, 1], stride=1, scope='conv1x1')
                dense_conv = slim.conv2d(dense_conv, growth_num, [3, 3], stride=1, rate=rate, scope='conv3x3')
                net = tf.concat([net, dense_conv], axis=-1)
    return net

def transition_module(inputs, transition_ratio, norm_func, scope, pool=True):
    with tf.variable_scope(scope) as sc:
        depth = inputs.get_shape().as_list()[-1] * transition_ratio
        net = slim.conv2d(inputs, depth, [1, 1], activation_fn=None,
                               normalizer_fn=None, stride=1, scope='conv1x1')
        if pool:
            net = slim.avg_pool2d(net, [2, 2], stride=2, padding='SAME', scope='avgpool')
        
        net = norm_func(net, activation_fn=tf.nn.relu)
    return net


def backbone(inputs, backbone_cfg, network_cfg, is_training=True, scope='tpn_backbone'):
    end_points = {}
    output_points = []
    net = inputs

    dense_block = cfg_utils.get_cfg_attr(backbone_cfg, 'dense_block', 6)
    bottleneck_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'bottleneck_ratio', 4)
    growth_num = cfg_utils.get_cfg_attr(backbone_cfg, 'growth_num', 32)
    transition_ratio = cfg_utils.get_cfg_attr(backbone_cfg, 'transition_ratio', 0.5)

    norm_func = cfg_utils.get_norm_func(network_cfg)

    tmp_points = {}
    with slim.arg_scope([norm_func], is_training=is_training):
        with tf.variable_scope(scope, 'tpn_backbone'):

            depth = 64
            with tf.variable_scope('block_1'):
                net = slim.conv2d(net, depth, [3, 3], stride=1, scope='conv1')#[3,3]
                net = slim.conv2d(net, depth, [3, 3], stride=1, scope='conv2')#[5,5]
                net = slim.conv2d(net, depth, [3, 3], stride=2, scope='conv3')#[7,7]
            tmp_points['block1'] = net

            depth = 128
            with tf.variable_scope('block_2'):
                net = slim.conv2d(net, depth, [3, 3], stride=1, scope='conv1')#[11,11]
                net = slim.conv2d(net, depth, [3, 3], stride=1, scope='conv2')#[15,15]
                net = slim.conv2d(net, depth, [3, 3], stride=2, scope='conv3')#[19,19]
            tmp_points['block2'] = net

            net = dense_module(net, growth_num, bottleneck_ratio, dense_block, 'denseblock1')
            net = transition_module(net, transition_ratio, 'transitblock1')
            tmp_points['block3_0'] = net

            net = dense_module(net, growth_num, bottleneck_ratio, dense_block, 'denseblock2')
            net = transition_module(net, transition_ratio, 'transitblock2', pool=False)
            tmp_points['block3_1'] = net

            depth = 256

            net = tf.concat([tmp_points['block3_0'], tmp_points['block3_1']], axis=-1)
            net = slim.conv2d(net, depth, [1, 1], stride=1, scope='conv1x1')
            end_points['bf_map'] = net

            output_depth = 512
            
            with tf.variable_scope('block_4'):
                start = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv1')#[59,59]
                net = slim.conv2d(start, output_depth/2, [3, 3], stride=1, scope='conv2')#[75,75]
                net = slim.conv2d(net, output_depth/2, [3, 3], stride=1, scope='conv3')

                net = tf.concat([start, net], axis = -1)
                net = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv4')

            end_points['block4'] = [net]

            # end_points = xy_module(net, end_points, 'block4')

            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            with tf.variable_scope('block_5'):
                start = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv1')#[59,59]
                net = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv2')#[75,75]
                net = slim.conv2d(net, output_depth, [3, 3], rate=2, scope='conv4')#[75,75]
                net = tf.concat([start, net], axis = -1)
                net = slim.conv2d(net, output_depth * 2, [3, 3], stride=1, scope='conv5')
                net = slim.conv2d(net, output_depth * 2, [3, 3], stride=1, scope='conv6')

            end_points['block5'] = [net]

            end_points = xy_module(net, end_points, 'block5')

            end_point = 'block6'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, output_depth, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv1')
                net = slim.conv2d(net, output_depth, [3, 3], stride=1, scope='conv2')
                net = slim.conv2d(net, output_depth, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = [net]
            end_points = xy_module(net, end_points, 'block6')

            end_point = 'block7'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, output_depth/2, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, output_depth/2, [3, 3], stride=1, scope='conv1')
                net = slim.conv2d(net, output_depth/2, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = [net]
            end_points = xy_module(net, end_points, 'block7')

            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = [net]
            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = [net]
    return end_points