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
            'group_size':16
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
                            epsilon=cfg_utils.get_cfg_attr(network_cfg, 'pwsepsilon', 0.001),
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True)):
            with slim.arg_scope([tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d],
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True)):
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

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [factor, factor], stride=factor, scope=scope)

def process_fn(inputs, norm_fn):
    if norm_fn is not None:
        preact = norm_fn(inputs, activation_fn=tf.nn.relu, scope='preact')
    else:
        preact = inputs / math.sqrt(2)
        preact = tf.nn.relu(preact)
    
    return preact

def bottleneck(inputs,
               conv_fn,
               norm_fn,
               depth,
               depth_bottleneck,
               stride,
               scope=None):

    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = inputs.shape.as_list()[-1]
        
        preact = process_fn(inputs, norm_fn)

        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = conv_fn(preact, depth, [1, 1], normalizer_fn=None, activation_fn=None, scope='shortcut')

        residual = conv_fn(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv_fn(residual, depth_bottleneck, 3, stride=stride, scope='conv2')
        residual = conv_fn(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual

        return output

def resnet_block(inputs, scope, conv_fn, norm_fn, base_depth, num_units, stride):
    with tf.variable_scope(scope):
        res_info = [{
                'depth': base_depth * 4,
                'depth_bottleneck': base_depth,
                'stride': 1
            }] * (num_units - 1) + [{
                'depth': base_depth * 4,
                'depth_bottleneck': base_depth,
                'stride': stride
            }]
        
        net = inputs
        for i in range(num_units):
            with tf.variable_scope('unit_{}'.format(i)):
                net = bottleneck(net, conv_fn, norm_fn, **res_info[i])
        
        return net

def backbone(inputs, backbone_cfg, network_cfg, num_classes, is_training=True, scope='vgg_backbone'):

    net = inputs

    norm_func = cfg_utils.get_norm_func(network_cfg)
    conv_fn = cfg_utils.get_conv_func(network_cfg)

    depth_list = [16, 32, 64]
    stride_list = [2, 2, 1]
    num_unit = 6    

    tmp_points = {}
    with slim.arg_scope([norm_func], is_training=is_training):
        with tf.variable_scope(scope, 'resnet_backbone'):

            if norm_func == tf_layer.none_layer:
                norm_func = None
            
            net = conv_fn(net, 64, [3, 3], scope='startconv')

            for i in range(len(depth_list)):
                net = resnet_block(net, "res_{}".format(i), conv_fn, norm_func, depth_list[i], num_unit, stride_list[i])

            net = process_fn(net, norm_func)
            net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
            net = conv_fn(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='mylogits')
            print(net.shape)
            net = tf.reshape(net, [-1, num_classes])

            prediction = slim.softmax(net)
            print(prediction)
    return net, prediction