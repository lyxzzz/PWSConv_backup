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
    else:
        batch_norm_params = None
        norm_function = None
    
    if not network_cfg.get('header_norm', False):
        batch_norm_params = None
        norm_function = None

    with slim.arg_scope([tf_layer.pws_conv, tf_layer.wn_conv, tf_layer.ws_conv, slim.conv2d, tf_layer.myconv],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([tf_layer.pws_conv, tf_layer.wn_conv],
                            epsilon=cfg_utils.get_cfg_attr(network_cfg, 'pwsepsilon', 0.001),
                            single_beta=True,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True)):
            with slim.arg_scope([tf_layer.ws_conv, slim.conv2d, tf_layer.myconv],
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

def _locsubnet(net, output_nums, last_layer_norm_func, num_anchors, conv_func):
    net_channels = net.get_shape().as_list()[-1]
    outputs = conv_func(net, output_nums, 3,
                               activation_fn=None,
                               normalizer_fn=None,
                               padding="SAME",
                               scope="loc_pred")

    if last_layer_norm_func is not None:
        print('use last layer norm')
        if last_layer_norm_func == slim.batch_norm:
            scopename = 'lastbatchlocnorm'
        else:
            scopename = 'lastlocnorm'

        if last_layer_norm_func == tf_layer.group_norm:
            outputs = last_layer_norm_func(outputs, num_anchors, activation_fn=None, scope=scopename)
        else:
            outputs = last_layer_norm_func(outputs, activation_fn=None, scope=scopename)
    return outputs

def _clssubnet(net, output_nums, last_layer_norm_func, num_anchors, conv_func):
    net_channels = net.get_shape().as_list()[-1]

    outputs = conv_func(net, output_nums, 3,
                               activation_fn=None,
                               normalizer_fn=None,
                               padding="SAME",
                               scope="cls_pred")

    if last_layer_norm_func is not None:
        print('use last layer norm')
        if last_layer_norm_func == slim.batch_norm:
            scopename = 'lastbatchclsnorm'
        else:
            scopename = 'lastclsnorm'

        if last_layer_norm_func == tf_layer.group_norm:
            outputs = last_layer_norm_func(outputs, num_anchors, activation_fn=None, scope=scopename)
        else:
            outputs = last_layer_norm_func(outputs, activation_fn=None, scope=scopename)
    return outputs

def _reshape(inputs, num_anchors, num_classes):
    inputs = tf.reshape(inputs,
                        tf_utils.tensor_shape(inputs, 4)[:-1]+[num_anchors, num_classes])
    
    middle_nums = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
    outputs = tf.reshape(inputs,[-1, middle_nums, num_classes])

    return outputs

def _multibox_layer(inputs,
                    is_training,
                    num_classes,
                    num_anchors,
                    last_layer_norm_func,
                    conv_func):

    N, H, W, C = inputs.shape.as_list()
    num_loc_pred = num_anchors * 4
    loc_pred = _locsubnet(inputs, num_loc_pred, last_layer_norm_func, num_anchors, conv_func)
    loc_pred = _reshape(loc_pred, num_anchors, 4)

    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = _clssubnet(inputs, num_cls_pred, last_layer_norm_func, num_anchors, conv_func)
    cls_pred = _reshape(cls_pred, num_anchors, num_classes)

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

def header(end_points, num_classes, header_cfg, network_cfg, anchors_cfg, is_training=True, scope='ssd_header'):

    print(header_cfg)
    last_layer_norm = cfg_utils.get_cfg_attr(header_cfg, 'last_layer_norm', 'None')
    last_layer_norm_func = get_last_layer_func(last_layer_norm)

    norm_func = cfg_utils.get_norm_func(network_cfg)
    conv_func = cfg_utils.get_conv_func(network_cfg)
    anchors_cfg, anchor_func = anchors_cfg

    with slim.arg_scope([norm_func], is_training=is_training):
        with tf.variable_scope(scope, 'ssd_header'):
            predictions = []
            logits = []
            localisations = []
            for i, feat_layer in enumerate(anchors_cfg["feat_layers"]):
                end_points_layer = end_points[feat_layer]
                
                num_anchors = anchor_func.anchors_per_layer(anchors_cfg, i)

                uniq_name = '{}_box'.format(feat_layer)
                with tf.variable_scope(uniq_name):
                    p, l= _multibox_layer(end_points_layer,
                                                is_training,
                                                num_classes,
                                                num_anchors,
                                                last_layer_norm_func,
                                                conv_func)
                    predictions.append(slim.softmax(p))
                    logits.append(p)
                    localisations.append(l)
    return predictions, localisations, logits