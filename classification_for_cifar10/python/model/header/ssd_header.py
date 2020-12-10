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
        batch_norm_params = None
        norm_function = None
    
    if not network_cfg.get('header_norm', False):
        batch_norm_params = None
        norm_function = None

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
                                        normalizer_params=batch_norm_params) as norm_sc:
                        print('!!!use normalization!!!')
                        return norm_sc

def _post_norm(outputs, num_anchors, last_layer_norm_func):
    if last_layer_norm_func is not None:
        print('use last layer norm')
        if last_layer_norm_func == tf_layer.group_norm:
            outputs = last_layer_norm_func(outputs, num_anchors, activation_fn=None)
        else:
            outputs = last_layer_norm_func(outputs, activation_fn=None)
    return outputs

def _multibox_layer(inputs,
                    is_training,
                    num_classes,
                    normalization,
                    feat_offset_ratio,
                    ratios=[1],
                    last_layer_norm_func=None):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = 1 + len(ratios)
    if feat_offset_ratio != 1:
            num_anchors = num_anchors * feat_offset_ratio

    N, H, W, C = net.shape.as_list()
    num_loc_pred = num_anchors * 4

    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, normalizer_fn=None,scope='conv_loc')
    loc_pred = _post_norm(loc_pred, num_anchors, last_layer_norm_func)
    loc_pred = tf.reshape(loc_pred,
                        tf_utils.tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, normalizer_fn=None,scope='conv_cls')
    cls_pred = _post_norm(cls_pred, num_anchors, last_layer_norm_func)
    cls_pred = tf.reshape(cls_pred,
                        tf_utils.tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    
    if feat_offset_ratio != 1:
        loc_pred = tf.split(loc_pred,feat_offset_ratio,axis=3)
        loc_pred = tf.concat(loc_pred, axis = 2)

        cls_pred = tf.split(cls_pred,feat_offset_ratio,axis=3)
        cls_pred = tf.concat(cls_pred, axis = 2)
    
    # print(cls_pred.shape)
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
    else:
        return None

def header(end_points, num_classes, header_cfg, network_cfg, anchors_cfg, is_training=True, scope='ssd_header'):

    norm_func = cfg_utils.get_norm_func(network_cfg)
    last_layer_norm = cfg_utils.get_cfg_attr(header_cfg, 'last_layer_norm', 'None')
    last_layer_norm_func = get_last_layer_func(last_layer_norm)

    with slim.arg_scope([norm_func], is_training=is_training):
        with tf.variable_scope(scope, 'ssd_300_vgg'):
            predictions = []
            logits = []
            localisations = []
            for i, layer in enumerate(anchors_cfg.feat_layers):
                with tf.variable_scope(layer + '_box'):
                    p, l = _multibox_layer(end_points[layer],
                                                is_training,
                                                num_classes,
                                                anchors_cfg.normalizations[i],
                                                anchors_cfg.feat_offset_ratio[i],
                                                anchors_cfg.anchor_ratios[i],
                                                last_layer_norm_func)
                predictions.append(slim.softmax(p))
                logits.append(p)
                localisations.append(l)
    return predictions, localisations, logits