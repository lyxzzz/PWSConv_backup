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

    if not network_cfg.get('header_activation', False):
        print("not use act")
        ha_fn = None
    else:
        ha_fn = tf.nn.relu

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=ha_fn,
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

def _locsubnet(net, output_nums, subnet_nums, subnet_channels, norm_func, last_layer_norm_func, num_anchors):
    net_channels = net.get_shape().as_list()[-1]
    # net_channels = 512
    net = slim.conv2d(net, net_channels, 1, stride=1, padding="SAME",
                                  scope='loc_bottleneck')
    for i in range(subnet_nums):
        net = slim.conv2d(net, subnet_channels, 3, stride=1, padding="SAME",
                                  scope='loc_{}'.format(i))
    outputs = slim.conv2d(net, output_nums, 3,
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

def _clssubnet(net, output_nums, subnet_nums, subnet_channels, norm_func, last_layer_norm_func, num_anchors):
    net_channels = net.get_shape().as_list()[-1]
    # net_channels = 512
    # net = slim.conv2d(net, net_channels, 1, stride=1, padding="SAME",
    #                             scope='cls_bottleneck')
    for i in range(subnet_nums):
        net = slim.conv2d(net, subnet_channels, 3, stride=1, padding="SAME",
                                  scope='cls_{}'.format(i))
    outputs = slim.conv2d(net, output_nums, 3,
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
                    norm_func,
                    is_training,
                    num_classes,
                    ratios,
                    subnet_nums,
                    subnet_channels,
                    bf_map,
                    last_layer_norm_func):

    net = inputs
    # Number of anchors.
    num_anchors = len(ratios)

    N, H, W, C = net.shape.as_list()
    num_loc_pred = num_anchors * 4

    net = tf.concat([net, bf_map], axis = -1)
    
    loc_pred = _locsubnet(net, num_loc_pred, subnet_nums, subnet_channels, norm_func, last_layer_norm_func, num_anchors)
    loc_pred = _reshape(loc_pred, num_anchors, 4)

    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    # last version
    # net = tf.concat([net, bf_map], axis = -1)
    cls_pred = _clssubnet(net, num_cls_pred, subnet_nums, subnet_channels, norm_func, last_layer_norm_func, num_anchors)
    cls_pred = _reshape(cls_pred, num_anchors, num_classes)

    return cls_pred, loc_pred

def gen_bfmap(net, feat_shapes, stop_gradient):
    if stop_gradient:
        net = tf.stop_gradient(net)
    bfmap_list = []
    avg_times = 0
    for feat_shape in feat_shapes:
        raw_shape = net.get_shape().as_list()[1:3]
        if raw_shape[0] == feat_shape[0] and raw_shape[1] == feat_shape[1]:
            # net = tf.stop_gradient(net)
            bfmap_list.append(net)
        else:
            stride_val = int(math.ceil(raw_shape[0] / feat_shape[0]))
            net = slim.avg_pool2d(net, [stride_val, stride_val], stride=stride_val, padding='SAME', scope='bfavgpool{}'.format(avg_times))
            # net = tf.stop_gradient(net)
            bfmap_list.append(net)     
    print(bfmap_list)
    return bfmap_list     

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

def header(end_points, num_classes, header_cfg, network_cfg, anchors_cfg, is_training=True, scope='tpn_header'):

    subnet_nums = cfg_utils.get_cfg_attr(header_cfg, 'subnet_nums', 0)
    subnet_channels = cfg_utils.get_cfg_attr(header_cfg, 'subnet_channels', 256)
    stop_gradient = cfg_utils.get_cfg_attr(header_cfg, 'stop_gradient', False)

    print(header_cfg)
    last_layer_norm = cfg_utils.get_cfg_attr(header_cfg, 'last_layer_norm', 'None')
    last_layer_norm_func = get_last_layer_func(last_layer_norm)

    norm_func = cfg_utils.get_norm_func(network_cfg)

    with slim.arg_scope([norm_func], is_training=is_training):
        with tf.variable_scope(scope, 'tpn_header'):
            bfmap_list = gen_bfmap(end_points['bf_map'], anchors_cfg.feat_shapes, stop_gradient)
            predictions = []
            logits = []
            localisations = []
            for i, feat_layer in enumerate(anchors_cfg.feat_layers):
                # if anchors_cfg.extra_layer[i] != 1:
                #     end_points_layer = tf.concat(end_points[feat_layer], axis = -1)
                # else:
                end_points_layer = end_points[feat_layer][0]
                
                ratios_layer = []
                for layer_index in range(anchors_cfg.extra_layer[i]):
                    ratios_layer += anchors_cfg.anchor_ratios[layer_index]

                uniq_name = '{}_box'.format(feat_layer)
                # with tf.variable_scope('{}_box'.format(anchors_cfg.extra_name[layer_index]), reuse=tf.AUTO_REUSE):
                with tf.variable_scope(uniq_name):
                    p, l= _multibox_layer(end_points_layer,
                                                norm_func,
                                                is_training,
                                                num_classes,
                                                ratios_layer,
                                                subnet_nums,
                                                subnet_channels,
                                                bfmap_list[i],
                                                last_layer_norm_func)
                    predictions.append(slim.softmax(p))
                    logits.append(p)
                    localisations.append(l)
    return predictions, localisations, logits