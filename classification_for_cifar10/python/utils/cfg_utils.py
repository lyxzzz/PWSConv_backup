from utils import tf_layer
import tensorflow as tf
from tensorflow.contrib import slim

def get_cfg_attr(CFG, name, default_value):
    r = CFG.get(name, default_value)
    print('{}:{}'.format(name, r))
    return r

def get_norm_func(network_cfg):
    norm_func = network_cfg.get('norm_func',"None")
    if norm_func == 'LayerNorm':
        print("use layer normalization")
        return tf_layer.layer_norm
    elif norm_func == 'BatchNorm':
        print("use batch normalization")
        return slim.batch_norm
    elif norm_func == 'GroupNorm':
        print("use group normalization")
        return tf_layer.group_norm
    elif norm_func == 'InstanceNorm':
        print("use instance normalization")
        return tf_layer.instance_norm
    elif norm_func == 'SwitchNorm':
        print("use switch normalization")
        return tf_layer.switch_norm
    else:
        print("not use normalization")
        return tf_layer.none_layer

def get_conv_func(network_cfg):
    conv_func = network_cfg.get('conv_type',"None")
    if conv_func == 'PWS':
        print("use PWS conv")
        return tf_layer.pws_conv
    elif conv_func == 'WN':
        print("use WN conv")
        return tf_layer.wn_conv
    elif conv_func == 'WS':
        print("use WS conv")
        return tf_layer.ws_conv
    elif conv_func == 'MY':
        print("use test conv")
        return tf_layer.myconv        
    else:
        print("use norm conv")
        return slim.conv2d