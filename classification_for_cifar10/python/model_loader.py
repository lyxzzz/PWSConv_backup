import tensorflow as tf
from tensorflow.contrib import slim

from register import backbone_dict
from register import loss_dict

def forward(ROOT_CFG, num_classes, inputs_image, is_training, backbone_name='SSD'):
    if backbone_name not in backbone_dict:
        return 0

    network_cfg = ROOT_CFG.json_cfg.get('network', {})
    backbone_cfg = ROOT_CFG.json_cfg.get('backbone', {})

    print("---------start construct backbone---------")
    with slim.arg_scope(backbone_dict[backbone_name].arg_scope(network_cfg)):
        logits, prediction = backbone_dict[backbone_name].backbone(inputs_image, backbone_cfg, network_cfg, num_classes, is_training)
    print("---------success construct backbone---------")
    
    return logits, prediction

def losses(ROOT_CFG, model_outputs, placeholders, loss_name='normal'):
    if loss_name not in loss_dict:
        return 0
    assert placeholders[2].dtype == tf.uint8

    losses_cfg = ROOT_CFG.json_cfg.get('losses', {})

    print("---------start construct losses---------")
    r = loss_dict[loss_name].losses(model_outputs[0], placeholders[2], losses_cfg)
    print("---------success construct header---------")
    return r

def losses_description(loss_name='normal'):
    if loss_name not in loss_dict:
        return 0
    return loss_dict[loss_name].losses_description()
