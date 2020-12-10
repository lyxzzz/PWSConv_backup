import tensorflow as tf
from tensorflow.contrib import slim

from register import backbone_dict
from register import header_dict
from register import loss_dict
from register import nms_dict
from register import anchor_dict

def forward(ROOT_CFG, num_classes, inputs_image, is_training, backbone_name='SSD', header_name='SSD'):
    if backbone_name not in backbone_dict or header_name not in header_dict:
        return 0

    network_cfg = ROOT_CFG.get('network', {})
    backbone_cfg = ROOT_CFG.get('backbone', {})
    header_cfg = ROOT_CFG.get('header', {})
    anchors_cfg = ROOT_CFG.get('anchors', {})
    anchor_func = anchor_dict[ROOT_CFG["anchors"].get('type', 'default')]

    print("---------start construct backbone---------")
    with slim.arg_scope(backbone_dict[backbone_name].arg_scope(network_cfg)):
        end_points = backbone_dict[backbone_name].backbone(inputs_image, backbone_cfg, network_cfg, is_training)
    print("---------success construct backbone---------")

    print("---------start construct header---------")
    with slim.arg_scope(header_dict[header_name].arg_scope(network_cfg)):
        result = header_dict[header_name].header(end_points, num_classes, header_cfg, network_cfg, [anchors_cfg, anchor_func], is_training)
    print("---------success construct header---------")
    
    return result

def losses(ROOT_CFG, model_outputs, placeholders, loss_name='normal'):
    if loss_name not in loss_dict:
        return 0
    assert placeholders[2].dtype == tf.int32
    assert placeholders[3].dtype == tf.float32
    assert placeholders[4].dtype == tf.int8

    losses_cfg = ROOT_CFG.get('losses', {})
    background_label = ROOT_CFG.get('background_label', 0)

    print("---------start construct losses---------")
    r = loss_dict[loss_name].losses(model_outputs[0],
                                        model_outputs[1],
                                        model_outputs[2],
                                        placeholders[2],
                                        placeholders[3],
                                        placeholders[4],
                                        background_label,
                                        losses_cfg)
    print("---------success construct header---------")
    return r

def losses_description(loss_name='normal'):
    if loss_name not in loss_dict:
        return 0
    return loss_dict[loss_name].losses_description()

def postprocessing(ROOT_CFG, num_classes, anchors, model_outputs, postprocessing_name='nms'):
    if postprocessing_name not in nms_dict:
        return 0

    background_label = ROOT_CFG.get('background_label', 0)
    postprocessing_cfg = ROOT_CFG.get('postprocessing', {})
    anchors_cfg = ROOT_CFG.get('anchors', {})

    print("---------start construct postprocessing---------")
    r = nms_dict[postprocessing_name].postprocessing(model_outputs,
                                        num_classes,
                                        anchors,
                                        background_label,
                                        postprocessing_cfg,
                                        anchors_cfg)
    print("---------success construct postprocessing---------")    
    return r