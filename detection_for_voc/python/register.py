from model.backbone import ssd_backbone
from model.backbone import densenet
from model.backbone import tpn_backbone
from model.backbone import new_backbone

from model.header import ssd_header
from model.header import stdn_header
from model.header import tpn_header

from model.loss import normal_loss
from model.loss import ssd_loss
from model.loss import new_ssd_loss
from model.loss import focal_loss

from model.postprocessing import nms
from model.postprocessing import fastnms
from model.postprocessing import ssdnms

backbone_dict = {
    'SSD':ssd_backbone,
    'DenseNet':densenet,
    'TPN':tpn_backbone,
    'New':new_backbone
}
header_dict = {
    'SSD':ssd_header,
    'STDN':stdn_header,
    'TPN':tpn_header
}
loss_dict = {
    'normal':normal_loss,
    'SSD':ssd_loss,
    'NewSSD':new_ssd_loss,
    'focal':focal_loss
}
nms_dict = {
    'nms':nms,
    'fastnms':fastnms,
    'ssdnms':ssdnms
}

import dataloader_voc.voc_loader as VOC_LOADER
import dataloader_voc.voc_cfg as VOC_CFG
import dataloader_coco.data_loader as COCO_LOADER

train_data_dict = {
    'VOC07':('/home/t/dataset/voc/VOC2007_train.csv', VOC_LOADER, VOC_CFG),
    'VOC12':('/home/t/dataset/voc/VOC2012_train.csv', VOC_LOADER, VOC_CFG),
    'COCO':(["../dataset/COCO", "train2017"], COCO_LOADER, COCO_LOADER)
}

test_data_dict = {
    'VOC07':('/home/t/dataset/voc/VOC2007_test.csv', VOC_LOADER, VOC_CFG),
    'COCO':("../dataset/COCO", "val2017", COCO_LOADER)
}

from bboxes import fpn_anchors
from bboxes import tpn_anchors
from bboxes import anchors

anchor_dict = {
    'default':anchors,
    'TPN':tpn_anchors,
    'fpn':fpn_anchors
}
