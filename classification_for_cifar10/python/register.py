from model.backbone import vgg_backbone
from model.backbone import cifar_vgg_backbone
from model.backbone import cifar_resnet_backbone

from model.loss import softmax_loss

backbone_dict = {
    'cifar_VGG':cifar_vgg_backbone,
    'cifar_Resnet':cifar_resnet_backbone,
    'VGG':vgg_backbone
}

loss_dict = {
    "softmax":softmax_loss
}

IMAGENET = {
    "root":"dataset/imagenet",
    "idmap":"dataset/imagenet/devkit/data/idmap.json"
}

CIFAR10 = {
    "path":"/home/t/dataset/cifar10/cifar-10-batches-py",
}
