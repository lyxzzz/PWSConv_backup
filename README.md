# PWSConv_backup
Detection and tensorflow version of our implementation.

## Introduction
We provide four repositories to experiment with different tasks and datasets.
1. [mmdet](mmdet) is an open source object detection repository. We use mmdet to test Faster-RCNN on COCO datasets.

2. [ssd](ssd) is our own repository for SSD experiments on VOC and COCO

3. [classification_for_cifar10](classification_for_cifar10) is our own repository for image classification on CIFAR10.

Our implementation of classification is in https://github.com/lyxzzz/PWSConv

## Results
|Structure | Backbone | Head | Methods | Pretrained  | Schedule | mAP BBOX | mAP Mask |
|  ----  |  ----  | ----  | ----  | :----: | :----: | :----: | :----: |
| MaskRCNN | ResNet50 | Normal | PWS | √ | 12 | 38.9 | 35.3
| MaskRCNN | ResNet50 | Normal | GN | √ | 12 | 38.2 | 34.5
| MaskRCNN | ResNet50 | Normal | BN | √ | 12 | 38.0 | 34.4
| MaskRCNN | ResNet50 | 4Conv | PWS | √ | 12 | 39.5 | 35.6
| MaskRCNN | ResNet50 | 4Conv | GN | √ | 12 | 39.1| 35.3
