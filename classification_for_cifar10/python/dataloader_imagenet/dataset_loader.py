import os
import numpy as np
import math
import pickle
import dataloader_imagenet.dataset_cfg as CFG

def __load_train(path):
    cls_list = os.listdir(path)
    dataset = []
    for cls_name in cls_list:
        cls_id = CFG.NAME_TO_ID[cls_name] - 1
        cls_path = "{}/{}".format(path, cls_name)
        image_list = os.listdir(cls_path)
        for image_name in image_list:
            image_path = "{}/{}".format(cls_path, image_name)
            image_data = [image_path, cls_id]
            dataset.append(image_data)
    print("total_img:{}".format(len(dataset)))
    return dataset

def data_loader(root_dir, is_train=True):
    print('*****************************')
    print('[IMAGENET]')

    if is_train:
        load_path = "{}/train".format(root_dir)
        return __load_train(load_path)
    else:
        load_path = "{}/val".format(root_dir)
        return __load_train(load_path)