import os
import numpy as np
import math
import pickle
import dataloader_cifar.dataset_cfg as CIFAR_CFG

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

__train_file_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
__test_file_list = ["test_batch"]

def data_loader(root_dir, is_train=True):
    print('*****************************')
    print('[CIFAR]')

    load_file_list = None
    if is_train:
        load_file_list = __train_file_list
    else:
        load_file_list = __test_file_list
    
    image_num = 0
    dataset = []
    for file_name in load_file_list:
        real_path = os.path.join(root_dir, file_name)
        pkg_dataset = unpickle(real_path)

        for i in range(len(pkg_dataset[b'labels'])):
            img = pkg_dataset[b"data"][i].reshape([3, 32, 32])
            img = img.transpose(1, 2, 0)
            label = pkg_dataset[b"labels"][i]

            img_data = [img, label]
            dataset.append(img_data)

    print("total_img:{}".format(len(dataset)))

    return dataset