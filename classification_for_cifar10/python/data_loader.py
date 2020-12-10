import os
import numpy as np
import cv2
import dataloader_cifar.dataset_loader as CIFAR_LOADER
import dataloader_cifar.dataset_cfg as CIFAR_CFG
import dataloader_imagenet.dataset_loader as IMAGENET_LOADER
import dataloader_imagenet.dataset_cfg as IMAGENET_CFG

from utils import data_utils

from utils import new_image_augmentation
from utils import data_management
from bboxes import bboxes_wrapper_np

from register import IMAGENET
from register import CIFAR10

def get_train_dataset(dataset_list):
    dataset_path = {
        'CIFAR10':(CIFAR10['path'], CIFAR_LOADER, CIFAR_CFG),
        'ImageNet':(IMAGENET['root'], IMAGENET_LOADER, IMAGENET_CFG)
    }
    dataset = []
    obj_type_nums = 0

    for dataset_name in dataset_list:
        if dataset_name not in dataset_path:
            continue
        dataset = dataset + dataset_path[dataset_name][1].data_loader(dataset_path[dataset_name][0], is_train=True)
        obj_type_nums = dataset_path[dataset_name][2].OBJ_TYPE_NUMS
        
    return dataset, obj_type_nums

def get_test_dataset(dataset_list):
    dataset_path = {
        'CIFAR10':(CIFAR10['path'], CIFAR_LOADER, CIFAR_CFG),
        'ImageNet':(IMAGENET['root'], IMAGENET_LOADER, IMAGENET_CFG)
    }
    dataset = []
    obj_type_nums = 0
    id_to_name = None
    for dataset_name in dataset_list:
        if dataset_name not in dataset_path:
            continue
        dataset = dataset + dataset_path[dataset_name][1].data_loader(dataset_path[dataset_name][0], is_train=False)
        obj_type_nums = dataset_path[dataset_name][2].OBJ_TYPE_NUMS
        id_to_name = dataset_path[dataset_name][2].ID_TO_NAME

    return dataset, obj_type_nums, id_to_name

def _get_now_shape(epoch, shape_list, epoch_list):
    for i in range(len(epoch_list)):
        if epoch >= epoch_list[i]:
            result_shape = shape_list[i]
        else:
            return result_shape
    return result_shape

class _TrainGenerator():
    def __init__(self, dataset, AUGMENT_PARAMETERS):
        self.total_data_list = dataset
        self.augmentation = new_image_augmentation.Augmentation(AUGMENT_PARAMETERS)
    
    def parameterNum(self):
        return 2

    def dataSize(self):
        print('datanum:{}'.format(len(self.total_data_list)))
        return len(self.total_data_list)

    def load_func(self, index, epoch):
        raw_data = self.total_data_list[index]
        img = raw_data[0]
        label = raw_data[1]

        aug_img, aub_labels = self.augmentation(img, label)

        return [aug_img, aub_labels]

class _TestGenerator():
    def __init__(self, dataset, augment_dict):
        self.total_data_list = dataset
        self.mean = augment_dict['dataset_distribution']['mean']
        self.var = augment_dict['dataset_distribution']['var']
    
    def parameterNum(self):
        return 2

    def dataSize(self):
        print('datanum:{}'.format(len(self.total_data_list)))
        return len(self.total_data_list)

    def load_func(self, index, epoch):
        raw_data = self.total_data_list[index]
        img = raw_data[0]
        label = raw_data[1]

        (h, w) = img.shape[0:2]

        img = img.astype(np.float32)
        img = (img - self.mean) / self.var

        return [img, label]

def load_train_dataset(epoch, dataset, ROOT_CFG, AUGMENT_PARAMETERS):
    datacfgs = ROOT_CFG.data_cfg

    batch_nums = datacfgs.train_batch_nums
    thread_nums = datacfgs.train_thread_nums
    queue_size = datacfgs.train_queue_size

    gen = _TrainGenerator(dataset, AUGMENT_PARAMETERS)

    datapool = data_management.dataPool(thread_num=thread_nums,
                                     batch_num=batch_nums,
                                     queue_size=queue_size, 
                                     data_size=gen.dataSize(), 
                                     item_size=gen.parameterNum(), 
                                     load_func=gen.load_func, 
                                     epoch=epoch)
    
    datapool.start()

    for i in range(epoch):
        data = datapool.getBatchData()
        while data != 0:
            yield data
            data = datapool.getBatchData()
        if i == (epoch - 1):
            datapool.close()
            datapool.join()
            yield 0
        else:
            yield 0
    while True:
        yield 0

def load_test_dataset(dataset, ROOT_CFG, AUGMENT_PARAMETERS):
    epoch = 1

    datacfgs = ROOT_CFG.data_cfg
    
    batch_nums = datacfgs.test_batch_nums
    thread_nums = datacfgs.test_thread_nums
    queue_size = datacfgs.test_queue_size

    gen = _TestGenerator(dataset, AUGMENT_PARAMETERS)
    datapool = data_management.dataPool(thread_num=thread_nums,
                                     batch_num=batch_nums,
                                     queue_size=queue_size, 
                                     data_size=gen.dataSize(), 
                                     item_size=gen.parameterNum(), 
                                     load_func=gen.load_func, 
                                     epoch=1)
    
    datapool.start()

    get_times = 0
    for i in range(epoch):
        data = datapool.getBatchData()
        get_times += 1
        while data != 0:
            yield data
            data = datapool.getBatchData()
            get_times += 1
        if i == (epoch - 1):
            datapool.close()
            datapool.join()
            yield 0
        else:
            yield 0
    while True:
        yield 0