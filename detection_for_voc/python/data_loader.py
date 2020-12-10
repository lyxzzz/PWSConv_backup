import os
import numpy as np
import cv2
from utils import data_utils
from utils import image_augmentation
from utils import new_image_augmentation
from utils import data_management
from bboxes import bboxes_wrapper_np

from register import train_data_dict
from register import test_data_dict

import dataloader_coco.data_loader as COCO_LOADER

def get_train_dataset(dataset_list, load_difficult = 1, percent = None):
    dataset_path = train_data_dict
    dataset = []
    obj_type_nums = 0
    for dataset_name in dataset_list:
        if dataset_name not in dataset_path:
            continue
        dataset = dataset + dataset_path[dataset_name][1].data_loader(dataset_path[dataset_name][0], load_difficult, percent)
        obj_type_nums = dataset_path[dataset_name][2].OBJ_TYPE_NUMS
    return dataset, obj_type_nums

def get_test_dataset(dataset_list, load_difficult = 1):
    dataset_path = test_data_dict
    dataset = []
    obj_type_nums = 0
    id_to_name = None
    for dataset_name in dataset_list:
        if dataset_name not in dataset_path:
            continue
        dataset = dataset + dataset_path[dataset_name][1].data_loader(dataset_path[dataset_name][0], load_difficult)
        obj_type_nums = dataset_path[dataset_name][2].OBJ_TYPE_NUMS
        id_to_name = dataset_path[dataset_name][2].ID_TO_NAME
    return dataset, obj_type_nums, id_to_name

def get_coco_minval():
    para = test_data_dict['COCO']
    coco = para[2].COCO_DATASET(para[0], para[1])
    return coco.coco, coco.dataset, COCO_LOADER.OBJ_TYPE_NUMS, coco.catids

def _get_now_shape(epoch, shape_list, epoch_list):
    for i in range(len(epoch_list)):
        if epoch >= epoch_list[i]:
            result_shape = shape_list[i]
        else:
            return result_shape
    return result_shape

class _TrainGenerator():
    def __init__(self, AUGMENT_PARAMETERS, dataset, 
            positive_threshold, negative_threshold, anchors, prior_scaling, aug_epochs):
        self.total_data_list = dataset
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.anchors = anchors
        self.prior_scaling = prior_scaling
        self.augmentation = new_image_augmentation.Augmentation(AUGMENT_PARAMETERS)
        self.aug_epochs = aug_epochs
    
    def parameterNum(self):
        return 6

    def dataSize(self):
        print('datanum:{}'.format(len(self.total_data_list)))
        return len(self.total_data_list)

    def load_func(self, index, epoch):
        raw_data = self.total_data_list[index]
        path = raw_data[0]

        gt_boxes = np.array(raw_data[3], dtype=np.float32)
        gt_labels = np.array(raw_data[4], dtype=np.int32)

        process_anchors = self.anchors

        # aug_img, aug_boxes = image_augmentation.augmentation_image_tltle(img, gt_boxes)

        if epoch < self.aug_epochs:
            aug_img, aug_bboxes, aug_labels = self.augmentation(path, gt_boxes, gt_labels, islittle=True)
        else:
            aug_img, aug_bboxes, aug_labels = self.augmentation(path, gt_boxes, gt_labels)

        label, loc, scores = bboxes_wrapper_np.bboxes_encode(aug_labels, aug_bboxes, process_anchors, self.positive_threshold, self.negative_threshold, self.prior_scaling)
        
        # times = 0
        # while np.sum(scores>0) <= 0:
        #     times += 1
        #     if times >= 10:
        #         gt_boxes[:,0::2] = gt_boxes[:,0::2] / float(w)
        #         gt_boxes[:,1::2] = gt_boxes[:,1::2] / float(h)
        #         img = cv2.resize(img, (int(process_shape[1]), int(process_shape[0])), interpolation=cv2.INTER_LINEAR)
        #         label, loc, scores = bboxes_wrapper_np.bboxes_encode(gt_labels, gt_boxes, process_anchors, self.positive_threshold, self.negative_threshold, self.prior_scaling)
        #         return [img, label, loc, scores, gt_boxes, gt_labels]
        #     aug_img, aug_bboxes, aug_labels = self.augmentation(img, gt_boxes, gt_labels, True)
        #     label, loc, scores = bboxes_wrapper_np.bboxes_encode(aug_labels, aug_bboxes, process_anchors, self.positive_threshold, self.negative_threshold, self.prior_scaling)
        return [aug_img, label, loc, scores, aug_bboxes, aug_labels]
        
        # gt_boxes[:,0::2] = gt_boxes[:,0::2] / float(w)
        # gt_boxes[:,1::2] = gt_boxes[:,1::2] / float(h)
        # img = cv2.resize(img, (int(process_shape[1]), int(process_shape[0])), interpolation=cv2.INTER_LINEAR)
        # label, loc, scores = bboxes_wrapper_np.bboxes_encode(gt_labels, gt_boxes, process_anchors, self.positive_threshold, self.negative_threshold, self.prior_scaling)
        # return [img, label, loc, scores, gt_boxes, gt_labels]

class _TestGenerator():
    def __init__(self, AUGMENT_PARAMETERS, dataset):
        self.total_data_list = dataset
        self.augmentation = new_image_augmentation.Augmentation(AUGMENT_PARAMETERS)
    
    def parameterNum(self):
        return 6

    def dataSize(self):
        print('datanum:{}'.format(len(self.total_data_list)))
        return len(self.total_data_list)

    def load_func(self, index, epoch):
        raw_data = self.total_data_list[index]
        path = raw_data[0]

        basename = os.path.basename(path)

        gt_boxes = np.array(raw_data[3], dtype=np.float32)
        gt_labels = np.array(raw_data[4], dtype=np.int32)
        gt_difficult = np.array(raw_data[5], dtype=np.int8)

        img, gt_boxes, gt_labels = self.augmentation.test(path, gt_boxes, gt_labels)
        
        h, w = img.shape[0:2]
        
        return [img, gt_boxes, gt_labels, gt_difficult, basename, (h, w)]

def load_train_dataset(epoch, dataset, ROOT_CFG, anchors, aug_epochs=0):
    datacfgs = ROOT_CFG["train_parameters"]
    assignercfgs = ROOT_CFG["assigner"]
    prior_scaling = ROOT_CFG["anchors"]["prior_scaling"]

    batch_nums = datacfgs["train_batch_nums"]
    thread_nums = datacfgs["train_thread_nums"]
    queue_size = datacfgs["train_queue_size"]
    AUGMENT_PARAMETERS = ROOT_CFG["augmentation"]

    gen = _TrainGenerator(AUGMENT_PARAMETERS, dataset, 
                    assignercfgs["positive_threshold"], assignercfgs["negative_threshold"], 
                    anchors, prior_scaling, aug_epochs)

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

def load_test_dataset(dataset, ROOT_CFG):
    epoch = 1

    datacfgs = ROOT_CFG["test_parameters"]
    
    batch_nums = datacfgs["test_batch_nums"]
    thread_nums = datacfgs["test_thread_nums"]
    queue_size = datacfgs["test_queue_size"]
    AUGMENT_PARAMETERS = ROOT_CFG["augmentation"]

    gen = _TestGenerator(AUGMENT_PARAMETERS, dataset)
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