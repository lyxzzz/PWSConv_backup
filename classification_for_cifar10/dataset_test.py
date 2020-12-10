import datetime
import os
import sys
import numpy as np
import json
import cv2
sys.path.append('python')
sys.path.append('python/dataloader_coco')
import data_loader
import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
from utils import new_image_augmentation

def draw_hist(myList,name):
    myList = np.array(myList)
    # myList = myList[np.where(myList<=100)[0]]
    xmin = np.min(myList)
    xmax = np.max(myList)
    plt.hist(myList,100)
    plt.xlabel(name)
    plt.xlim(xmin,xmax)
    plt.ylabel("Number")
    plt.ylim(0,1000)
    title = name + 'distribution'
    plt.title(title)
    plt.savefig('{}.png'.format(title))
    plt.show()

preload_train_dataset, obj_type_nums = data_loader.get_train_dataset(['ImageNet'])

print(len(preload_train_dataset))

index = 0
obj_ratio_list = []
img_ratio_list = []
height_list = []
width_list = []

max_ratio = 0.0
max_path = None
mean_list = []
var_list = []
distribution_calculator = new_image_augmentation.Augmentation({
        "type":"cal_distribution",
        "parameters":{},
        "dataset_distribution":{'mean':0, 'var':0}})
for single_img in preload_train_dataset:
    img, label = single_img
    mean, var = distribution_calculator(img, label)
    mean_list.append(mean)
    var_list.append(var)
    index += 1
    print(index)

mean_list = np.stack(mean_list)
var_list = np.stack(var_list)
total_mean = np.mean(mean_list, axis=0)
total_var = np.mean(var_list, axis=0) + np.var(mean_list - total_mean, axis=0)
print(total_mean)
print(total_var)
# draw_hist(height_list, 'obj_ratios')
# draw_hist(width_list, 'img_ratios')

# print(max_ratio)
# print(max_path)