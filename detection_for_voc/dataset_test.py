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

preload_train_dataset, obj_type_nums = data_loader.get_train_dataset(['VOC07', 'VOC12'], 1)

print(len(preload_train_dataset))

index = 0
obj_ratio_list = []
img_ratio_list = []
height_list = []
width_list = []

max_ratio = 0.0
max_path = None
mean = np.zeros((3,), dtype=np.float)
val = np.zeros((3,), dtype=np.float)

for single_img in preload_train_dataset:
    path, width, height, objects_bbox, objects_label, objects_difficult = single_img
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape([-1, 3])
    val += np.sqrt(np.var(img, axis=0))
    mean += np.mean(img, axis=0)
    index += 1
    print(index)

print(val)
print(mean)
# draw_hist(height_list, 'obj_ratios')
# draw_hist(width_list, 'img_ratios')

# print(max_ratio)
# print(max_path)