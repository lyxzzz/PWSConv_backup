import os
import numpy as np
import math
import dataloader_voc.voc_cfg as VOC_CFG
# def data_loader(load_path, load_difficult=0):
#     preprocessing_path = load_path
#     dataset = []
#     print('*****************************')
#     print('[VOC] [{}]'.format(os.path.basename(load_path)))
#     with open(preprocessing_path, "r", encoding = 'utf-8') as f:
#         lines = f.readlines()
#         print('Images: {}'.format(len(lines)))
#         obj_label_nums = np.zeros((VOC_CFG.OBJ_TYPE_NUMS,), dtype=np.int)
#         for line in lines:
#             if len(line) < 10:
#                 continue
#             line = line.split(",")
#             path = line[0]
#             width = int(line[1])
#             height = int(line[2])
#             objects_bbox = []
#             objects_label = []
#             objects_difficult = []
#             object_nums = int((len(line) - 3) / 6)
#             for i in range(object_nums):
#                 label = int(line[3 + i * 6 + 0])
#                 difficult = int(line[3 + i * 6 + 1])
#                 xmin = int(line[3 + i * 6 + 2])
#                 ymin = int(line[3 + i * 6 + 3])
#                 xmax = int(line[3 + i * 6 + 4])
#                 ymax = int(line[3 + i * 6 + 5])
            
#                 obj_width = xmax - xmin
#                 obj_height = ymax - ymin
#                 if width == 0 or height == 0:
#                     continue
#                 if load_difficult < difficult:
#                     continue
#                 objects_bbox.append([xmin, ymin, xmax, ymax])
#                 objects_label.append(label)
#                 objects_difficult.append(difficult)

#                 obj_label_nums[label] += 1
#             img_data = [path, width, height, objects_bbox, objects_label, objects_difficult]
#             # img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),1)
#             # assert img.shape[0] == 1800
#             # assert img.shape[1] == 3200
#             dataset.append(img_data)
#         print(obj_label_nums)
#         print('ObjectNums: {}'.format(np.sum(obj_label_nums)))
#         print('*****************************')
#     return dataset

def data_loader(load_path, load_difficult=0, percent=None):
    preprocessing_path = load_path
    dataset = []
    print('*****************************')
    print('[VOC] [{}]'.format(os.path.basename(load_path)))
    with open(preprocessing_path, "r", encoding = 'utf-8') as f:
        lines = f.readlines()
        print('Images: {}'.format(len(lines)))
        obj_label_nums = np.zeros((VOC_CFG.OBJ_TYPE_NUMS,), dtype=np.int)
        for line in lines:
            if len(line) < 10:
                continue
            line = line.split(",")
            path = line[0]
            width = int(line[1])
            height = int(line[2])
            objects_bbox = []
            objects_label = []
            objects_difficult = []
            object_nums = int((len(line) - 3) / 6)
            for i in range(object_nums):
                label = int(line[3 + i * 6 + 0])
                difficult = int(line[3 + i * 6 + 1])
                xmin = int(line[3 + i * 6 + 2])
                ymin = int(line[3 + i * 6 + 3])
                xmax = int(line[3 + i * 6 + 4])
                ymax = int(line[3 + i * 6 + 5])
            
                obj_width = xmax - xmin
                obj_height = ymax - ymin
                if width == 0 or height == 0:
                    continue
                if load_difficult < difficult:
                    continue
                objects_bbox.append([xmin, ymin, xmax, ymax])
                objects_label.append(label)
                objects_difficult.append(difficult)

                obj_label_nums[label] += 1
            img_data = [path, width, height, objects_bbox, objects_label, objects_difficult]
            # img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),1)
            # assert img.shape[0] == 1800
            # assert img.shape[1] == 3200
            dataset.append(img_data)
        if percent is None:
            print(obj_label_nums)
            print('ObjectNums: {}'.format(np.sum(obj_label_nums)))
            print('*****************************')
        else:
            obj_label_nums = np.zeros((VOC_CFG.OBJ_TYPE_NUMS,), dtype=np.int)
            total_dataset_nums = len(dataset)
            total_image_nums = int(math.ceil(len(dataset) * percent))
            newdataset = []
            indexset = []
            step = int(1 / percent)
            now_index = 0
            for i in range(total_image_nums):
                while now_index in indexset:
                    now_index = (now_index + 1) % total_dataset_nums
                newdataset.append(dataset[now_index])
                indexset.append(now_index)
                for obj_label in dataset[now_index][4]:
                    obj_label_nums[obj_label] += 1

                now_index = (now_index + step) % total_dataset_nums
            print(obj_label_nums)
            print('ObjectNums: {}'.format(np.sum(obj_label_nums)))
            print('*****************************')
            dataset = newdataset
    return dataset