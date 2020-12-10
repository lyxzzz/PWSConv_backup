from xml.dom.minidom import parse
import xml.dom.minidom as minidom
import voc_cfg
import voc2007_cfg
import os
import numpy as np

def judge_int(str):
    try:
        f = int(str)
    except ValueError:    
        f = float(str)
        f = int(f + 0.5)
    return f

def process_vocdataset(voc_type, voc_date):
    preprocessing_file_name = 'dataset/VOC{}_{}.csv'.format(voc_date, voc_type)
    preprocessing_file = open(preprocessing_file_name, 'w')
    img_root_dir = 'D:/LYX/obj_det/dataset/VOC/VOC{}/VOC{}/JPEGImages'.format(voc_type, voc_date)
    xml_root_dir = 'dataset/VOC/VOC{}/VOC{}/Annotations'.format(voc_type, voc_date)
    if voc_type == 'train':
        trainval_file_path = 'dataset/VOC/VOC{}/VOC{}/ImageSets/Main/trainval.txt'.format(voc_type, voc_date)
        trainval_img_list = []
        with open(trainval_file_path, 'r') as trainval_file:
            lines = trainval_file.readlines()
            for line in lines:
                trainval_img_list.append(line[:-1]+'.jpg')

    print(preprocessing_file_name)
    print(xml_root_dir)
    xml_list = os.listdir(xml_root_dir)
    index = 0
    total_length = len(xml_list)
    block_length = int(total_length / 50)
    # difficult_num = 0
    for xml_name in xml_list:
        index += 1
        str_speed = [' '] * 50
        for i in range(1 + index//block_length):
            if i < 50:
                str_speed[i] = '='
        str_speed = '['+ ''.join(str_speed) + ']'
        print('\rprocessing data[{}/{}]: '.format(index, total_length)+str_speed+'  '+str(np.round(100*index/total_length, 2))+'%', end='')
        # print(index)
        xml_path = os.path.join(xml_root_dir, xml_name)

        with open(xml_path, 'r', encoding='utf8') as f:
            dom = minidom.parse(f)
            root=dom.documentElement

            img_name = root.getElementsByTagName('filename')[0].childNodes[0].data
            if voc_type == 'train':
                if img_name not in trainval_img_list:
                    continue
            img_path = os.path.join(img_root_dir, img_name)

            img_shape_node = root.getElementsByTagName('size')[0]
            width = img_shape_node.getElementsByTagName('width')[0].childNodes[0].data
            height = img_shape_node.getElementsByTagName('height')[0].childNodes[0].data

            preprocessing_file.write('{},{},{}'.format(img_path, width, height))

            object_nodes = root.getElementsByTagName('object')
            for obj_node in object_nodes:
                obj_type = obj_node.getElementsByTagName('name')[0].childNodes[0].data
                difficult_node = obj_node.getElementsByTagName('difficult')
                if len(difficult_node) > 0:
                    obj_difficult = int(difficult_node[0].childNodes[0].data)
                else:
                    obj_difficult = 0
                # difficult_num += obj_difficult

                obj_bbox_node = obj_node.getElementsByTagName('bndbox')[0]
                
                obj_xmin = judge_int(obj_bbox_node.getElementsByTagName('xmin')[0].childNodes[0].data)
                obj_ymin = judge_int(obj_bbox_node.getElementsByTagName('ymin')[0].childNodes[0].data)
                obj_xmax = judge_int(obj_bbox_node.getElementsByTagName('xmax')[0].childNodes[0].data)
                obj_ymax = judge_int(obj_bbox_node.getElementsByTagName('ymax')[0].childNodes[0].data)
                # print((obj_type, obj_difficult, obj_xmin, obj_ymin, obj_xmax, obj_ymax))
                preprocessing_file.write(',{},{},{},{},{},{}'.format(voc_cfg.NAME_TO_ID[obj_type], obj_difficult, obj_xmin, obj_ymin, obj_xmax, obj_ymax))
            preprocessing_file.write('\n')
    preprocessing_file.close()
    print('\ncomlete')

process_vocdataset("train", "2007")
process_vocdataset("train", "2012")
process_vocdataset("test", "2007")
# print("difficult: {}".format(difficult_num))
# with open(xml_path, 'r', encoding='utf8') as f:
#     dom = minidom.parse(f)
#     root=dom.documentElement
#     filename_node = root.getElementsByTagName('filename')
#     print(filename_node)