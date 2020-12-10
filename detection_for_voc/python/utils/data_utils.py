import numpy as np
import cv2
import os

def image_resize_unfixed(img_info, min=600.0, max=800.0):
    h, w, c = img_info
    if h < w:
        scale1 = min / h
        scale2 = max / w
    else:
        scale1 = min / w
        scale2 = max / h
    scale = scale1 if scale1 < scale2 else scale2
    return scale

def image_resize_fixed(img_info, width=320.0, height=320.0):
    h, w = img_info[0:2]
    scale_h = height / h
    scale_w = width / w
    return scale_h,scale_w

def read_bbox_twopoints(annotation_path):
    if annotation_path == None:
        return np.zeros((1,4), dtype='float64')
    bbox = []
    with open(annotation_path, "r", encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            try:
                xmin, ymin, xmax, ymax = map(float, line[0:4])
            except:
                continue
            height = ymax - ymin
            width = xmax - xmin
            if height == 0 or width == 0:
                continue
            bbox.append([xmin,ymin,xmax,ymax])
    if len(bbox) == 0:
        return np.zeros((1,4), dtype='float64')
    else:
        return np.array(bbox, dtype='float64')

def read_bbox_fourpoints(annotation_path):
    bbox = []
    with open(annotation_path, "r", encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            try:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, line[0:8])
            except:
                continue
            xmin = min([x1,x2,x3,x4])
            ymin = min([y1,y2,y3,y4])
            xmax = max([x1,x2,x3,x4])
            ymax = max([y1,y2,y3,y4])
            height = ymax - ymin
            width = xmax - xmin
            if height == 0 or width == 0:
                continue
            bbox.append([xmin,ymin,xmax,ymax])
    return np.array(bbox, dtype='float64')

def read_label(annotation_path):
    with open(annotation_path, "r", encoding = 'utf-8') as f:
        lines = f.readlines()
        label = int(lines[0])   
        return label

def load_data_path(image_src, annotation_src = None):
    data_path_list = []

    image_exts = ['jpg', 'png', 'jpeg', 'JPG']
    annotation_exts = '.txt'

    annotation_path = None

    dataset = os.listdir(image_src)
    for image_name in dataset:
            for ext in image_exts:
                if image_name.endswith(ext):

                    fn, _ = os.path.splitext(image_name)
                    if annotation_src != None:
                        annotation_path = os.path.join(annotation_src, fn + annotation_exts)
                        if not os.path.exists(annotation_path):
                            print("annotation for image {} not exist!".format(image_name))
                            break

                    data_path_list.append((os.path.join(image_src, image_name), annotation_path))
                    break

    print('Find {} images'.format(len(data_path_list)))

    return data_path_list

def map_get_orDefault(map, key, default):
    if key in map:
        return map[key]
    else:
        return default