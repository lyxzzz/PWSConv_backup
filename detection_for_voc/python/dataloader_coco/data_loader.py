import os
from pycocotools.coco import COCO
import numpy as np

OBJ_TYPE_NUMS = 81

class COCO_DATASET():
    def __init__(self, rootDir, dataType):
        self.annFile = '{}/annotations/instances_{}.json'.format(rootDir,dataType)
        self.imgDir = '{}/{}/'.format(rootDir, dataType)
        print(self.annFile, self.imgDir)
        self.coco = COCO(self.annFile)

        self.catids = sorted(self.coco.getCatIds())
        self.catalogs = self.coco.loadCats(self.catids)
        self.catnames = ['none'] + [cat['name'] for cat in self.catalogs]

        self.typetoid = dict()
        for i in range(len(self.catids)):
            self.typetoid[i+1] = self.catids[i]


        imgIds = self.coco.getImgIds()
        print("total img size is {}".format(len(imgIds)))

        self.dataset = []
        for imgid in imgIds:
            img_info = self.coco.loadImgs(imgid)[0]
            img_path = self.imgDir + img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            assert img_info['id'] == int(img_info['file_name'][:-4])
            annIds = self.coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
            anns = self.coco.loadAnns(ids=annIds)

            obj_box = []
            obj_label = []
            obj_difficult = []
            for ann in anns:
                [left, top, width, height] = ann['bbox']

                # some boxes have no width / height
                if height < 1 or width < 1:
                    continue

                obj_box.append([round(left, 2), round(top, 2), round(left + width, 2), round(top + height, 2)])
                obj_label.append(self.catids.index(ann['category_id']) + 1)
                obj_difficult.append(0)
            if len(obj_box) > 0:
                self.dataset.append([img_path, img_width, img_height, obj_box, obj_label, obj_difficult, img_info['id']])

def data_loader(path_list, difficult=0, percent=None):
    coco = COCO_DATASET(path_list[0], path_list[1])
    return coco.dataset
