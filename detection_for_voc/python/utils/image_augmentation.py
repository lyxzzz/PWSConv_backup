import cv2
import numpy as np
import os
import copy
import random
import time

# # dont use, not helpful for the grayscale photoes
# def augmentation_histogram_equalization(src_image):
#     src_shape = src_image.shape

#     if len(src_shape) == 2:
#         dst_image = cv2.equalizeHist(src_image)
#     else:
#         dst_image = np.zeros(src_shape,np.uint8)
#         for i in range(src_shape[2]):
#             dst_image[:,:,i] = cv2.equalizeHist(src_image[:,:,i])

#     return dst_image
def augmentation_image_cls(img_input):
    raw_img = img_input
    tempboxes = np.zeros((0,4))
    if random.random() < 0.2:
        raw_img, raw_bboxes = augmentation_flip(raw_img, tempboxes)
    if random.random() < 0.5:
        raw_img, raw_bboxes = augmentation_rotate(raw_img, tempboxes, angle=90)
    if random.random() < 0.5:
        raw_img, raw_bboxes = augmentation_rotate(raw_img, tempboxes)
    if random.random() < 0.1:
        raw_img = augmentation_blur(raw_img)
    if random.random() < 0.5:
        raw_img = augmentation_brightness(raw_img)
    return raw_img

def augmentation_image_tltle(img_input, bbox_input):
    raw_img = img_input
    raw_bboxes = bbox_input
    if random.random() < 0.2:
        raw_img, raw_bboxes = augmentation_flip(raw_img, raw_bboxes)
    # if random.random() < 0.05:
    #     raw_img = augmentation_blur(raw_img)
    # if random.random() < 0.3:
    #     raw_img = augmentation_brightness(raw_img)
    return raw_img, raw_bboxes

def augmentation_padding(src_image, boxes):
    if boxes.shape[0] == 1:
        box = boxes[0]
        padding_value = random.randint(0, 255)
        new_image = src_image.copy()
        img_size = new_image.shape[0:2]
        new_image[0:int(box[1] * img_size[0]),:,:] = padding_value
        new_image[:,0:int(box[0] * img_size[1]),:] = padding_value
        new_image[int(box[3] * img_size[0]):,:,:] = padding_value
        new_image[:,int(box[2] * img_size[1]):,:] = padding_value
        return new_image
    else:
        return src_image

# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def augmentation_rotate(src_image, src_bboxes, angle=5):

    angle = random.randint(-1,1) * angle
    if angle == 0:
        return src_image, src_bboxes
    (h, w) = src_image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    pos = np.zeros((src_bboxes.shape[0], 4, 2), dtype=src_bboxes.dtype)
    newboxes = np.copy(src_bboxes)
    if newboxes.shape[0] > 0:
        pos[:, 0, 0] = src_bboxes[:, 0]
        pos[:, 0, 1] = src_bboxes[:, 1]
        pos[:, 1, 0] = src_bboxes[:, 2]
        pos[:, 1, 1] = src_bboxes[:, 1]
        pos[:, 2, 0] = src_bboxes[:, 2]
        pos[:, 2, 1] = src_bboxes[:, 3]
        pos[:, 3, 0] = src_bboxes[:, 0]
        pos[:, 3, 1] = src_bboxes[:, 3]
        pos -= 0.5
        pos[:, :, 0] = pos[:, :, 0] * w
        pos[:, :, 1] = pos[:, :, 1] * h
        new_pos = np.matmul(pos, M[:,0:2].T)
        newboxes[:, 0] = np.min(new_pos[:,:,0])/w
        newboxes[:, 1] = np.min(new_pos[:,:,1])/h
        newboxes[:, 2] = np.max(new_pos[:,:,0])/w
        newboxes[:, 3] = np.max(new_pos[:,:,1])/h
        newboxes += 0.5
    cos = M[0, 0]
    sin = M[0, 1]
    abs_cos = np.abs(cos)
    abs_sin = np.abs(sin)

    # nW = int((h * abs_sin) + (w * abs_cos))
    # nH = int((h * abs_cos) + (w * abs_sin))
    # # adjust the rotation matrix to take into account translation
    # M[0, 2] += (nW / 2) - cX
    # M[1, 2] += (nH / 2) - cY
    src_image = cv2.warpAffine(src_image, M, (w, h),borderValue=(255,255,255))
    return src_image, newboxes
    

def augmentation_flip(src_image, src_bboxes):
    src_image = cv2.flip(src_image, 1)
    newboxes = np.copy(src_bboxes)
    if newboxes.shape[0] > 0:
        newboxes[:,0::2] = 1 - newboxes[:,0::2]
        newboxes[:,[0,2]] = newboxes[:,[2,0]]
    return src_image, newboxes

def augmentation_blur(src_image):
    # if random.random() < 0.3:
    #     return cv2.blur(src_image, (3, 3))
    # else:
    return cv2.blur(src_image, (3, 1))

def augmentation_sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    return cv2.filter2D(image, -1, kernel=kernel)


def augmentation_brightness(image):
    blank = np.zeros(image.shape, image.dtype)
    brightness = 0.7
    return cv2.addWeighted(image, brightness, blank, 1-brightness, 0.0)

def augmentation_saltAndpepper(src_image, percetage):  
    SP_NoiseImg= copy.copy(src_image) 
    SP_NoiseNum=int(percetage*src_image.shape[0]*src_image.shape[1]) // 2
    max_x = src_image.shape[0]-1
    max_y = src_image.shape[1]-1

    for i in range(SP_NoiseNum): 
        randX=np.random.random_integers(0,max_x) 
        randY=np.random.random_integers(0,max_y) 
        SP_NoiseImg[randX,randY]=255 

    for i in range(SP_NoiseNum):
        randX=np.random.random_integers(0,max_x) 
        randY=np.random.random_integers(0,max_y) 
        SP_NoiseImg[randX,randY]=0

    return SP_NoiseImg

# dont use. cost too much time
def addGaussianNoise(src_image, mean, sigma): 
    G_Noiseimg = copy.copy(src_image)
    X = G_Noiseimg.shape[0]
    Y = G_Noiseimg.shape[1]
    Z = G_Noiseimg.shape[2]
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                G_Noiseimg[i][j][k] += random.gauss(mean, sigma)
                if G_Noiseimg[i][j][k] < 0:
                    G_Noiseimg[i][j][k] = 0
                elif G_Noiseimg[i][j][k] >255:
                    G_Noiseimg[i][j][k] = 255
    return G_Noiseimg



if __name__ == '__main__':
    image_src = 'C://Files/Data/读研资料/code/pdf/data/classification'
    #image_src = 'F://读研资料/数据集/RCTW2017/icdar2017rctw_train_v1.2/part1'

    data_path_list = []

    image_exts = ['jpg', 'png', 'jpeg', 'JPG']


    dataset = os.listdir(image_src)
    for image_name in dataset:
            for ext in image_exts:
                if image_name.endswith(ext):
                    filePath = os.path.join(image_src, image_name)
                    image=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),1)
                    
                    img_list = augmentation_image(image)

                    print(len(img_list))
                    for single_image in img_list:
                        cv2.imshow('image', single_image)
                        cv2.waitKey(0)
                    
                    # for i in range(1, 15, 3):
                    #     cv2.imshow('image', augmentation_blur(image, i))
                    #     cv2.waitKey(0)

                    cv2.destroyAllWindows() 
                    break


