import os
import cv2
import numpy as np
def touch_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_img_and_boxes(img, 
                boxes, scores, classes, 
                raw_path, 
                root_path, dir_path=None, 
                draw_rect=False, label_name=None):
    if dir_path == None:
        dst_path = root_path
    else:
        dst_path = os.path.join(root_path, dir_path)

    touch_dir(dst_path)
    
    img_name = os.path.basename(raw_path)

    if draw_rect:
        for box in boxes:
            cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,255,0), 2)
            if label_name == None:
                label_name = 'Unknown'
            cv2.putText(img, "{}".format(label_name), (int(box[0]),int(box[3]+30)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 4, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(dst_path, img_name), img)

    with open(os.path.join(dst_path, os.path.splitext(img_name)[0]) + ".txt",
                "w") as f:
        for i, box in enumerate(boxes):
            line = ",".join(str(box[k]) for k in range(4))
            line += "," + str(scores[i]) + ", " + str(classes[i]) + "\r\n"
            f.writelines(line)

def save_img_and_boxes_2(boxes, scores, classes, 
                raw_path, 
                root_path, dir_path=None, 
                draw_rect=False, label_name=None):
    if dir_path == None:
        dst_path = root_path
    else:
        dst_path = os.path.join(root_path, dir_path)

    touch_dir(dst_path)
    
    img = cv2.imdecode(np.fromfile(raw_path,dtype=np.uint8),1)
    img_name = os.path.basename(raw_path)

    h, w = img.shape[0:2]
    if draw_rect:
        for box in boxes:
            cv2.rectangle(img, (int(box[0] * w),int(box[1] * h)), (int(box[2] * w),int(box[3] * h)), (0,255,0), 2)
            if label_name == None:
                label_name = 'Unknown'
        cv2.putText(img, "{}".format(label_name), (int(10),int(80)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 4, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(dst_path, img_name), img)

    with open(os.path.join(dst_path, os.path.splitext(img_name)[0]) + ".txt",
                "w") as f:
        for i, box in enumerate(boxes):
            line = ",".join(str(box[k]) for k in range(4))
            line += "," + str(scores[i]) + ", " + str(classes[i]) + "\r\n"
            f.writelines(line)