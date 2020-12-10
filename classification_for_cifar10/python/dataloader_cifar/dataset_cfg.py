import pickle

label_name_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

OBJ_TYPE_NUMS = len(label_name_list)
NAME_TO_ID = {}
ID_TO_NAME = {}

for i, labelname in enumerate(label_name_list):
    NAME_TO_ID[labelname] = i
    ID_TO_NAME[i] = labelname