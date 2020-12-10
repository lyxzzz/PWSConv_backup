import tensorflow as tf
from register import anchor_dict

def prepare_before_model_construct(model_type, ROOT_CFG):
    h, w = ROOT_CFG["augmentation"]["size"]

    input_image = tf.placeholder(tf.float32, shape=[None, h, w, 3], name='input_image')
    input_training = tf.placeholder(tf.bool, name='input_training')

    anchor_func = anchor_dict[ROOT_CFG["anchors"].get('type', 'default')]

    # anchor_list = []
    # for shape in ROOT_CFG.data_cfg.shape_list:
    #     anchor_list.append()
    anchor_nums = 0
    all_anchors = anchor_func.anchors((h, w), ROOT_CFG["anchors"])
    anchor_layer_size = len(all_anchors)
    for i in range(anchor_layer_size):
        anchor_nums = anchor_nums + (all_anchors[i][0].shape[0] * all_anchors[i][2].shape[0])
    print("anchor nums {}".format(anchor_nums))

    if model_type == 'test' or model_type == 'val':
        return [input_image, input_training, all_anchors]

    input_batch_cls = tf.placeholder(tf.int32, shape=[None, anchor_nums], name='input_batch_cls')
    input_batch_loc = tf.placeholder(tf.float32, shape=[None, anchor_nums, 4], name='input_batch_loc')
    input_batch_score = tf.placeholder(tf.int8, shape=[None, anchor_nums], name='input_batch_score')
    
    return [input_image, input_training, input_batch_cls, input_batch_loc, input_batch_score, all_anchors]