import tensorflow as tf

def prepare_before_model_construct(model_type, ROOT_CFG):
    h, w = ROOT_CFG.json_cfg["augmentation"]["size"]

    input_image = tf.placeholder(tf.float32, shape=[None, h, w, 3], name='input_image')
    input_training = tf.placeholder(tf.bool, name='input_training')

    if model_type == 'test' or model_type == 'val':
        return [input_image, input_training]

    input_label = tf.placeholder(tf.uint8, shape=[None], name='input_label')
    
    return [input_image, input_training, input_label]