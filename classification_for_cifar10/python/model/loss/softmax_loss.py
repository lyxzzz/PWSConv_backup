import tensorflow as tf
from tensorflow.contrib import slim
from utils import custom_layers
from utils import tf_layer
from utils import cfg_utils

def losses(logits, glabels, TRAIN_CFG, scope=None):

    with tf.name_scope(scope, 'loss'):
        
        obj_type_nums = logits.shape[-1]

        with tf.name_scope('cross_entropy'):
            one_hot_labels = tf.one_hot(glabels, obj_type_nums)
            cls_loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits, reduction='none')

            print(cls_loss)
            cls_loss = tf.reduce_mean(cls_loss)
            print(cls_loss)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        total_loss = tf.add_n(regularization_losses) + cls_loss
        
        tf.summary.scalar('total_los', total_loss)
        tf.summary.scalar('cls_los', cls_loss)
        return cls_loss, total_loss

def losses_description():
    total_loss = 1
    print_loss = ['cls_loss', 'total_loss']
    print_loss_index = [0, 2]
    return total_loss, print_loss, print_loss_index