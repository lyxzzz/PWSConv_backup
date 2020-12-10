import tensorflow as tf
from tensorflow.contrib import slim
from utils import custom_layers
from utils import tf_layer
from utils import cfg_utils

def losses(predictions, localisations, logits,
               glabels, glocalisations, gscores,
               background_label,
               TRAIN_CFG,
               scope=None):
    
    loc_rate = cfg_utils.get_cfg_attr(TRAIN_CFG, 'loc_rate', 1.0)
    cls_alpha = cfg_utils.get_cfg_attr(TRAIN_CFG, 'cls_alpha', 0.25)
    cls_exp = cfg_utils.get_cfg_attr(TRAIN_CFG, 'cls_exp', 2)

    pos_label_smoothing = cfg_utils.get_cfg_attr(TRAIN_CFG, 'pos_label_smoothing', 0.0)
    neg_label_smoothing = cfg_utils.get_cfg_attr(TRAIN_CFG, 'neg_label_smoothing', 0.0)

    with tf.name_scope(scope, 'loss'):
        batch_size = tf.shape(logits[0])[0]
        f_batch_size = tf.cast(batch_size, dtype=tf.float32)

        logits = tf.concat(logits, axis=1)
        predictions = tf.concat(predictions, axis=1)
        localisations = tf.concat(localisations, axis=1)
        
        logits = tf.reshape(logits, [-1, logits.shape[-1]])
        predictions = tf.reshape(predictions, [-1, predictions.shape[-1]])
        localisations = tf.reshape(localisations, [-1, localisations.shape[-1]])

        glabels = tf.reshape(glabels, [-1])
        glocalisations = tf.reshape(glocalisations, [-1, glocalisations.shape[-1]])
        gscores = tf.reshape(gscores, [-1])
        dtype = logits.dtype

        pmask = gscores > 0
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        no_classes = tf.cast(pmask, tf.int32)

        nmask = gscores < 0
        fnmask = tf.cast(nmask, dtype)
        n_negatives = tf.reduce_sum(fnmask)

        n_positives = tf.where(tf.equal(n_positives, 0.0), 1.0 - n_positives, n_positives)

        obj_type_nums = logits.shape[-1]

        with tf.name_scope('cross_entropy_pos'):
            pos_one_hot_labels = tf.one_hot(glabels, obj_type_nums)

            pos_label_prediction = tf.reduce_sum(predictions * pos_one_hot_labels, axis=-1)
            pos_cls_weights = cls_alpha * tf.pow((1 - pos_label_prediction), cls_exp)

            pos_loss = tf.losses.softmax_cross_entropy(pos_one_hot_labels, logits, 
                                weights=pos_cls_weights,
                                label_smoothing=pos_label_smoothing, reduction='none')

            pos_loss = tf.reduce_sum(pos_loss * fpmask)
            pos_loss = tf.divide(pos_loss, n_positives)

        with tf.name_scope('cross_entropy_neg'):
            neg_one_hot_labels = tf.one_hot(no_classes, obj_type_nums)

            neg_label_prediction = tf.reduce_sum(predictions * neg_one_hot_labels, axis=-1)
            neg_cls_weights = (1 - cls_alpha) * tf.pow((1 - label_prediction), cls_exp)

            neg_loss = tf.losses.softmax_cross_entropy(neg_one_hot_labels, logits, 
                                weights=neg_cls_weights,
                                label_smoothing=neg_label_smoothing, reduction='none')
            neg_loss = tf.reduce_sum(neg_loss * fnmask)
            neg_loss = tf.divide(neg_loss, n_positives)

        with tf.name_scope('localization'):
            weights = tf.expand_dims(loc_rate * fpmask, axis=-1)
            # loc_loss = custom_layers.abs_smooth(localisations - glocalisations)
            loc_loss = tf_layer.l1_smooth(localisations - glocalisations)
            loc_loss = tf.reduce_sum(loc_loss * weights, axis=[0,1])
            loc_loss = tf.divide(loc_loss, n_positives)

        model_loss = pos_loss + neg_loss + loc_loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        total_loss = tf.add_n(regularization_losses) + model_loss
        
        tf.summary.scalar('model_los', model_loss)
        tf.summary.scalar('total_los', total_loss)
        tf.summary.scalar('pos_los', pos_loss)
        tf.summary.scalar('neg_los', neg_loss)
        tf.summary.scalar('loc_loss', loc_loss)
        return pos_loss, neg_loss, loc_loss, model_loss, total_loss

def losses_description():
    total_loss = 4
    print_loss = ['pos_loss', 'neg_loss', 'loc_loss', 'total_loss']
    print_loss_index = [0, 4]
    return total_loss, print_loss, print_loss_index