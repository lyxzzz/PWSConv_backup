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
    
    negative_ratio = cfg_utils.get_cfg_attr(TRAIN_CFG, 'negative_ratio', 3.0)
    alpha = cfg_utils.get_cfg_attr(TRAIN_CFG, 'alpha', 1.0)
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
        tf.add_to_collection("posnum", n_positives)

        no_classes = tf.cast(pmask, tf.int32)

        nmask = gscores < 0
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask, predictions[:, background_label], 1.0 - fnmask)
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        # avoid n_neg = 0 so that val[-1] may error
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
        n_neg = tf.maximum(n_neg, 16)
        n_neg = n_neg + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)
        
        val, idxes = tf.nn.top_k(-nvalues, k=n_neg)
        max_hard_pred = -val[-1] 
        # max_hard_pred = tf.cond(tf.equal(n_neg, 0), abortfunc(nvalues, n_neg), topkfunc(nvalues, n_neg))
        # Final negative mask.
        # max_hard_pred = tf.concat(max_hard_pred, axis=0)

        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)
        n_negatives = tf.reduce_sum(fnmask)

        # Add cross-entropy loss.
        n_positives = tf.where(tf.less(n_positives, 0.5), 1.0 - n_positives, n_positives)
        # logits = logits * tf.pow((1 - predictions),
        # 
        obj_type_nums = logits.shape[-1]

        # with tf.name_scope('cross_entropy_pos'):
        #     pos_one_hot_labels = tf.one_hot(glabels, obj_type_nums)
        #     pos_loss = tf.losses.softmax_cross_entropy(pos_one_hot_labels, logits, 
        #                         label_smoothing=pos_label_smoothing, reduction='none')

        #     pos_loss = tf.reduce_sum(pos_loss * fpmask)
        #     pos_loss = tf.divide(pos_loss, n_positives)

        # with tf.name_scope('cross_entropy_neg'):
        #     neg_one_hot_labels = tf.one_hot(no_classes, obj_type_nums)
        #     neg_loss = tf.losses.softmax_cross_entropy(neg_one_hot_labels, logits, 
        #                         label_smoothing=neg_label_smoothing, reduction='none')
        #     neg_loss = tf.reduce_sum(neg_loss * fnmask)
        #     neg_loss = tf.divide(neg_loss, n_positives)

        ipmask = tf.cast(pmask, tf.int32)
        glabels = tf.where(pmask, glabels, ipmask)

        with tf.name_scope('cross_entropy'):
            one_hot_labels = tf.one_hot(glabels, obj_type_nums)
            cls_loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits, 
                                label_smoothing=pos_label_smoothing, reduction='none')
            
            pos_loss = tf.reduce_sum(cls_loss * fpmask)
            pos_loss = tf.divide(pos_loss, n_positives)
            
            neg_loss = tf.reduce_sum(cls_loss * fnmask)
            neg_loss = tf.divide(neg_loss, n_positives)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            # loc_loss = custom_layers.abs_smooth(localisations - glocalisations)
            loc_loss = tf_layer.l1_smooth(localisations - glocalisations)
            loc_loss = tf.reduce_sum(loc_loss * weights, axis=[0,1])
            # loc_loss = _safe_divide(loc_loss, n_positives, "divide")
            loc_loss = tf.divide(loc_loss, n_positives)

        # model_loss = (negative_ratio * pos_loss + neg_loss) / (negative_ratio + 1) + loc_loss
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