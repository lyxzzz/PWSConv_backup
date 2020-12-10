import tensorflow as tf
from tensorflow.contrib import slim
from utils import tf_layer
from utils import cfg_utils

def losses(predictions, localisations, logits,
               glables, glocalisations, gscores,
               background_label,
               TRAIN_CFG,
               scope=None):
    
    negative_ratio = cfg_utils.get_cfg_attr(TRAIN_CFG, 'negative_ratio', 3.0)
    alpha = cfg_utils.get_cfg_attr(TRAIN_CFG, 'alpha', 1.0)

    with tf.name_scope(scope, 'loss'):
        logits = tf.concat(logits, axis=1)
        predictions = tf.concat(predictions, axis=1)
        localisations = tf.concat(localisations, axis=1)
        
        dtype = logits.dtype

        pmask = gscores > 0
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask, axis=1)

        no_classes = tf.cast(pmask, tf.int32)

        nmask = gscores < 0
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask, predictions[:, :, background_label], 1.0 - fnmask)
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask, axis=1), tf.int32)
        max_neg_entries = tf.reduce_min(max_neg_entries)
        # avoid n_neg = 0 so that val[-1] may error
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + 1
        n_neg = tf.minimum(n_neg, max_neg_entries)
        n_neg = tf.reduce_mean(n_neg)
        
        val, idxes = tf.nn.top_k(-nvalues, k=n_neg)
        max_hard_pred = -val[:,-1] 
        # max_hard_pred = tf.cond(tf.equal(n_neg, 0), abortfunc(nvalues, n_neg), topkfunc(nvalues, n_neg))
        # Final negative mask.
        # max_hard_pred = tf.concat(max_hard_pred, axis=0)

        max_hard_pred = tf.reshape(max_hard_pred, [-1, 1])

        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        n_positives = tf.where(tf.equal(n_positives, 0.0), 1.0 - n_positives, n_positives)
        # logits = logits * tf.pow((1 - predictions),

        with tf.name_scope('cross_entropy_pos'):
            pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=glables)
            pos_loss = tf.reduce_sum(negative_ratio * pos_loss * fpmask, axis=1)
            # pos_loss = tf.reduce_sum(pos_loss * fpmask, axis=1)
            # pos_loss = tf.reduce_sum(pos_loss * fpmask)
            # pos_loss = _safe_divide(pos_loss, n_positives, "divide")
            pos_loss = tf.divide(pos_loss, n_positives)
            pos_loss = tf.reduce_mean(pos_loss)

        with tf.name_scope('cross_entropy_neg'):
            neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            neg_loss = tf.reduce_sum(neg_loss * fnmask, axis=1)
            # neg_loss = _safe_divide(neg_loss, n_positives, "divide")
            f_neg = tf.cast(n_neg, dtype=tf.float32)
            neg_loss = tf.divide(neg_loss, n_positives)
            neg_loss = tf.reduce_mean(neg_loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loc_loss = tf_layer.l1_smooth(localisations - glocalisations)
            loc_loss = tf.reduce_sum(loc_loss * weights, axis=[1,2])
            # loc_loss = _safe_divide(loc_loss, n_positives, "divide")
            loc_loss = tf.divide(loc_loss, n_positives)
            loc_loss = tf.reduce_mean(loc_loss)

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