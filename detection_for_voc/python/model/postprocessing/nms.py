import tensorflow as tf
from bboxes import bboxes_wrapper_tf

import tensorflow as tf
from utils import tf_utils

def _bboxes_select_layer(prob_layer, loc_layer, obj_type_nums, background_label, select_threshold, scope=None):
    with tf.name_scope(scope, 'bboxes_select_layer',
                       [prob_layer, loc_layer]):
        d_scores = {}
        d_bboxes = {}
        for c in range(0, obj_type_nums):
            if c != background_label:
                # Remove boxes under the threshold.
                scores = prob_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = loc_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes

def _bboxes_select(prob_net, loc_net, obj_type_nums,
                  background_label,
                  select_threshold,
                  scope=None):
    with tf.name_scope(scope, 'bboxes_select',
                       [prob_net, loc_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(prob_net)):
            scores, bboxes = _bboxes_select_layer(prob_net[i],
                                                        loc_net[i],
                                                        obj_type_nums,
                                                        background_label,
                                                        select_threshold)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes

def _bboxes_clip(bbox_ref, bboxes, scope=None):
    with tf.name_scope(scope, 'bboxes_clip'):
        # Easier with transposed bboxes. Especially for broadcasting.
        bbox_ref = tf.transpose(bbox_ref)
        bboxes = tf.transpose(bboxes)
        # Intersection bboxes and reference bbox.
        ymin = tf.maximum(bboxes[0], bbox_ref[0])
        xmin = tf.maximum(bboxes[1], bbox_ref[1])
        ymax = tf.minimum(bboxes[2], bbox_ref[2])
        xmax = tf.minimum(bboxes[3], bbox_ref[3])
        # Double check! Empty boxes when no-intersection.
        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))
        return bboxes

def _bboxes_sort(scores, bboxes, top_k=400, scope=None):
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [bboxes, idxes],
                      dtype=[bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]
        return scores, bboxes

def __get_shape(x, rank=None):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]
def __pad_axis(x, offset, size, axis=0, name=None):
    with tf.name_scope(name, 'pad_axis'):
        shape = __get_shape(x)
        rank = len(shape)
        # Padding description.
        new_size = tf.maximum(size-offset-shape[axis], 0)
        pad1 = tf.stack([0]*axis + [offset] + [0]*(rank-axis-1))
        pad2 = tf.stack([0]*axis + [new_size] + [0]*(rank-axis-1))
        paddings = tf.stack([pad1, pad2], axis=1)
        x = tf.pad(x, paddings, mode='CONSTANT')
        # Reshape, to get fully defined shape if possible.
        # TODO: fix with tf.slice
        shape[axis] = size
        x = tf.reshape(x, tf.stack(shape))
        return x

def __bboxes_nms_single(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                            keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        scores = __pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = __pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes
    


def _bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    with tf.name_scope(scope, 'bboxes_nms'):
        r = tf.map_fn(lambda x: __bboxes_nms_single(x[0], x[1], nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes

def bboxes_detected(prob_net, loc_net, obj_type_nums, 
                background_label,
                select_threshold=0.05,
                nms_threshold=0.5,
                top_k=400,
                keep_top_k=200,
                scope=None):
    rscores, rbboxes = _bboxes_select(prob_net, loc_net, obj_type_nums,
                                        background_label,
                                        select_threshold=select_threshold)
    d_scores = {}
    d_bboxes = {}
    t_scores = []
    t_bboxes = []
    t_labels = []
    for c in rscores.keys():
        s_scores = rscores[c]
        s_bboxes = rbboxes[c]

        # d_bboxes[c] = bboxes_clip(bbox_ref, bboxes[c])
        s_scores, s_bboxes = _bboxes_sort(s_scores, s_bboxes, top_k)
        s_scores, s_bboxes = _bboxes_nms(s_scores, s_bboxes, nms_threshold, keep_top_k)

        # d_scores[c] = s_scores
        # d_bboxes[c] = s_bboxes
        t_scores.append(s_scores)
        t_bboxes.append(s_bboxes)
        s_labels = s_scores >= 0
        s_labels = tf.cast(s_labels, dtype=tf.int8)
        s_labels = s_labels * c
        t_labels.append(s_labels)
    t_scores = tf.concat(t_scores, axis=1)
    t_bboxes = tf.concat(t_bboxes, axis=1)
    t_labels = tf.concat(t_labels, axis=1)
    return t_scores, t_bboxes, t_labels

class __NMS_CFG:
    select_threshold = 0.01
    nms_threshold = 0.5
    top_k = 400
    keep_top_k = 200

def postprocessing(model_outputs, num_classes, anchors, background_label, postprocessing_cfg, anchor_cfg):
    for k in postprocessing_cfg.keys():
        if k == 'type':
            continue
        if hasattr(__NMS_CFG, k):
            setattr(__NMS_CFG, k, postprocessing_cfg[k])
            print('{}:{}'.format(k, postprocessing_cfg[k]))
    predictions, localisations, logits = model_outputs[0:3]
    all_anchors = anchors
    rbboxes = bboxes_wrapper_tf.bboxes_decode(localisations, all_anchors, anchor_cfg["prior_scaling"])

    r = bboxes_detected(predictions, rbboxes, num_classes, 
                                        background_label,
                                        __NMS_CFG.select_threshold,
                                        __NMS_CFG.nms_threshold,
                                        __NMS_CFG.top_k,
                                        __NMS_CFG.keep_top_k)
    return r