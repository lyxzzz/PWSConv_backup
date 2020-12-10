import tensorflow as tf
from bboxes import bboxes_wrapper_tf

import tensorflow as tf
from utils import tf_utils

def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=21,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tf_utils.tensor_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tf_utils.tensor_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def tf_ssd_bboxes_select(predictions_net, localizations_net,
                         select_threshold=None,
                         num_classes=21,
                         ignore_class=0,
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        select_threshold,
                                                        num_classes,
                                                        ignore_class)
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

def bboxes_sort_all_classes(classes, scores, bboxes, top_k=400, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    Assume the input Tensors mix-up objects with different classes.
    Assume a batch-type input.

    Args:
      classes: Batch x N Tensor containing integer classes.
      scores: Batch x N Tensor containing float scores.
      bboxes: Batch x N x 4 Tensor containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      classes, scores, bboxes: Sorted tensors of shape Batch x Top_k.
    """
    with tf.name_scope(scope, 'bboxes_sort', [classes, scores, bboxes]):
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the batch.
        def fn_gather(classes, bboxes, idxes):
            cl = tf.gather(classes, idxes)
            bb = tf.gather(bboxes, idxes)
            return [cl, bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1], x[2]),
                      [classes, bboxes, idxes],
                      dtype=[classes.dtype, bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        classes = r[0]
        bboxes = r[1]
        return classes, scores, bboxes


def bboxes_sort(scores, bboxes, top_k=400, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    If inputs are dictionnaries, assume every key is a different class.
    Assume a batch-type input.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      scores, bboxes: Sorted Tensors/Dictionaries of shape Batch x Top_k x 1|4.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the first dim.
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


def bboxes_clip(bbox_ref, bboxes, scope=None):
    """Clip bounding boxes to a reference box.
    Batch-compatible if the first dimension of `bbox_ref` and `bboxes`
    can be broadcasted.

    Args:
      bbox_ref: Reference bounding box. Nx4 or 4 shaped-Tensor;
      bboxes: Bounding boxes to clip. Nx4 or 4 shaped-Tensor or dictionary.
    Return:
      Clipped bboxes.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_clip_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_clip(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
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


def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(name, 'bboxes_resize_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_resize(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes

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

def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Should only be used on single-entries. Use batch version otherwise.

    Args:
      scores: N Tensor containing float scores.
      bboxes: N x 4 Tensor containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      classes, scores, bboxes Tensors, sorted by score.
        Padded with zero if necessary.
    """
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = __pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = __pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      scores, bboxes Tensors/Dictionaries, sorted by score.
        Padded with zero if necessary.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes

def bboxes_detected(predictions, localisations, num_classes, background_label,
                        select_threshold=None, nms_threshold=0.5,
                        top_k=400, keep_top_k=200):
    """Get the detected bounding boxes from the SSD network output.
    """
    # Select top_k bboxes from predictions, and clip
    rscores, rbboxes = \
        tf_ssd_bboxes_select(predictions, localisations,
                                        select_threshold=select_threshold,
                                        num_classes=num_classes,
                                        ignore_class=background_label)
    rscores, rbboxes = \
        bboxes_sort(rscores, rbboxes, top_k=top_k)
    # Apply NMS algorithm.
    rscores, rbboxes = \
        bboxes_nms_batch(rscores, rbboxes,
                                nms_threshold=nms_threshold,
                                keep_top_k=keep_top_k)
    t_scores = []
    t_bboxes = []
    t_labels = []
    for c in rscores.keys():
        s_scores = rscores[c]
        s_bboxes = rbboxes[c]

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