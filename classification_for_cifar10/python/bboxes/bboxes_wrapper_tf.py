import tensorflow as tf
import numpy as np

def _bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    # Anchors coordinates and volume.
    yref, xref, href, wref, xmin, ymin, xmax, ymax, vol_anchors, anchor_shape = anchors_layer

    # Initialize tensors...
    shape = (yref.shape[0], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int32)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_xmin = tf.maximum(xmin, bbox[0])
        int_ymin = tf.maximum(ymin, bbox[1])
        int_xmax = tf.minimum(xmax, bbox[2])
        int_ymax = tf.minimum(ymax, bbox[3])
        
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_xmin = tf.maximum(xmin, bbox[0])
        int_ymin = tf.maximum(ymin, bbox[1])
        int_xmax = tf.minimum(xmax, bbox[2])
        int_ymax = tf.minimum(ymax, bbox[3])

        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, ignore_threshold))

        imask = tf.cast(mask, tf.int32)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_xmin = fmask * bbox[0] + (1 - fmask) * feat_xmin
        feat_ymin = fmask * bbox[1] + (1 - fmask) * feat_ymin
        feat_xmax = fmask * bbox[2] + (1 - fmask) * feat_xmax
        feat_ymax = fmask * bbox[3] + (1 - fmask) * feat_ymax

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cx = (feat_cx - xref) / wref / prior_scaling[0]
    feat_cy = (feat_cy - yref) / href / prior_scaling[1]
    feat_w = tf.log(feat_w / wref) / prior_scaling[2]
    feat_h = tf.log(feat_h / href) / prior_scaling[3]
    
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores


def bboxes_encode_single(labels,
                         bboxes,
                         anchors,
                         positive_threshold,
                         negative_threshold,
                         prior_scaling,
                         dtype=tf.float32,
                         scope='bboxes_encode'):
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    _bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               prior_scaling, 
                                               dtype)
                t_labels = tf.reshape(t_labels, [-1])
                t_scores = tf.reshape(t_scores, [-1])
                t_loc = tf.reshape(t_loc, [-1, 4])
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)

        target_labels = tf.concat(target_labels, axis=0)
        target_localizations = tf.concat(target_localizations, axis=0)
        target_scores = tf.concat(target_scores, axis=0)

        max_value = tf.reduce_max(target_scores)
        nmask = target_scores < negative_threshold
        match_threshold = tf.minimum(positive_threshold, max_value)
        pmask = target_scores >= match_threshold
        nmask = tf.logical_and(tf.logical_not(pmask), nmask)
        # pmask = pmask - nmask
        fpmask = tf.cast(pmask, dtype=tf.int8)
        fnmask = tf.cast(nmask, dtype=tf.int8)
        fpmask = fpmask - fnmask
        return target_labels, target_localizations, fpmask


def _bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    yref, xref, href, wref = anchors_layer[0:4]
    anchor_shape = anchors_layer[-1]

    feat_raw_shape = (-1, anchor_shape[0], anchor_shape[1], href.shape[0], 4)
    middle_nums = anchor_shape[0] * anchor_shape[1] * href.shape[0]
    xy_shape = (anchor_shape[0], anchor_shape[1], 1)
    # Compute center, height and width
    feat_localizations = tf.reshape(feat_localizations, feat_raw_shape)
    yref = np.reshape(yref, xy_shape)
    xref = np.reshape(xref, xy_shape)
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = tf.maximum(cy - h / 2., 0.0)
    xmin = tf.maximum(cx - w / 2., 0.0)
    ymax = tf.minimum(cy + h / 2., 1.0)
    xmax = tf.minimum(cx + w / 2., 1.0)
    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    bboxes = tf.reshape(bboxes, [-1, middle_nums, 4])
    return bboxes


def bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='bboxes_decode'):
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                _bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes
