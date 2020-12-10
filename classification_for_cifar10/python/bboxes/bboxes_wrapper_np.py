import numpy as np

def _bboxes_encode_layer(labels, bboxes, anchors_layer,
                            prior_scaling,
                            dtype=np.float32):

    # Anchors coordinates and volume.
    yref, xref, href, wref, xmin, ymin, xmax, ymax, anchors_area, anchor_shape = anchors_layer

    # Initialize...
    shape = (yref.shape[0], href.size)

    feat_labels = np.zeros(shape, dtype=np.int64)
    feat_scores = np.zeros(shape, dtype=dtype)

    feat_xmin = np.zeros(shape, dtype=dtype)
    feat_ymin = np.zeros(shape, dtype=dtype)
    feat_xmax = np.ones(shape, dtype=dtype)
    feat_ymax = np.ones(shape, dtype=dtype)

    for i in range(len(labels)):
        label = labels[i]
        bbox = bboxes[i]

        int_xmin = np.maximum(xmin, bbox[0])
        int_ymin = np.maximum(ymin, bbox[1])
        int_xmax = np.minimum(xmax, bbox[2])
        int_ymax = np.minimum(ymax, bbox[3])
        
        h = np.maximum(int_ymax - int_ymin, 0.)
        w = np.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = anchors_area - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = inter_vol / union_vol

        mask = np.greater(jaccard, feat_scores)

        imask = mask.astype(np.int64)
        fmask = mask.astype(dtype)
        # Update values using mask.
        # feat_labels = np.where(mask, label, feat_labels)
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = np.where(mask, jaccard, feat_scores)

        feat_xmin = fmask * bbox[0] + (1 - fmask) * feat_xmin
        feat_ymin = fmask * bbox[1] + (1 - fmask) * feat_ymin
        feat_xmax = fmask * bbox[2] + (1 - fmask) * feat_xmax
        feat_ymax = fmask * bbox[3] + (1 - fmask) * feat_ymax

    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    
    # Encode features.
    feat_cx = (feat_cx - xref) / wref / prior_scaling[0]
    feat_cy = (feat_cy - yref) / href / prior_scaling[1]
    feat_w = np.log(feat_w / wref) / prior_scaling[2]
    feat_h = np.log(feat_h / href) / prior_scaling[3]
    
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = np.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores


def bboxes_encode(labels, bboxes, anchors,
                    positive_threshold,
                    negative_threshold,
                    prior_scaling,
                    hard_negative_threshold=0.35,
                    dtype=np.float32):
    target_labels = []
    target_localizations = []
    target_scores = []
    for i, anchors_layer in enumerate(anchors):
        t_labels, t_loc, t_scores = \
            _bboxes_encode_layer(labels, bboxes, anchors_layer,
                                        prior_scaling, 
                                        dtype)
        target_labels.append(t_labels.reshape([-1]))
        target_localizations.append(t_loc.reshape([-1, 4]))
        target_scores.append(t_scores.reshape([-1]))
    target_labels = np.concatenate(target_labels, axis=0)
    target_localizations = np.concatenate(target_localizations, axis=0)
    target_scores = np.concatenate(target_scores, axis=0)

    nmask = target_scores < negative_threshold
    max_value = np.max(target_scores)
    match_threshold = np.minimum(positive_threshold, max_value)
    # if match_threshold < hard_negative_threshold:
    #     match_threshold = hard_negative_threshold
    pmask = target_scores >= match_threshold
    nmask = np.logical_and(np.logical_not(pmask), nmask)
    # pmask = pmask - nmask
    ipmask = np.array(pmask, dtype=np.int8)
    inmask = np.array(nmask, dtype=np.int8)
    ipmask = ipmask - inmask
    return target_labels, target_localizations, ipmask

def _bboxes_select_layer(predictions_layer, localizations_layer, select_threshold=0.0):

    classes = np.argmax(predictions_layer, axis=1)
    scores = np.max(predictions_layer, axis=1)
    scores = np.where(classes > 0, scores, 0.0)

    idxes = np.where(scores > select_threshold)[0]

    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = localizations_layer[idxes]
    # classes = classes * np.cast(mask, classes.dtype)
    # scores = scores * np.cast(mask, scores.dtype)
    # bboxes = localizations_layer * np.expand_dims(mask, axis=-1)
    return classes, scores, bboxes

def _bboxes_clip(bboxes, scope=None):

    max_value = 1.0

    bboxes = np.transpose(bboxes)
    # Intersection bboxes and reference bbox.
    xmin = np.maximum(np.minimum(bboxes[0], max_value), 0)
    ymin = np.maximum(np.minimum(bboxes[1], max_value), 0)
    xmax = np.maximum(np.minimum(bboxes[2], max_value), 0)
    ymax = np.maximum(np.minimum(bboxes[3], max_value), 0)
    # Double check! Empty boxes when no-intersection.
    xmin = np.minimum(xmin, xmax)
    bboxes = np.transpose(np.stack([xmin, ymin, xmax, ymax], axis=0))
    return bboxes

def _bboxes_sort(classes, scores, bboxes, top_k=400, scope=None):
    idxes = np.argsort(scores)[::-1]
    nums = top_k
    if scores.shape[0] > top_k:
        idxes = idxes[:top_k]

    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    return classes, scores, bboxes

def _bboxes_new_nms(bboxes, scores, classes, thresh, keep_k):
    nboxes = bboxes.shape[0]
    item_array = np.zeros((nboxes), dtype=np.int8)
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    results_idx = []
    for i in range(nboxes):
        iarea = areas[i]
        if item_array[i] >= 1 or iarea == 0.0:
            continue
        results_idx.append(i)
        if len(results_idx) >= keep_k:
            break
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]

        # xx1 = np.maximum(ix1, x1[i+1:])
        # yy1 = np.maximum(iy1, y1[i+1:])
        # xx2 = np.maximum(ix2, x2[i+1:])
        # yy2 = np.maximum(iy2, y2[i+1:])
        # w = np.maximum(0.0, xx2 - xx1)
        # h = np.maximum(0.0, yy2 - yy1)
        # inter = w * h
        # ovr = inter / (areas[i+1:] + iarea + )
        for j in range(i + 1, nboxes):
            if areas[j] == 0.0:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                item_array[j] = 1
    if len(results_idx) < keep_k:
        return results_idx
    else:
        return results_idx[0:keep_k]
    

def _bboxes_nms(classes, scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
        # Apply NMS algorithm.
    # idxes = mybboxes.my_nms(bboxes, scores, nms_threshold, keep_top_k)
    # idxes = _bboxes_new_nms_3(bboxes, scores, classes, nms_threshold, keep_top_k)
    idxes = _bboxes_new_nms(bboxes, scores, classes, nms_threshold, keep_top_k)

    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    # Pad results.
    # scores = npe_tensors.pad_axis(scores, 0, keep_top_k, axis=0)
    # bboxes = npe_tensors.pad_axis(bboxes, 0, keep_top_k, axis=0)
    return classes, scores, bboxes

def bboxes_select(predictions_net, localizations_net,
                    select_threshold=0.0, nms_threshold=0.5,
                    top_k=2000, keep_top_k=400):
    l_classes = []
    l_scores = []
    l_bboxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = \
            _bboxes_select_layer(predictions_net[i],
                            localizations_net[i],
                            select_threshold)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)

    classes = np.concatenate(l_classes, axis=0)
    scores = np.concatenate(l_scores, axis=0)
    bboxes = np.concatenate(l_bboxes, axis=0)

    bboxes = _bboxes_clip(bboxes)

    classes, scores, bboxes = _bboxes_sort(classes, scores, bboxes, top_k=top_k)
    # print("=================================")
    # print(classes)
    # print(scores)
    # print("=================================")
    # Apply NMS algorithm.
    classes, scores, bboxes = \
        _bboxes_nms(classes, scores, bboxes,
                                nms_threshold=nms_threshold,
                                keep_top_k=keep_top_k)

    return classes, scores, bboxes

def bboxes_select_debug(predictions_net, localizations_net, file,
                    select_threshold=0.0, nms_threshold=0.5,
                    top_k=400, keep_top_k=200):
    l_classes = []
    l_scores = []
    l_bboxes = []
    l_layers = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = \
            _bboxes_select_layer(predictions_net[i],
                            localizations_net[i],
                            select_threshold)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)
        l_layers.append([i] * scores.shape[0])

    classes = np.concatenate(l_classes, axis=0)
    scores = np.concatenate(l_scores, axis=0)
    bboxes = np.concatenate(l_bboxes, axis=0)
    layers = np.concatenate(l_layers, axis=0)

    bboxes = _bboxes_clip(bboxes)

    idxes = np.argsort(scores)[::-1]
    nums = top_k
    if scores.shape[0] > top_k:
        idxes = idxes[:top_k]

    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    layers = layers[idxes]
    
    file.write("=================================\n")
    for i in classes:
        file.write("{} ".format(i))
    file.write('\n')
    for i in scores:
        file.write("{} ".format(i))
    file.write('\n')
    file.write("=================================\n")
    file.flush()

    idxes = _bboxes_new_nms(bboxes, scores, classes, nms_threshold, keep_top_k)

    classes = classes[idxes]
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    layers = layers[idxes]
    # Pad results.
    # scores = npe_tensors.pad_axis(scores, 0, keep_top_k, axis=0)
    # bboxes = npe_tensors.pad_axis(bboxes, 0, keep_top_k, axis=0)
    return classes, scores, bboxes, layers
