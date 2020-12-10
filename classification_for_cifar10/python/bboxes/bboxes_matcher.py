import numpy as np

def match_bboxes(pred_scores, pred_boxes, pred_labels, gt_boxes, gt_labels, matching_threshold):
    rm_threshold = 1e-4
    idxes = np.where(pred_scores >= rm_threshold)[0]
    scores = pred_scores[idxes]
    boxes = pred_boxes[idxes]
    labels = pred_labels[idxes]

    pred_boxes_num = boxes.shape[0]
    real_boxes_num = gt_boxes.shape[0]
    if pred_boxes_num == 0:
        return [0, 0, real_boxes_num]

    xmin = gt_boxes[:,0]
    ymin = gt_boxes[:,1]
    xmax = gt_boxes[:,2]
    ymax = gt_boxes[:,3]
    area = (xmax - xmin) * (ymax - ymin)

    true_positive = 0
    true_boxes = np.zeros((real_boxes_num), dtype=np.bool)
    for i in range(pred_boxes_num):
        score = scores[i]
        box = boxes[i]
        label = labels[i]
        
        int_xmin = np.maximum(xmin, box[0])
        int_ymin = np.maximum(ymin, box[1])
        int_xmax = np.minimum(xmax, box[2])
        int_ymax = np.minimum(ymax, box[3])
        
        h = np.maximum(int_ymax - int_ymin, 0.)
        w = np.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = area - inter_vol \
            + (box[2] - box[0]) * (box[3] - box[1])
        jaccard = inter_vol / union_vol

        jaccard = np.where(label == gt_labels, jaccard, 0.0)

        idxmax = np.argmax(jaccard, axis=0)
        jcdmax = jaccard[idxmax]

        match = jcdmax > matching_threshold
        existing_match = true_boxes[idxmax]
        # TP: match & no previous match and FP: previous match | no match.
        # If difficult: no record, i.e FP=False and TP=False.
        tp = np.logical_and(match, np.logical_not(existing_match))
        if tp:
            true_positive += 1
            true_boxes[idxmax] = True
        
    return [true_positive, pred_boxes_num, real_boxes_num]

def match_bboxes_single_label(pred_label, pred_scores, pred_boxes, gt_boxes, gt_labels, gt_difficults, matching_threshold):
    rm_threshold = 1e-5
    idxes = np.where(pred_scores >= rm_threshold)[0]
    scores = pred_scores[idxes]
    boxes = pred_boxes[idxes]

    pred_boxes_num = boxes.shape[0]
    real_boxes_num = gt_boxes.shape[0]

    n_gbboxes = np.count_nonzero(np.logical_and(np.equal(gt_labels, pred_label),
                                                    np.logical_not(gt_difficults)))

    xmin = gt_boxes[:,0]
    ymin = gt_boxes[:,1]
    xmax = gt_boxes[:,2]
    ymax = gt_boxes[:,3]
    area = (xmax - xmin) * (ymax - ymin)

    gmatch = np.zeros((real_boxes_num), dtype=np.bool)
    ta_tp = np.zeros((pred_boxes_num), dtype=np.bool)
    ta_fp = np.zeros((pred_boxes_num), dtype=np.bool)
    for i in range(pred_boxes_num):
        pred_box = boxes[i]

        int_xmin = np.maximum(xmin, pred_box[0])
        int_ymin = np.maximum(ymin, pred_box[1])
        int_xmax = np.minimum(xmax, pred_box[2])
        int_ymax = np.minimum(ymax, pred_box[3])
        
        h = np.maximum(int_ymax - int_ymin, 0.)
        w = np.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = area - inter_vol \
            + (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        jaccard = inter_vol / union_vol

        jaccard = np.where(pred_label == gt_labels, jaccard, 0.0)

        idxmax = np.argmax(jaccard, axis=0)
        jcdmax = jaccard[idxmax]

        match = jcdmax > matching_threshold
        existing_match = gmatch[idxmax]
        not_difficult = np.logical_not(gt_difficults[idxmax])
        # TP: match & no previous match and FP: previous match | no match.
        # If difficult: no record, i.e FP=False and TP=False.
        tp = np.logical_and(not_difficult,
                            np.logical_and(match, np.logical_not(existing_match)))
        ta_tp[i] = tp
        fp = np.logical_and(not_difficult,
                            np.logical_or(existing_match, np.logical_not(match)))

        ta_fp[i] = fp
        # Update grountruth match.
        if not_difficult and match:
            gmatch[idxmax] = True
    return [scores, n_gbboxes, ta_tp, ta_fp]