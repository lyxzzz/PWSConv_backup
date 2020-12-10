import numpy as np
from bboxes import bboxes_matcher

# def simple_cls_acc(preparedata, modeloutputs):
def get_tp_and_fp(pred_scores, pred_boxes, pred_labels, gt_boxes, gt_labels, matching_threshold):
    return bboxes_matcher.match_bboxes(pred_scores, pred_boxes, pred_labels, gt_boxes, gt_labels, matching_threshold)

class EvalmAP:
    def __init__(self, obj_type_nums):
        self.scores_dict = {}
        self.numbox_dict = {}
        self.numdetect_dict = {}
        self.tp_dict = {}
        self.fp_dict = {}
        self.obj_type_nums = obj_type_nums
    def _create_or_append(self, c, result):
        if c not in self.scores_dict:
            self.scores_dict[c] = [result[0]]
            self.numbox_dict[c] = result[1]
            self.tp_dict[c] = [result[2]]
            self.fp_dict[c] = [result[3]]
            self.numdetect_dict[c] = len(result[0])
        else:
            self.scores_dict[c].append(result[0])
            self.numbox_dict[c] += result[1]
            self.tp_dict[c].append(result[2])
            self.fp_dict[c].append(result[3])
            self.numdetect_dict[c] += len(result[0])
    def addsample(self, batch_size, b_scores, b_bboxes, b_labels, gt_boxes, gt_labels, gt_diff=0, matching_threshold=0.45):
        for batch_index in range(batch_size):
            for c in range(1, self.obj_type_nums):
                idxes = np.where(b_labels[batch_index] == c)[0]
                detected_scores = b_scores[batch_index][idxes]
                detected_boxes = b_bboxes[batch_index][idxes]
                if gt_diff == 0:
                    gt_diff_batch = np.zeros(gt_labels[batch_index].shape, dtype=np.int8)
                    r = bboxes_matcher.match_bboxes_single_label(c, detected_scores, detected_boxes
                                    , gt_boxes[batch_index], gt_labels[batch_index], gt_diff_batch, matching_threshold)
                else:
                    r = bboxes_matcher.match_bboxes_single_label(c, detected_scores, detected_boxes
                                    , gt_boxes[batch_index], gt_labels[batch_index], gt_diff[batch_index], matching_threshold)
                # r = bboxes_matcher.match_bboxes_single_label(c, detected_scores[c][batch_index], detected_boxes[c][batch_index]
                #                                 , gt_boxes[batch_index], gt_labels[batch_index], gt_diff[batch_index], matching_threshold)
                self._create_or_append(c, r)
    
    def calmAP(self):
        pr_dict = {}
        ap_dict = {}
        info_dict = {}
        total_ap = 0.0
        scores_dict = {}
        tp_dict = {}
        fp_dict = {}
        for c in self.scores_dict.keys():
            scores_dict[c] = np.concatenate(self.scores_dict[c], axis=0)
            tp_dict[c] = np.concatenate(self.tp_dict[c], axis=0)
            fp_dict[c] = np.concatenate(self.fp_dict[c], axis=0)
            # print('{}:{}'.format(c,len(self.scores_dict[c])))
            idxes = np.argsort(scores_dict[c])[::-1]
            tp = tp_dict[c][idxes]
            fp = fp_dict[c][idxes]
            tp = np.cumsum(tp, axis=0)
            fp = np.cumsum(fp, axis=0)
            if self.numbox_dict[c] == 0:
                recall = 0.0
            else: 
                recall = tp / self.numbox_dict[c]
            total_predict = tp + fp
            precision = tp / np.maximum(total_predict, np.finfo(np.float64).eps)
            if tp.shape[0] == 0:
                true_positive_nums = 0
            else:
                true_positive_nums = tp[-1]
            total_pred_nums = true_positive_nums + fp[-1]
            info_dict[c] = [true_positive_nums, total_pred_nums, self.numbox_dict[c]]
            # precision = tp / total_predict
            cls_ap = 0.
            pr_dict[c] = []
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                pr_dict[c].append(p)
                cls_ap = cls_ap + p / 11.
            # precision = np.concatenate([precision, [0.]], axis=0)
            # recall = np.concatenate([recall, [np.inf]], axis=0)


            # cls_ap = 0.0
            # for t in np.arange(0., 1.1, 0.1):
            #     mask_idxes = np.where(recall >= t)[0]
            #     v = np.max(precision[mask_idxes])
            #     cls_ap = cls_ap + v / 11.
            ap_dict[c] = cls_ap
            total_ap = total_ap + cls_ap
        total_ap = total_ap / len(scores_dict.keys())
        return ap_dict, total_ap, info_dict, pr_dict