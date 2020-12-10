import numpy as numpy

def _safe_divide(a, b):
    if b == 0.0:
        return 0.0
    else:
        return a/b

def f_measure(eval_array):
    '''
    eval_array:
    [True_positive, Predict_num, True_nums]
    return:
    [f-measure, precision, recall]
    '''
    
    recall = _safe_divide(eval_array[0] , float(eval_array[2]))
    precision = _safe_divide(eval_array[0] , float(eval_array[1]))
    f = _safe_divide(2 * recall * precision , (recall + precision))
    return [100*f, 100*precision, 100*recall]