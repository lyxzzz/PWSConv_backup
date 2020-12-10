class ASSIGNER_CONFIG:
    positive_threshold = 0.5
    negative_threshold = 0.5

class DATA_CONFIG:
    ## train
    shape_list = [(300, 300)]
    epoch_list = [0]
    train_batch_nums = 32
    train_thread_nums = 8
    train_queue_size = 16

    ## test
    test_shape = (300, 300)
    test_batch_nums = 16
    test_thread_nums = 8
    test_queue_size = 16
    
class ANCHOR_CFG:
    feat_layers = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
    feat_offset_ratio = [1, 1, 1, 1, 1, 1]
    feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_sizes = [(21., 45.), (45., 99.), (99., 153.), (153., 207.), (207., 261.), (261., 315.)]
    anchor_ratios = [[1, 2, .5], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3], [1, 2, .5], [1, 2, .5]]
    anchor_steps = [8, 16, 32, 64, 100, 300]
    normalizations = [20, -1, -1, -1, -1, -1]
    anchor_offset = 0.5
    prior_scaling = [0.1, 0.1, 0.2, 0.2]

