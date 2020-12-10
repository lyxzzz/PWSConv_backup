class ASSIGNER_CONFIG:
    positive_threshold = 0.5
    negative_threshold = 0.4

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
    feat_layers = ['densenet/dense_module_3/block31', 'densenet/dense_module_3/block24', 'densenet/dense_module_3/block19',
                'densenet/dense_module_3/block14', 'densenet/dense_module_3/block9', 'densenet/dense_module_3/block4']
    feat_shapes = [(40, 40), (20, 20), (10, 10), (5, 5), (3, 3), (1, 1)]
    feat_scales = [4, 2, 1, -2, -4, -10]
    # anchor_sizes = [(21., 45.), (45., 99.), (99., 153.), (153., 207.), (207., 261.), (261., 315.)]
    anchor_sizes = [(32.0,64.0),(64.0,118.4),(118.4,172.8),(172.8,227.2),(227.2,281.6),(281.6,336.0)]
    anchor_ratios = [[1, 1.6, 2.0, 3.0, .5], [1, 1.6, 2, .5, 3, 1./3], [1, 1.6, 2, .5, 3, 1./3],
                 [1, 1.6, 2, .5, 3, 1./3], [1, 1.6, 2, 3, .5], [1, 1.6, 2, 3.0, .5]]
    anchor_steps = [8, 16, 32, 64, 100, 300]
    prior_scaling = [0.1, 0.1, 0.2, 0.2]
    feat_offset_ratio = [1, 1, 1, 1, 1, 1]
    anchor_offset = 0.5

