class ASSIGNER_CONFIG:
    positive_threshold = 0.5
    negative_threshold = 0.4

class DATA_CONFIG:
    ## train
    shape_list = [(320, 320)]
    epoch_list = [0]
    train_batch_nums = 32
    train_thread_nums = 8
    train_queue_size = 16

    ## test
    test_shape = (320, 320)
    test_batch_nums = 16
    test_thread_nums = 8
    test_queue_size = 16
    
# class ANCHOR_CFG:
#     feat_layers = ['block4', 'block5', 'block6', 'block7', 'block8', 'block9']
#     feat_shapes = [(40,40), (20,20), (10,10), (5,5), (3, 3), (1, 1)]
#     extra_layer = [3, 3, 3, 3, 1, 1]
#     extra_name = ['square', 'xaxis', 'yaxis']
#     anchor_sizes = [(21., 45.), (45., 99.), (99., 153.), (153., 207.), (207., 261.), (261., 315.)]
#     anchor_ratios = [[1, -1, 2., .5],  [3., 5.], [1/3., 1/5.]]
#     anchor_steps = [8, 16, 32, 64, 106, 320]
#     # normalizations = [20, -1, -1, -1, -1, -1]
#     anchor_offset = 0.5
#     prior_scaling = [0.1, 0.1, 0.2, 0.2]

class ANCHOR_CFG:
    feat_layers = ['block4', 'block5', 'block6', 'block7', 'block8', 'block9']
    feat_shapes = [(40,40), (20,20), (10,10), (5,5), (3, 3), (1, 1)]
    extra_layer = [1, 3, 3, 3, 1, 1]
    extra_name = ['square', 'xaxis', 'yaxis']
    anchor_sizes = [(32.0,64.0),(64.0,118.4),(118.4,172.8),(172.8,227.2),(227.2,281.6),(281.6,336.0)]
    anchor_ratios = [[1, -1, 2., .5],  [3.], [1/3.]]
    anchor_steps = [8, 16, 32, 64, 107, 320]
    # normalizations = [20, -1, -1, -1, -1, -1]
    anchor_offset = 0.5
    prior_scaling = [0.1, 0.1, 0.2, 0.2]

# class ANCHOR_CFG:
#     feat_layers = ['block3','block4', 'block5', 'block6', 'block7', 'block8', 'block9']
#     feat_shapes = [(40,40),(40,40), (20,20), (10,10), (5,5), (3, 3), (1, 1)]
#     extra_layer = [1,1, 3, 3, 3, 1, 1]
#     extra_name = ['square', 'xaxis', 'yaxis']
#     anchor_sizes = [(20,40),(32.0,64.0),(64.0,118.4),(118.4,172.8),(172.8,227.2),(227.2,281.6),(281.6,336.0)]
#     anchor_ratios = [[1, -1, 2., .5],  [3.], [1/3.]]
#     anchor_steps = [8,8, 16, 32, 64, 107, 320]
#     # normalizations = [20, -1, -1, -1, -1, -1]
#     anchor_offset = 0.5
#     prior_scaling = [0.1, 0.1, 0.2, 0.2]
