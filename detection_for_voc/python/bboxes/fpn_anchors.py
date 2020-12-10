import numpy as np
import math
import itertools
def _anchors_one_layer(img_shape,
                        ratios,
                        anchor_scale,
                        feat_shape,
                        step,
                        offset=0.5,
                        dtype=np.float32):

    xasix = np.arange(0, feat_shape[1], dtype = np.float32)
    yasix = np.arange(0, feat_shape[0], dtype = np.float32)
    x, y = np.meshgrid(xasix,yasix)
    y = (y + offset) * step / img_shape[0]
    x = (x + offset) * step / img_shape[1]

    anchor_shape = x.shape

    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors = len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.

    assert img_shape[0] == img_shape[1]
    base_size = step * anchor_scale / img_shape[0]

    for i, ratio in enumerate(ratios):
        w[i] = base_size * ratio[0]
        h[i] = base_size * ratio[1]
    
    xmin = x - w / 2.
    ymin = y - h / 2.
    xmax = x + w / 2.
    ymax = y + h / 2.
    area = (xmax - xmin) * (ymax - ymin)
    return y, x, h, w, xmin, ymin, xmax, ymax, area, anchor_shape

def anchors(img_shape, anchorcfg, dtype=np.float32):
    layers_anchors = []
    ratios_layer = []

    anchor_scale = anchorcfg["anchor_scale"]

    for scale, ratio in itertools.product(anchorcfg["scales"], anchorcfg["ratios"]):
        base_scale = 2 ** (scale / 3.0)
        ratios_layer.append([base_scale * ratio[0], base_scale * ratio[1]])

    for i in range(len(anchorcfg["feat_shapes"])):
        anchor_bboxes = _anchors_one_layer(img_shape,
                                            ratios_layer,
                                            anchor_scale,
                                            anchorcfg["feat_shapes"][i],
                                            anchorcfg["anchor_steps"][i],
                                            offset=anchorcfg["anchor_offset"],
                                            dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def anchors_per_layer(anchorcfg, layer_index):
    return len(anchorcfg["scales"]) * len(anchorcfg["ratios"])