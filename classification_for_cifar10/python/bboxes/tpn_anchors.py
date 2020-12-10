import numpy as np
import math
def _anchors_one_layer(img_shape,
                        feat_shape,
                        sizes,
                        ratios,
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
    base_h = [0.0, 0.0]
    base_w = [0.0, 0.0]
    base_h[0] = sizes[0] / img_shape[0]
    base_w[0] = sizes[0] / img_shape[1]
    base_h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
    base_w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]

    for i, r in enumerate(ratios):
        if r == 1:
            h[i] = base_h[0]
            w[i] = base_w[0]
        elif r <= 0:
            h[i] = base_h[1]
            w[i] = base_w[1]
        else:
            ratio = math.sqrt(r)
            h[i] = base_h[0] / ratio
            w[i] = base_w[0] * ratio
    
    xmin = x - w / 2.
    ymin = y - h / 2.
    xmax = x + w / 2.
    ymax = y + h / 2.
    area = (xmax - xmin) * (ymax - ymin)
    return y, x, h, w, xmin, ymin, xmax, ymax, area, anchor_shape

def anchors(img_shape, networkcfgs, dtype=np.float32):
    layers_anchors = []
    for i in range(len(networkcfgs.feat_shapes)):
        ratios_layer = []
        for layer_index in range(networkcfgs.extra_layer[i]):
            ratios_layer += networkcfgs.anchor_ratios[layer_index]

        anchor_bboxes = _anchors_one_layer(img_shape,
                                            networkcfgs.feat_shapes[i],
                                            networkcfgs.anchor_sizes[i],
                                            ratios_layer,
                                            networkcfgs.anchor_steps[i],
                                            offset=networkcfgs.anchor_offset,
                                            dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors