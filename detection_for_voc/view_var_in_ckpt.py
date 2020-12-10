from tensorflow.python import pywrap_tensorflow
import os
import numpy as np

def normalization(input):
    H, W, IN, OUT = input.shape
    # print(input.shape)
    input = np.array(input, dtype=np.float)
    input = input.reshape([-1, OUT])
    mean = np.mean(input, axis=0)
    var = np.var(input, axis=0)
    epsilon = 0.001
    input = (input - mean)/np.sqrt((var + epsilon))
    print(np.max(input, axis=0) - np.min(input,axis=0))

def info(var, mul=1.0):
    print("shape:{}, var:{}, mean:{}, mul:{}, \n\t\tafter mul, var:{}, mean:{}\n".format(var.shape, np.var(var), np.mean(var), mul, np.var(var) * mul, np.mean(var) * mul))    # print(input.shape)
# checkpoint_path = os.path.join('checkpoints/ssd/ssd_300_vgg', "ssd_300_vgg.ckpt")
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
# var_to_shape_map1 = reader.get_variable_to_shape_map()

checkpoint_path = os.path.join('checkpoints/tpn', "tpn_last.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map2 = reader.get_variable_to_shape_map()
while True:
    key = input("input:")
    if key == "exit":
        break
    else:
        val_list = []
        name_list = []
        for name in var_to_shape_map2.keys():
            if key in name and "Momentum" not in name and "ExponentialMovingAverage" not in name:
                val = reader.get_tensor(name)
                print("{},{}".format(val.shape, name))
                val_list.append(val)
                name_list.append(name)
        while True:
            key = input("query:")
            if key == "q":
                break
            if key == "p":
                for i, v in enumerate(val_list):
                    print(name_list[i])
            else:
                for i, name in enumerate(name_list):
                    if key in name:
                        val = val_list[i]
                        print("{},{}".format(val.shape, name))
                        if len(val.shape) > 3:
                            info(val, val.shape[0] * val.shape[1] * val.shape[2])
                        else:
                            info(val, val.shape[-1])
                        if key == "rate_map":
                            normalization(val)
        
# print(var_to_shape_map1)
# print(var_to_shape_map2)