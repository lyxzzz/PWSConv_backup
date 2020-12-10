import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
import math
from utils.switch_norm import switch_norm

def unbalance_flatten_conv(inputs, output_size, kernel_size, start_pos, end_pos, alpha=3.0, scope=None):
    with tf.variable_scope(scope, 'unbalance_conv', [inputs]) as sc:
        trunk = slim.conv2d(inputs, output_size, [1, kernel_size], stride=1, 
                        activation_fn=None,
                        normalizer_fn=None,
                        padding='SAME',
                        scope='trunk_conv')
        
        branch_start = start_pos - int((kernel_size + 1) / 2)
        if branch_start <= 0:
            paddings = [[0, 0], [0, 0], [-branch_start, 0], [0, 0]]
            branch_inputs = tf.pad(inputs, paddings)
        else:
            branch_inputs = inputs[:, :, branch_start:, :]
        # print(branch_start)
        # print(branch_inputs.shape)
        branch_end = end_pos - int((kernel_size + 1) / 2)
        if branch_end >= 0:
            paddings = [[0, 0], [0, 0], [0, branch_end], [0, 0]]
            branch_inputs = tf.pad(branch_inputs, paddings)
        else:
            branch_inputs = branch_inputs[:, :, :branch_end, :]
        # print(branch_end)
        # print(branch_inputs.shape)
        branch_kernel_width = end_pos - start_pos + 1
        # print(branch_kernel_width)
        branch = slim.conv2d(branch_inputs, output_size, [1, branch_kernel_width], stride=1, 
                        activation_fn=None,
                        normalizer_fn=None,
                        padding='VALID',
                        scope='branch_conv')
        # alpha_value = tf.get_variable('alpha_value', [], initializer=tf.constant_initializer(alpha), trainable=True)
        # output = trunk + alpha_value * branch
        
        output = trunk + alpha * branch
        
        output = batch_norm(output, activation_fn=tf.nn.relu, scope='unbalance_bn')
        return output

@add_arg_scope
def layer_norm(inputs, epsilon=0.001, is_training=True, trainable=True, scope=None, activation_fn=None, reuse=None):
    with tf.variable_scope(scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        N, H, W, C = inputs.shape
        beta  = tf.get_variable('beta', 1, initializer=tf.constant_initializer(0), trainable=trainable)

        # 计算当前整个batch的均值与方差
        axes = [1, 2, 3]
        batch_mean, batch_var = tf.nn.moments(inputs,axes,name='moments')
        batch_mean = tf.reshape(batch_mean, [-1, 1, 1, 1])
        batch_var = tf.reshape(batch_var, [-1, 1, 1, 1])
        outputs = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, None, epsilon)
        # 最后执行batch normalization
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

@add_arg_scope
def group_norm(inputs, group_size=32, epsilon=0.001, is_training=True, trainable=True, scope=None, activation_fn=None, reuse=None):
    with tf.variable_scope(scope, 'GroupNorm', [inputs], reuse=reuse) as sc:
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        N, H, W, C = inputs.shape.as_list()
        inputs = tf.reshape(inputs, [-1, H, W, C // group_size, group_size]) 
        beta  = tf.get_variable('beta', C, initializer=tf.constant_initializer(0), trainable=trainable)

        mean, var = tf.nn.moments(inputs, [1, 2, 3], keep_dims=True)

        outputs = (inputs - mean) / tf.sqrt(var + epsilon)
        
        outputs = tf.reshape(outputs, [-1, H, W, C]) 

        outputs = outputs + beta
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

@add_arg_scope
def instance_norm(inputs, epsilon=0.001, is_training=True, trainable=True, scope=None, activation_fn=None, reuse=None):
    with tf.variable_scope(scope, 'InstanceNorm', [inputs], reuse=reuse) as sc:
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        N, H, W, C = inputs.shape
        beta  = tf.get_variable('beta', C, initializer=tf.constant_initializer(0), trainable=trainable)

        # 计算当前整个batch的均值与方差
        axes = [1, 2]
        batch_mean, batch_var = tf.nn.moments(inputs,axes,name='moments')
        batch_mean = tf.reshape(batch_mean, [-1, 1, 1, C])
        batch_var = tf.reshape(batch_var, [-1, 1, 1, C])
        outputs = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, None, epsilon)
        # 最后执行batch normalization
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

# @add_arg_scope
# def switch_norm(inputs, epsilon=1e-5, is_training=True, trainable=True, scope=None, activation_fn=None) :
#     with tf.variable_scope(scope, 'SwitchNorm', [inputs]) as sc:
#         N, H, W, C = inputs.shape

#         batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], keep_dims=True)
#         ins_mean, ins_var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
#         layer_mean, layer_var = tf.nn.moments(inputs, [1, 2, 3], keep_dims=True)

#         beta = tf.get_variable("beta", [C], initializer=tf.constant_initializer(0.0), trainable=trainable)

#         mean_weight = tf.nn.softmax(tf.get_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0)))
#         var_wegiht = tf.nn.softmax(tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0)))

#         mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
#         var = var_wegiht[0] * batch_var + var_wegiht[1] * ins_var + var_wegiht[2] * layer_var

#         outputs = (inputs - mean) / (tf.sqrt(var + eps)) + beta
#         if activation_fn is not None:
#             outputs = activation_fn(outputs)
#         return outputs


@add_arg_scope
def wn_conv(inputs, channels, kernel_size, weights_initializer, weights_regularizer, stride=1,
            epsilon=0.001, biases_initializer=0, single_beta=False,
            padding='SAME', activation_fn=None, normalizer_fn=None, normalizer_params=None, reuse=None, scope=None):
    N, H, W, C = inputs.shape.as_list()
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) == 1:
        kernel_size = [kernel_size[0], kernel_size[0]]

    if isinstance(stride, int):
        stride = [1, stride, stride, 1]
    elif len(stride) == 1:
        stride = [1, stride[0], stride[0], 1]
    elif len(stride) == 2:
        stride = [1, stride[0], stride[1], 1]

    with tf.variable_scope(scope, 'WNConv', [inputs], reuse=reuse) as sc: 
        filter_shape = [kernel_size[0], kernel_size[1], C, channels]

        bias_shape = [channels]


        kernel = slim.variable("kernel", shape=filter_shape,
                        initializer=tf.random_normal_initializer(0, 0.05),
                        regularizer=weights_regularizer,
                        trainable=True)
        
        g = slim.variable(name='wn/g', shape=(channels,),
                    initializer=tf.constant_initializer(1.0),
                    dtype=kernel.dtype,
                    trainable=True)

        V = tf.nn.l2_normalize(kernel, [0, 1, 2])
        g = tf.reshape(g, [1, 1, 1, channels])

        kernel = g * V

        output = tf.nn.conv2d(inputs, kernel, stride, padding=padding)

        bias = slim.variable("bias", shape=bias_shape,
                    initializer=biases_initializer,
                    regularizer=weights_regularizer,
                    trainable=True)
                        
        output = output + bias
        if activation_fn is not None:
            output = activation_fn(output)

        return output

@add_arg_scope
def ws_conv(inputs, channels, kernel_size, weights_initializer, weights_regularizer, stride=1,
            epsilon=0.001, biases_initializer=0, 
            padding='SAME', activation_fn=None, normalizer_fn=None, normalizer_params=None, reuse=None, scope=None):
    N, H, W, C = inputs.shape.as_list()
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) == 1:
        kernel_size = [kernel_size[0], kernel_size[0]]

    if isinstance(stride, int):
        stride = [1, stride, stride, 1]
    elif len(stride) == 1:
        stride = [1, stride[0], stride[0], 1]
    elif len(stride) == 2:
        stride = [1, stride[0], stride[1], 1]

    with tf.variable_scope(scope, 'WSConv', [inputs], reuse=reuse) as sc: 
        filter_shape = [kernel_size[0], kernel_size[1], C, channels]

        root_conv = slim.variable("root_filter", shape=filter_shape,
                        initializer=weights_initializer,
                        regularizer=weights_regularizer,
                        # device=None,
                        trainable=True)
    
        mean, var = tf.nn.moments(root_conv, [0, 1, 2], name='moments')
        mean = tf.reshape(mean, [1, 1, 1, -1])
        var = tf.reshape(var, [1, 1, 1, -1])

        root_conv = tf.nn.batch_normalization(root_conv, mean, var, None, None, epsilon)

        output = tf.nn.conv2d(inputs, root_conv, stride, padding=padding)

        normalizer_params = normalizer_params or {}
        output = normalizer_fn(output, **normalizer_params)
        
        if activation_fn is not None:
            output = activation_fn(output)

        return output
        
@add_arg_scope
def channel_norm(inputs, epsilon=0.001, is_training=True, trainable=True, scope=None, activation_fn=None, reuse=None):
    with tf.variable_scope(scope, 'ChannelNorm', [inputs], reuse=reuse) as sc:
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        N, H, W, C = inputs.shape
        beta  = tf.get_variable('beta', C, initializer=tf.constant_initializer(0), trainable=trainable)

        # 计算当前整个batch的均值与方差
        axes = [1, 2]
        batch_mean, batch_var = tf.nn.moments(inputs,axes,name='moments')
        batch_mean = tf.reshape(batch_mean, [-1, 1, 1, C])
        batch_var = tf.reshape(batch_var, [-1, 1, 1, C])
        outputs = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, None, epsilon)
        # 最后执行batch normalization
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

@add_arg_scope
def none_layer(is_training=True):
    print(is_training)

@add_arg_scope
def scale_transfer_layer(inputs, scale, scope=None):
    if scale < 2:
        return inputs
    with tf.variable_scope(scope, 'scale_transfer', [inputs]) as sc:
        N, H, W, C = inputs.shape.as_list()
        if C % (scale * scale) != 0:
            raise(EOFError)
        outputs = tf.depth_to_space(inputs, scale)
        # C = int(C / scale / scale)
        # H = int(H * scale)
        # W = int(W * scale)
        # outputs = tf.reshape(inputs, [-1, H, W, C])
        # outputs = tf.reshape(inputs, [-1, H, W, C])
        return outputs

@add_arg_scope
def upsample_layer(inputs, scale, scope=None):
    if scale < 2:
        return inputs
    with tf.variable_scope(scope, 'upsample', [inputs]) as sc:
        N, H, W, C = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1, H, 1, W, 1, C])
        ones =  tf.ones([1, 1, scale, 1, scale, 1], dtype=inputs.dtype)
        outputs = inputs * ones
        outputs = tf.reshape(outputs, [-1, H * scale, W * scale, C])
        return outputs

@add_arg_scope
def deconv_layer(inputs, channels, kernel_size, scale, weights_initializer, weights_regularizer, biases_initializer=0,
            activation_fn=None, normalizer_fn=None, normalizer_params=None, scope=None):
    with tf.variable_scope(scope, 'deconv', [inputs]) as sc:
        use_bias= normalizer_fn is None
        output = tf.layers.conv2d_transpose(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            strides=scale,
            kernel_initializer=weights_initializer,
            kernel_regularizer=weights_regularizer,
            bias_regularizer=weights_regularizer,
            padding='same',
            use_bias=use_bias,
            activation=None)
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            output = normalizer_fn(output, **normalizer_params)
        if activation_fn is not None:
            output = activation_fn(output)
    return output

# @add_arg_scope
# def pws_conv(inputs, channels, kernel_size, weights_initializer, weights_regularizer, stride=1,
#             epsilon=0.001, biases_initializer=0, single_beta=False,
#             padding='SAME', activation_fn=None, normalizer_fn=None, normalizer_params=None, reuse=None, scope=None):
#     N, H, W, C = inputs.shape.as_list()
#     if isinstance(kernel_size, int):
#         kernel_size = [kernel_size, kernel_size]
#     elif len(kernel_size) == 1:
#         kernel_size = [kernel_size[0], kernel_size[0]]

#     if isinstance(stride, int):
#         stride = [1, stride, stride, 1]
#     elif len(stride) == 1:
#         stride = [1, stride[0], stride[0], 1]
#     elif len(stride) == 2:
#         stride = [1, stride[0], stride[1], 1]

#     with tf.variable_scope(scope, 'PWSCONV', [inputs], reuse=reuse) as sc: 
#         filter_shape = [kernel_size[0], kernel_size[1], C, channels]
        # if single_beta == True:
        #     bias_shape = []
        # else:
        #     bias_shape = [channels]
#         var_value = kernel_size[0] * kernel_size[1] * C 
#         var_value = math.sqrt(2 / float(var_value))

#         root_conv = slim.variable("root_filter", shape=filter_shape,
#                         initializer=weights_initializer,
#                         regularizer=weights_regularizer,
#                         # device=None,
#                         trainable=True)
    
#         mean, var = tf.nn.moments(root_conv, [0, 1, 2], name='moments')
#         mean = tf.reshape(mean, [1, 1, 1, -1])
#         var = tf.reshape(var, [1, 1, 1, -1])

#         # tf.add_to_collection("filter", root_conv)
#         # # after_norm_conv = multi_gamma * var_value * tf.nn.batch_normalization(root_conv, mean, var, None, None, epsilon)
#         # # tf.add_to_collection("filter_b", after_norm_conv)

#         # tf.add_to_collection("conv_input", inputs)

#         # def __cal_person(X, W):
#         #     info_X = tf.nn.moments(X, [0, 1, 2, 3], name="X_moment")
#         #     info_W = tf.nn.moments(W, [0, 1, 2, 3], name="W_moment")
#         #     W_shape = W.shape.as_list()
#         #     W_mul = W_shape[0] * W_shape[1] * W_shape[2]
#         #     XW = tf.nn.conv2d(X, W, [1, 1, 1, 1], padding="SAME")
#         #     info_XW = tf.nn.moments(XW, [0, 1, 2, 3], name="W_moment")
#         #     per = (info_XW[0] / W_mul - info_X[0] * info_W[0]) / (tf.sqrt(info_X[1]) * tf.sqrt(info_W[1]))

#         #     return per
#         # per = __cal_person(inputs, root_conv)
#         # tf.add_to_collection("per", per)
#         output = tf.nn.conv2d(inputs, root_conv, stride, padding=padding)
#         # tf.add_to_collection("conv_output", output)



#         if normalizer_fn is not None:
#             normalizer_params = normalizer_params or {}
#             output = normalizer_fn(output, **normalizer_params)
#         else:
#             root_bias = slim.variable("root_bias", shape=bias_shape,
#                         initializer=biases_initializer,
#                         regularizer=weights_regularizer,
#                         # device=None,
#                         trainable=True)
                        
#             output = output + root_bias
#         if activation_fn is not None:
#             output = activation_fn(output)

#         return output
# @add_arg_scope
# def unbalance_conv(inputs, channels, kernel_size, weights_initializer, weights_regularizer, rate_initializer, stride=1,
#             multiplier=1.0, normalization_rate=False, normalization_point=1.0, epsilon=0.001, biases_initializer=0, single_beta=False,
#             padding='SAME', activation_fn=None, normalizer_fn=None, normalizer_params=None, reuse=None, scope=None):
#     N, H, W, C = inputs.shape.as_list()
#     if isinstance(kernel_size, int):
#         kernel_size = [kernel_size, kernel_size]
#     elif len(kernel_size) == 1:
#         kernel_size = [kernel_size[0], kernel_size[0]]

#     if isinstance(stride, int):
#         stride = [1, stride, stride, 1]
#     elif len(stride) == 1:
#         stride = [1, stride[0], stride[0], 1]
#     elif len(stride) == 2:
#         stride = [1, stride[0], stride[1], 1]

#     with tf.variable_scope(scope, 'UnbalanceConv', [inputs], reuse=reuse) as sc: 
#         filter_shape = [kernel_size[0], kernel_size[1], C, channels]
#         if single_beta == True:
#             bias_shape = []
#         else:
#             bias_shape = [channels]

#         root_conv = slim.variable("root_filter", shape=filter_shape,
#                         initializer=weights_initializer,
#                         regularizer=weights_regularizer,
#                         # device=None,
#                         trainable=True)

#         output = tf.nn.conv2d(inputs, root_conv, stride, padding=padding)

#         if normalizer_fn is not None:
#             # mean, var = tf.nn.moments(output, [0, 1, 2], name='moments')
#             # mean = tf.reshape(mean, [1, 1, 1, -1])
#             # var = tf.sqrt(tf.reshape(var, [1, 1, 1, -1]) + epsilon)
            
#             # normalizer_params = normalizer_params or {}
#             # output = normalizer_fn(output, center=False, **normalizer_params)

#             with tf.variable_scope("BatchNorm"):
#                 var = tf.get_variable('moving_mean', shape=[channels], trainable=False)
#                 output = output - var
#             # mean1, var1 = tf.nn.moments(output, [0, 1, 2], name='moments')
#             # mean1 = tf.reshape(mean1, [1, 1, 1, -1])
#             # var1 = tf.reshape(var1, [1, 1, 1, -1])

#             # output = output * var

#             root_bias = slim.variable("root_bias", shape=bias_shape,
#                         initializer=biases_initializer,
#                         regularizer=weights_regularizer,
#                         # device=None,
#                         trainable=True)
                        
#             output = output + root_bias
#         if activation_fn is not None:
#             output = activation_fn(output)

#         return output

@add_arg_scope
def pws_conv(inputs, channels, kernel_size, weights_initializer, weights_regularizer, stride=1,
            epsilon=0.001, biases_initializer=0, single_beta=False,
            padding='SAME', activation_fn=None, normalizer_fn=None, normalizer_params=None, reuse=None, scope=None):
    N, H, W, C = inputs.shape.as_list()
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) == 1:
        kernel_size = [kernel_size[0], kernel_size[0]]

    if isinstance(stride, int):
        stride = [1, stride, stride, 1]
    elif len(stride) == 1:
        stride = [1, stride[0], stride[0], 1]
    elif len(stride) == 2:
        stride = [1, stride[0], stride[1], 1]

    with tf.variable_scope(scope, 'PWSConv', [inputs], reuse=reuse) as sc: 
        filter_shape = [kernel_size[0], kernel_size[1], C, channels]
        if single_beta == True:
            bias_shape = []
        else:
            bias_shape = [channels]
        gamma = epsilon
        kernels = slim.variable("root_filter", shape=filter_shape,
                        initializer=weights_initializer,
                        regularizer=weights_regularizer,
                        # device=None,
                        trainable=True)
    
        alpha = slim.variable("alpha", shape=bias_shape,
                initializer=tf.constant_initializer(1.0),
                # device=None,
                trainable=True)

        alpha = tf.reshape(alpha, [1, 1, 1, -1])
    
        mean, var = tf.nn.moments(kernels, [0, 1, 2], name='moments', keep_dims=True)
        var_value = math.sqrt(2 / float(kernel_size[0] * kernel_size[1] * C))
        root_conv = alpha * var_value * tf.nn.batch_normalization(kernels, mean, var, None, None, gamma)
        #normal type is:output = tf.nn.conv2d(inputs, root_conv, stride, padding=padding)
        output = tf.nn.conv2d(inputs, root_conv, stride, padding=padding)

        # if normalizer_fn is not None:
        #     normalizer_params = normalizer_params or {}
        #     output = normalizer_fn(output, **normalizer_params)
        # else:
        root_bias = slim.variable("root_bias", shape=bias_shape,
                    initializer=biases_initializer,
                    regularizer=weights_regularizer,
                    # device=None,
                    trainable=True)
                    
        output = output + root_bias
        if activation_fn is not None:
            output = activation_fn(output)

        return output




@add_arg_scope
def myconv(inputs, channels, kernel_size, weights_initializer, weights_regularizer, stride=1, epsilon=0.001, biases_initializer=0, single_beta=True,
            padding='SAME', activation_fn=None, normalizer_fn=None, normalizer_params=None, reuse=None, scope=None):
    N, H, W, C = inputs.shape.as_list()
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) == 1:
        kernel_size = [kernel_size[0], kernel_size[0]]

    if isinstance(stride, int):
        stride = [1, stride, stride, 1]
    elif len(stride) == 1:
        stride = [1, stride[0], stride[0], 1]
    elif len(stride) == 2:
        stride = [1, stride[0], stride[1], 1]

    with tf.variable_scope(scope, 'MyConv', [inputs], reuse=reuse) as sc:
        filter_shape = [kernel_size[0], kernel_size[1], C, channels]
        if single_beta == True:
            bias_shape = []
        else:
            bias_shape = [channels]

        root_conv = slim.variable("root_filter", shape=filter_shape,
                        initializer=weights_initializer,
                        regularizer=weights_regularizer,
                        # device=None,
                        trainable=True)
        root_bias = slim.variable("root_bias", shape=bias_shape,
                        initializer=biases_initializer,
                        regularizer=weights_regularizer,
                        # device=None,
                        trainable=True)

        output = tf.nn.conv2d(inputs, root_conv, stride, padding=padding)

        output = output + root_bias
        if activation_fn is not None:
            output = activation_fn(output)

        return output
# @add_arg_scope
# def unbalance_conv(inputs, channels, kernel_size, weights_initializer, weights_regularizer, rate_initializer, stride=1,
#             multiplier=1.0, normalization_rate=False, normalization_point=1.0, epsilon=0.001, biases_initializer=0, 
#             padding='SAME', activation_fn=None, normalizer_fn=None, normalizer_params=None, reuse=None, scope=None):
#     N, H, W, C = inputs.shape.as_list()
#     if isinstance(kernel_size, int):
#         kernel_size = [kernel_size, kernel_size]
#     elif len(kernel_size) == 1:
#         kernel_size = [kernel_size[0], kernel_size[0]]

#     if isinstance(stride, int):
#         stride = [1, stride, stride, 1]
#     elif len(stride) == 1:
#         stride = [1, stride[0], stride[0], 1]
#     elif len(stride) == 2:
#         stride = [1, stride[0], stride[1], 1]

#     with tf.variable_scope(scope, 'UnbalanceConv', [inputs], reuse=reuse) as sc:
#         filter_shape = [kernel_size[0], kernel_size[1], C, channels]
#         map_shape = [kernel_size[0], kernel_size[1], 1, channels]
#         bias_shape = [channels]
#         root_conv = slim.variable("root_filter", shape=filter_shape,
#                         initializer=weights_initializer,
#                         regularizer=weights_regularizer,
#                         # device=None,
#                         trainable=True)
#         root_bias = slim.variable("root_bias", shape=bias_shape,
#                         initializer=biases_initializer,
#                         regularizer=weights_regularizer,
#                         # device=None,
#                         trainable=True)
        
#         multi_gamma = slim.variable("gamma", shape=1,
#                         initializer=tf.constant_initializer(multiplier),
#                         # device=None,
#                         trainable=False)
        
#         rate_map = slim.variable("rate_map", shape=map_shape,
#                         initializer=rate_initializer,
#                         regularizer=weights_regularizer,
#                         # device=None,
#                         trainable=True)
        
#         rate_bias = slim.variable("rate_bias", shape=bias_shape,
#                         initializer=biases_initializer,
#                         regularizer=weights_regularizer,
#                         # device=None,
#                         trainable=True)
        
#         if normalization_rate:
#             # min_rate = tf.reduce_min(rate_map, [0, 1, 2], name='minrate')
#             # max_rate = tf.reduce_max(rate_map, [0, 1, 2], name='maxrate')
#             # max_rate = tf.reshape(max_rate, [1, 1, 1, -1])
#             # min_rate = tf.reshape(min_rate, [1, 1, 1, -1])

#             # (rate_map - min_rate) / ()
#             mean, var = tf.nn.moments(rate_map, [0, 1, 2], name='moments')
#             mean = tf.reshape(mean, [1, 1, 1, -1])
#             var = tf.reshape(var, [1, 1, 1, -1])

#             rate_map = normalization_point + tf.nn.batch_normalization(rate_map, mean, var, None, None, epsilon)
        
#         real_rate_map = multi_gamma * rate_map + rate_bias
#         # print(real_rate_map.shape)
#         real_filter = root_conv * real_rate_map
#         # print(real_filter.shape)
#         # exit()

#         output = tf.nn.conv2d(inputs, real_filter, stride, padding=padding)

#         if normalizer_fn is not None:
#             normalizer_params = normalizer_params or {}
#             output = normalizer_fn(output, **normalizer_params)
#         else:
#             output = output + root_bias
#         if activation_fn is not None:
#             output = activation_fn(output)

#         return output

# def image_subtraction(images, means=[116., 111., 102.], var=[61., 60., 61.]):
#     # result = images - means
#     # result = result / var
#     return result

def global_average_pool(net, scope='global_avg'):
    kernel_size = net.get_shape()[1:3]
    if kernel_size[0] == 1 and kernel_size[1] == 1:
        return net
        
    net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope=scope)
    return net

def l1_smooth(x, sigma2=1.0):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    # absx = tf.abs(x)
    # minx = tf.minimum(absx, 1)
    # r = 0.5 * ((absx - 1) * minx + absx)
    sigma2 = sigma2
    abs = tf.abs(x)
    smoothL1_sign = tf.cast(tf.less(abs, 1.0 / sigma2), tf.float32)
    return tf.square(x) * 0.5 * sigma2 * smoothL1_sign + \
            (abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)
