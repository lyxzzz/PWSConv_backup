import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from tensorflow.contrib.slim import nets

from utils import tf_layer

def arg_scope(weight_decay=0.00005):
    """Network arg_scope.
    """
    batch_norm_params = {
    }
    with slim.arg_scope([slim.conv2d],
                    # normalizer_fn=tf_layer.batch_norm,
                    # normalizer_params=batch_norm_params
                    ):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME') as sc:
                return sc

def vgg_16(inputs, num_classes=20, is_training=True, dropout_prob=0.5, scope=None):
    end_points = {}
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        with slim.arg_scope([tf_layer.batch_norm, slim.dropout],
                    is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 1, slim.conv2d, 512, [3, 3], scope='conv5')
                
                # Use conv2d instead of fully_connected layers.
                kernel_size = net.shape.as_list()[1:3]
                net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='avg_pool')
                net = slim.dropout(net, dropout_prob, is_training=is_training,
                                    scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='myfc7')

                if num_classes:
                    net = slim.dropout(net, dropout_prob, is_training=is_training,
                                    scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='logits')
                end_points['Logits'] = net
                end_points['Predictions'] = tf.nn.softmax(net, name='predictions')
    return net, end_points