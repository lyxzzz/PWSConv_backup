import datetime
import os
import sys
import numpy as np
import json
import cv2
sys.path.append('python')

from utils import tf_layer
import tensorflow as tf
from tensorflow.contrib import slim

num = tf.Variable(100)
params = {'is_training':False, 'num':num}
with slim.arg_scope([tf_layer.none_layer], **params):
    a = tf_layer.none_layer()
    c = a[1] + tf.Variable(-20)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run([c]))