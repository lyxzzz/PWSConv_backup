from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages
import tensorflow as tf

DATA_FORMAT_NHWC = 'NHWC'

def __switch_norm_training(inputs, offset, epsilon=0.001, mean_weight=None, var_wegiht=None, name=None):
  
	inputs = ops.convert_to_tensor(inputs, name="input")
	offset = ops.convert_to_tensor(offset, name="offset")

	min_epsilon = 1.001e-5
	epsilon = epsilon if epsilon > min_epsilon else min_epsilon

	in_mean, in_var = tf.nn.moments(inputs, [1, 2], keep_dims=True)

	squre_inputs = in_var + tf.square(in_mean)

	ln_mean = tf.reduce_mean(in_mean, [3], keep_dims=True)

	ln_var = tf.reduce_mean(squre_inputs, [3], keep_dims=True) - tf.square(ln_mean)

	bn_mean = tf.reduce_mean(in_mean, [0], keep_dims=True)

	bn_var = tf.reduce_mean(squre_inputs, [0], keep_dims=True) - tf.square(bn_mean)

	mean = mean_weight[0] * bn_mean + mean_weight[1] * in_mean + mean_weight[2] * ln_mean
	var = var_wegiht[0] * bn_var + var_wegiht[1] * in_var + var_wegiht[2] * ln_var

	outputs = (inputs - mean) / (tf.sqrt(var + epsilon)) + offset

	return outputs, bn_mean, bn_var

def __switch_norm_inference(inputs, offset, mean, variance, epsilon=0.001, mean_weight=None, var_wegiht=None, name=None):
  
	inputs = ops.convert_to_tensor(inputs, name="input")
	offset = ops.convert_to_tensor(offset, name="offset")

	min_epsilon = 1.001e-5
	epsilon = epsilon if epsilon > min_epsilon else min_epsilon

	in_mean, in_var = tf.nn.moments(inputs, [1, 2], keep_dims=True)

	squre_inputs = in_var + tf.square(in_mean)

	ln_mean = tf.reduce_mean(in_mean, [3], keep_dims=True)

	ln_var = tf.reduce_mean(squre_inputs, [3], keep_dims=True) - tf.square(ln_mean)

	bn_mean = mean

	bn_var = variance

	mean = mean_weight[0] * bn_mean + mean_weight[1] * in_mean + mean_weight[2] * ln_mean
	var = var_wegiht[0] * bn_var + var_wegiht[1] * in_var + var_wegiht[2] * ln_var

	outputs = (inputs - mean) / (tf.sqrt(var + epsilon)) + offset

	return outputs, bn_mean, bn_var

@add_arg_scope
def switch_norm(inputs,
					  fused=False,
					  decay=0.999,
					  center=True,
					  scale=False,
					  epsilon=0.001,
					  activation_fn=None,
					  param_initializers=None,
					  param_regularizers=None,
					  updates_collections=ops.GraphKeys.UPDATE_OPS,
					  is_training=True,
					  reuse=None,
					  variables_collections=None,
					  outputs_collections=None,
					  trainable=True,
					  data_format=DATA_FORMAT_NHWC,
					  zero_debias_moving_mean=False,
					  scope=None):
	with variable_scope.variable_scope(scope, 'SwitchNorm', [inputs], reuse=reuse) as sc:
		inputs = ops.convert_to_tensor(inputs)
		original_shape = inputs.get_shape()
		original_inputs = inputs
		original_rank = original_shape.ndims

		if original_rank is None:
			raise ValueError('Inputs %s has undefined rank' % inputs.name)
		elif original_rank not in [2, 4]:
	  		raise ValueError('Inputs %s has unsupported rank.'
					   ' Expected 2 or 4 but got %d' %
					   (inputs.name, original_rank))

		inputs_shape = inputs.get_shape()

		if data_format == DATA_FORMAT_NHWC:
			params_shape = inputs_shape[-1:]
			bn_shape = [1, 1, 1, int(inputs_shape[-1])]

		if not params_shape.is_fully_defined():
			raise ValueError('Inputs %s has undefined `C` dimension %s.' %
						(inputs.name, params_shape))

		# Allocate parameters for the beta and gamma of the normalization.
		beta_collections = utils.get_variable_collections(variables_collections,
														'beta')
		
		# Float32 required to avoid precision-loss when using fp16 input/output
		variable_dtype = dtypes.float32

		if not param_initializers:
			param_initializers = {}
		if not param_regularizers:
			param_regularizers = {}

		beta_regularizer = param_regularizers.get('beta')
		gamma_regularizer = param_regularizers.get('gamma')

		if center:
			beta_initializer = param_initializers.get('beta', init_ops.zeros_initializer())
			beta = variables.model_variable('beta',
				shape=params_shape,
				dtype=variable_dtype,
				initializer=beta_initializer,
				regularizer=beta_regularizer,
				collections=beta_collections,
				trainable=trainable)
		else:
			beta = array_ops.constant(0.0, dtype=variable_dtype, shape=params_shape)

		if scale:
			gamma_collections = utils.get_variable_collections(
				variables_collections, 'gamma')
			gamma_initializer = param_initializers.get('gamma',
														init_ops.ones_initializer())
			gamma = variables.model_variable('gamma',
				shape=params_shape,
				dtype=variable_dtype,
				initializer=gamma_initializer,
				regularizer=gamma_regularizer,
				collections=gamma_collections,
				trainable=trainable)
		else:
			gamma = array_ops.constant(1.0, dtype=variable_dtype, shape=params_shape)

		mean_weight = tf.nn.softmax(tf.get_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0)))
		var_wegiht = tf.nn.softmax(tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0)))

		# Create moving_mean and moving_variance variables and add them to the
		# appropriate collections. We disable variable partitioning while creating
		# them, because assign_moving_average is not yet supported for partitioned
		# variables (this needs to be handled carefully, as it may break
		# the checkpoint backward compatibility).
		with variable_scope.variable_scope(variable_scope.get_variable_scope()) as local_scope:
			local_scope.set_partitioner(None)
			moving_mean_collections = utils.get_variable_collections(
				variables_collections, 'moving_mean')

			moving_mean_initializer = param_initializers.get(
				'moving_mean', init_ops.zeros_initializer())

			moving_mean = variables.model_variable(
				'moving_mean',
				shape=bn_shape,
				dtype=variable_dtype,
				initializer=moving_mean_initializer,
				trainable=False,
				collections=moving_mean_collections)

			moving_variance_collections = utils.get_variable_collections(
				variables_collections, 'moving_variance')

			moving_variance_initializer = param_initializers.get(
				'moving_variance', init_ops.ones_initializer())

			moving_variance = variables.model_variable(
				'moving_variance',
				shape=bn_shape,
				dtype=variable_dtype,
				initializer=moving_variance_initializer,
				trainable=False,
				collections=moving_variance_collections)

			def _fused_switch_norm_training():
				return __switch_norm_training(
					inputs, beta, epsilon=epsilon, mean_weight=mean_weight, var_wegiht=var_wegiht)

			def _fused_switch_norm_inference():
				return __switch_norm_inference(
					inputs,
					beta,
					mean=moving_mean,
					variance=moving_variance,
					epsilon=epsilon,
					mean_weight=mean_weight, var_wegiht=var_wegiht)

			outputs, mean, variance = utils.smart_cond(is_training,
													_fused_switch_norm_training,
													_fused_switch_norm_inference)

		# If `is_training` doesn't have a constant value, because it is a `Tensor`,
		# a `Variable` or `Placeholder` then is_training_value will be None and
		# `need_updates` will be true.
		is_training_value = utils.constant_value(is_training)
		need_updates = is_training_value is None or is_training_value
		if need_updates:
			if updates_collections is None:
				no_updates = lambda: outputs

				def _force_updates():
					"""Internal function forces updates moving_vars if is_training."""
					update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
					update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay, zero_debias=False)
					with ops.control_dependencies([update_moving_mean, update_moving_variance]):
						return array_ops.identity(outputs)

				outputs = utils.smart_cond(is_training, _force_updates, no_updates)
			else:
				moving_vars_fn = lambda: (moving_mean, moving_variance)

				def _delay_updates():
					"""Internal function that delay updates moving_vars if is_training."""
					update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
					update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay, zero_debias=False)
					return update_moving_mean, update_moving_variance

				update_mean, update_variance = utils.smart_cond(is_training, _delay_updates, moving_vars_fn)

				ops.add_to_collections(updates_collections, update_mean)
				ops.add_to_collections(updates_collections, update_variance)

		outputs.set_shape(inputs_shape)

		if original_shape.ndims == 2:
			outputs = array_ops.reshape(outputs, array_ops.shape(original_inputs))
		if activation_fn is not None:
			outputs = activation_fn(outputs)
		return utils.collect_named_outputs(outputs_collections, sc.name, outputs)