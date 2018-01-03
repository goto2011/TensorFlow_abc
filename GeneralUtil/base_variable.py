#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf


def base_variable_dump(sess):
	with sess.as_default():
		print("input_node=%d" % (get_input_node().eval()))
		print("output_node=%d" % (get_output_node().eval()))
		print("batch_size=%d" % (get_batch_size().eval()))
		print("learning_rate_base=%f" % (get_learning_rate_base().eval()))
		print("learning_rate_decay=%f" % (get_learning_rate_decay().eval()))
		print("regularization_rate=%f" % (get_regularization_rate().eval()))
		print("training_steps=%d" % (get_training_steps().eval()))
		print("moving_average_decay=%f" % (get_moving_average_decay().eval()))

		'''
		print("layer_count=%d" % (get_layer_count()))
		for ii in range(get_layer_count()):
			print("layer_tensor[%d]=%d" % (ii, get_gived_layer(ii)))
		'''

# 初始化除隐层节点数量之外的基本的神经网络参数。
def init_base_variable(input_node, output_node, batch_size, learning_rate_base, learning_rate_decay, regularization_rate, training_steps, moving_average_decay):
	with tf.variable_scope('base_variable'):
		# 1.输入层节点数
		tf.get_variable("input_node", [1], initializer=tf.constant_initializer(input_node), dtype=tf.int32, trainable=False)
		# 2.输出层节点数
		tf.get_variable("output_node", [1], initializer=tf.constant_initializer(output_node), dtype=tf.int32, trainable=False)
		# 3.每次batch打包的样本个数
		tf.get_variable("batch_size", [1], initializer=tf.constant_initializer(batch_size), dtype=tf.int32, trainable=False)
		# 4.基础学习learning_rate_base率
		tf.get_variable("learning_rate_base", [1], initializer=tf.constant_initializer(learning_rate_base), dtype=tf.float32, trainable=False)
		# 5.学习率的衰减率
		tf.get_variable("learning_rate_decay", [1], initializer=tf.constant_initializer(learning_rate_decay), dtype=tf.float32, trainable=False)
		# 6.描述模型复杂度的正则化项在损失函数中的系数
		tf.get_variable("regularization_rate", [1], initializer=tf.constant_initializer(regularization_rate), dtype=tf.float32, trainable=False)
		# 7.训练轮数
		tf.get_variable("training_steps", [1], initializer=tf.constant_initializer(training_steps), dtype=tf.int32, trainable=False)
		# 8.滑动平均衰减率
		tf.get_variable("moving_average_decay", [1], initializer=tf.constant_initializer(moving_average_decay), dtype=tf.float32, trainable=False)


''' 初始化隐层节点数量
	layer_count: 隐层的数量
	layer_tensor: 各层的节点数量
'''
def int_layers_variable(layer_count, layer_tensor):
	with tf.variable_scope('base_variable'):
		layer_count = tf.get_variable("layer_count", [1], initializer=tf.constant_initializer(layer_count))
		layer_tensor = tf.get_variable("layer_tensor", [layer_count], initializer=tf.constant_initializer(layer_tensor))

		# 检查入参的正确性
		if (layer_count < 1) :
			tf.assign(layer_count, 0);
		if (layer_count != tf.size(layer_tensor)):
			tf.assign(layer_count, 0);

# 初始化功能性参数
def init_func_variable(input_data_path):
	with tf.variable_scope('base_variable'):
		input_data_path = tf.get_variable("input_data_path", [1], initializer=tf.constant_initializer(input_data_path))



# 1.输入层节点数
def get_input_node():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("input_node", [1], dtype=tf.int32)[0]

# 2.输出层节点数
def get_output_node():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("output_node", [1], dtype=tf.int32)[0]

# 3.每次batch打包的样本个数
def get_batch_size():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("batch_size", [1], dtype=tf.int32)[0]

# 4.基础学习率
def get_learning_rate_base():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("learning_rate_base", [1])[0]

# 5.学习率的衰减率
def get_learning_rate_decay():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("learning_rate_decay", [1])[0]

# 6.描述模型复杂度的正则化项在损失函数中的系数
def get_regularization_rate():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("regularization_rate", [1])[0]

# 7.训练轮数
def get_training_steps():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("training_steps", [1], dtype=tf.int32)[0]

# 8.滑动平均衰减率
def get_moving_average_decay():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("moving_average_decay", [1])[0]

# 9.隐层的数量
def get_layer_count():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("layer_count", [1], dtype=tf.int32)[0]

# 10.各层的节点数量
def get_layer_tensor():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("layer_tensor", [get_layer_count()])

# 11.指定层的节点数
def get_gived_layer(layindex):
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("layer_tensor", [get_layer_count()], dtype=tf.int32)[layindex]

