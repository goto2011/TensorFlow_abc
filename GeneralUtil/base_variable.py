#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf


# 初始化基本的神经网络参数
# 1. input_node, 输入层节点数
# 2. output_node, 输出层节点数
# 3. batch_size, 每次batch打包的样本个数
# 4. learning_rate_base, 基础学习learning_rate_base率
# 5. learning_rate_decay, 学习率的衰减率
# 6. regularization_rate, 描述模型复杂度的正则化项在损失函数中的系数
# 7. training_steps, 训练轮数
# 8. moving_average_decay, 滑动平均衰减率
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



# 打印 base variable
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


###############################################################################


# 初始化输入数据属性。针对图像数据。
# input_width: 宽
# input_height: 高
# input_depth: 深度
def init_input_variable(input_width, input_height, input_depth):
	with tf.variable_scope('base_variable'):
		# 1.宽
		tf.get_variable("input_width", [1], initializer=tf.constant_initializer(input_width), dtype=tf.int32, trainable=False)
		# 2.高
		tf.get_variable("input_height", [1], initializer=tf.constant_initializer(input_height), dtype=tf.int32, trainable=False)
		# 3.深度
		tf.get_variable("input_depth", [1], initializer=tf.constant_initializer(input_depth), dtype=tf.int32, trainable=False)

# 打印 input variable
def input_variable_dump(sess):
	with sess.as_default():
		print("input_width=%d" % (get_input_width().eval()))
		print("input_height=%d" % (get_input_height().eval()))
		print("input_depth=%d" % (get_input_depth().eval()))


# 1.输入数据之宽
def get_input_width():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("input_width", [1], dtype=tf.int32)[0]

# 2.输入数据之高
def get_input_height():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("input_height", [1], dtype=tf.int32)[0]

# 3.输入数据之深度
def get_input_depth():
	with tf.variable_scope('base_variable', reuse=True):
		return tf.get_variable("input_depth", [1], dtype=tf.int32)[0]



###############################################################################

# 定义全局变量，以保存隐层属性
g_layer_tensor = []

''' 初始化隐层属性
	layer_tensor: 各层的属性。其格式如下：
		type: 有4类：
			"conv": 卷积层； 
			"max-pool": 最大池化层; 
			"average-pool": 平均池化层； 
			"fc": 全连通层
		variable:
			卷积层是[kernel_length, kernel_depth]
			池化层是[kernel_length]
			全连通层是[output_length]
		other_variable:
			卷积层是["SAME"/"VALID", step]，默认值是["SAME", 1]
			池化层是["SAME"/"VALID", step]，默认值是["SAME", 1]
			全连通层是[activity_index, enable_dropout, dropout_rate]
				activity_index：表示使用何种激活函数。默认值0，表示不使用；
					1 表示 Relu；2 表示 Sigmoid；3 表示 tanh； 4 表示 softplus；5 表示 relu6。
				enable_dropout：表示是否使能 dropout，0不使能，1使能。默认值为0。
				dropout_rate: dropout 系数，为一(0, 1.0) 之间的浮点数。默认值为0。
'''
def init_layer_variable(layer_tensor):
	global g_layer_tensor
	with tf.variable_scope('base_variable'):
		layer_count = tf.get_variable("layer_count", [1], initializer=tf.constant_initializer(len(layer_tensor)))
		# layer_tensor = tf.get_variable("layer_tensor", [layer_count], initializer=tf.constant_initializer(layer_tensor))
		g_layer_tensor = layer_tensor

# 打印
def layer_variable_dump(sess):
	global g_layer_tensor
	print("layer_count=%d" % (get_layer_count()))
	for ii in range(get_layer_count()):
		print("layer_tensor[%d]" % ii, g_layer_tensor[ii])

# 1.隐层的数量
def get_layer_count():
	global g_layer_tensor
	with tf.variable_scope('base_variable', reuse=True):
		return len(g_layer_tensor)

# 2.指定层的节点数
def get_gived_layer(layer_index):
	global g_layer_tensor
	with tf.variable_scope('base_variable', reuse=True):
		return g_layer_tensor[layer_index]

###############################################################################
# 全局打印开关

g_debug_flag = False

def init_debug_flag(debug_flag):
	global g_debug_flag
	g_debug_flag = debug_flag

def get_debug_flag():
	global g_debug_flag
	return g_debug_flag





