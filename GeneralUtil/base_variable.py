#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
import numpy as np


class base_variable(object):
	g_debug_flag = True

	# 初始化输入数据属性。针对图像数据。
	# input_width: 宽
	# input_height: 高
	# input_depth: 数据深度, 黑白为1, 彩色为3
	def __init__(self, input_width, input_height, input_depth):
		self.input_width = input_width
		self.input_height = input_height
		self.input_depth = input_depth

	# 打印 input variable
	def input_variable_dump(self):
		print("input_width=%d" % (get_input_width()))
		print("input_height=%d" % (get_input_height()))
		print("input_depth=%d" % (get_input_depth()))


	# 1.输入数据之宽
	def get_input_width(self):
		return self.input_width;

	# 2.输入数据之高
	def get_input_height(self):
		return self.input_height

	# 3.输入数据之深度
	def get_input_depth(self):
		return self.input_depth


###############################################################################
	# 初始化基本的神经网络参数
	# 1. input_node, 输入层节点数
	# 2. output_node, 输出层节点数
	# 3. batch_size, 每次batch打包的样本个数
	# 4. learning_rate_base, 基础学习learning_rate_base率
	# 5. learning_rate_decay, 学习率的衰减率
	# 6. regularization_rate, 描述模型复杂度的正则化项在损失函数中的系数
	# 7. training_steps, 训练轮数
	# 8. moving_average_decay, 滑动平均衰减率
	def init_base_variable(self, input_node, output_node, batch_size, 
			learning_rate_base, learning_rate_decay, regularization_rate, 
			training_steps, moving_average_decay):
		self.input_node = input_node
		self.output_node = output_node
		self.batch_size = batch_size
		self.learning_rate_base = learning_rate_base
		self.learning_rate_decay = learning_rate_decay
		self.regularization_rate = regularization_rate
		self.training_steps = training_steps
		self.moving_average_decay = moving_average_decay		

	# 打印 base variable
	def base_variable_dump(self):
		print("input_node=%d" % (get_input_node()))
		print("output_node=%d" % (get_output_node()))
		print("batch_size=%d" % (get_batch_size()))
		print("learning_rate_base=%f" % (get_learning_rate_base()))
		print("learning_rate_decay=%f" % (get_learning_rate_decay()))
		print("regularization_rate=%f" % (get_regularization_rate()))
		print("training_steps=%d" % (get_training_steps()))
		print("moving_average_decay=%f" % (get_moving_average_decay()))


	# 1.输入层节点数
	def get_input_node(self):
		return self.input_node

	# 2.输出层节点数
	def get_output_node(self):
		return self.output_node

	# 3.每次batch打包的样本个数
	def get_batch_size(self):
		return self.batch_size

	# 4.基础学习率
	def get_learning_rate_base(self):
		return self.learning_rate_base

	# 5.学习率的衰减率
	def get_learning_rate_decay(self):
		return self.learning_rate_decay

	# 6.描述模型复杂度的正则化项在损失函数中的系数
	def get_regularization_rate(self):
		return self.regularization_rate

	# 7.训练轮数
	def get_training_steps(self):
		return self.training_steps

	# 8.滑动平均衰减率
	def get_moving_average_decay(self):
		return self.moving_average_decay


	###############################################################################


	# 定义全局变量，以保存隐层属性
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
	def init_layer_variable(self, layer_tensor):
		self.ayer_tensor = layer_tensor

	# 打印
	def layer_variable_dump(self):
		print("layer_count=%d" % (get_layer_count()))
		for ii in range(get_layer_count()):
			print("layer_tensor[%d]" % ii, self.layer_tensor[ii])

	# 1.隐层的数量
	def get_layer_count(self):
		return len(self.layer_tensor)

	# 2.指定层的节点数
	def get_gived_layer(self, layer_index):
		return self.layer_tensor[layer_index]

	# 获取上一层的输入数据的深度。
	def get_previous_depth(self, layer_index):
	    if (layer_index == 1):
	        # 返回输入数据的深度
	        layer_variable = get_gived_layer(0)
	        return layer_variable[1][2]
	    # 如果不是输入层，那么就要找最邻近的卷积层
	    elif (layer_index > 1):
	        for ii in range(0, layer_index)[::-1]:
	            layer_variable = get_gived_layer(ii)
	            if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)
	            if (layer_variable[0] == "conv"):
	                return layer_variable[1][1];

	    print(DEBUG_MODULE, "error: layer index error")
	    return -1


	# 获取 other variable。
	def get_other_variable(self, layer_variable):
	    # 默认值
	    if (len(layer_variable) == 2):
	        return 'SAME', 1
	    elif (len(layer_variable) == 3):
	        return layer_variable[2][0], layer_variable[2][1]    


	# 判断当前层是否为第一个全连通层。
	def is_first_fc_layer(self, layer_index):
	    layer_variable = variable.get_gived_layer(layer_index)
	    if (layer_variable[0] != "fc"):
	        return False
	    for ii in range(0, layer_index)[::-1]:
	        layer_variable = get_gived_layer(ii)
	        if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)
	        if (layer_variable[0] == "fc"):
	            return False
	    return True


	# 使用变量化的元参数来生成神经网络结构。
	def inference_ext(self, input_data, train, regularizer, layer_index):
	    with tf.variable_scope('layer' + bytes(layer_index)):
	        layer_variable = get_gived_layer(layer_index)
	        print(DEBUG_MODULE, "***", layer_index, layer_variable)

	        if (layer_variable[0] == "conv"):
	            kernel_length = layer_variable[1][0]
	            kernel_depth = layer_variable[1][1]
	            input_depth = get_previous_depth(layer_index)
	            padding, step = get_other_variable(layer_variable)

	            if (DEBUG_FLAG): print(DEBUG_MODULE, kernel_length, kernel_depth, input_depth)

	            conv_weights = tf.get_variable("weight", [kernel_length, kernel_length, input_depth, kernel_depth], initializer=tf.truncated_normal_initializer(stddev=0.1))
	            conv_biases = tf.get_variable("bias", [kernel_depth], initializer=tf.constant_initializer(0.0))
	            conv = tf.nn.conv2d(input_data, conv_weights, strides=[1, step, step, 1], padding=padding)

	            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

	        elif (layer_variable[0] == "max-pool"):
	            kernel_length = layer_variable[1]
	            padding, step = get_other_variable(layer_variable)

	            return tf.nn.max_pool(input_data, ksize = [1,kernel_length,kernel_length,1], 
	                strides=[1,step,step,1], padding=padding)        

	        elif (layer_variable[0] == "fc"):
	            # 如果是第一个全连通层，需要把上一层的输入数据拉直为向量。
	            if (is_first_fc_layer(layer_index)):
	                # 获取输入数据的维度
	                pool_shape = input_data.get_shape().as_list()
	                if (DEBUG_FLAG): print(DEBUG_MODULE, pool_shape)
	                # 计算拉直后的向量的长度。
	                nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	                # 拉直。pool_shape[0]是一个 batch 中数据的个数。
	                input_data = tf.reshape(input_data, [pool_shape[0], nodes])
	            else:
	                # 如果不是第一个时，就直接取上一个。因为所有的全连通层都在一起，上一个也是全连通层。
	                nodes = variable.get_gived_layer(layer_index - 1)[1]

	            current_nodes = layer_variable[1]

	            fc_weights = tf.get_variable("weight", [nodes, current_nodes],
	                            initializer=tf.truncated_normal_initializer(stddev=0.1))
	            # 只有全连通层需要加入正则化
	            if regularizer != None: tf.add_to_collection('losses', regularizer(fc_weights))
	            fc_biases = tf.get_variable("bias", [current_nodes], initializer=tf.constant_initializer(0.1))
	            logit = tf.matmul(input_data, fc_weights) + fc_biases

	            if (len(layer_variable) == 3):
	                activity_index = layer_variable[2][0]
	                enable_dropout = layer_variable[2][1]
	                dropout_rate = layer_variable[2][2]

	                # 如果不是最后一个全连接层，则需要激活函数
	                if (activity_index == 1):
	                    logit = tf.nn.relu(logit)
	                elif (activity_index == 2):
	                    logit = tf.sigmoid(logit)
	                elif (activity_index == 3):
	                    logit = tf.tanh(logit)
	                elif (activity_index == 4):
	                    logit = tf.nn.softplus(logit)
	                elif (activity_index == 5):
	                    logit = tf.nn.relu6(logit)

	                if (train):
	                    if (enable_dropout == 1):
	                        logit = tf.nn.dropout(logit, dropout_rate)

	            return logit
	        else:
	            print(DEBUG_MODULE, "error: layer variable error")
	            return []


	###############################################################################
	# 全局打印开关
	@staticmethod
	def init_debug_flag(debug_flag):
		g_debug_flag = debug_flag

	@staticmethod
	def get_debug_flag():
		return g_debug_flag





