#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
import base_variable


# 处理学习率
def get_train_op(global_step, train_number, loss, variables_averages_op):
	with tf.name_scope("learning_rate"):
		# 设置指数衰减的学习率
		learning_rate = tf.train.exponential_decay(
			base_variable.get_learning_rate_base(),     #基础学习率
			global_step,            #迭代轮数
			train_number / base_variable.get_batch_size(),  #过完所有训练数据需要的迭代次数
			base_variable.get_learning_rate_decay(),    #学习率衰减速率
			staircase=True)

		# 优化损失函数，用梯度下降法来优化
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

		# 反向传播更新参数和更新每一个参数的滑动平均值
		with tf.control_dependencies([train_step, variables_averages_op]):
			return tf.no_op(name='train')