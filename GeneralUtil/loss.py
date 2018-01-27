#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf

# 计算总损失函数
# forecast: 预测值。
# real: 实际值。
# method: 指定计算方法。
#  =0：表示 sparse_softmax_cross_entropy_with_logits()
#  =1：表示 softmax_cross_entropy_with_logits()
#  =2：表示 sigmoid_cross_entropy_with_logits()
#  =3：表示 weighted_cross_entropy_with_logits()
def get_total_loss(forecast, real, method=0):
	with tf.name_scope("loss_function"):
		# 计算交叉熵及其平均值
		if (method == 0):
			# 这里tf.argmax(y_,1)表示在“行”这个维度上张量最大元素的索引号
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real, 1), logits=forecast)
			cross_entropy_mean = tf.reduce_mean(cross_entropy)
			#总损失函数=交叉熵损失和正则化损失的和
			return cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
		elif (method == 1):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=real, logits=forecast)
		elif (method == 2):
			cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.argmax(real, 1), logits=forecast)
		else:
			cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=tf.argmax(real, 1), logits=forecast)

		return tf.reduce_mean(cross_entropy)
