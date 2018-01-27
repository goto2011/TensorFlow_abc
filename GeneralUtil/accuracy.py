#coding=utf-8
__author__ = 'duangan'

import time
import tensorflow as tf
import inference
import base_variable

'''[summary]

[description]
	mnist: 输入数据源
	decay: 滑动平均衰减率
	model_path: model文件的路径
	time_interval: 统计间隔，单位为秒
'''
def evaluate(mnist, model_path, time_interval, input_node, output_node, moving_average_decay):
	with tf.Graph().as_default() as g:
		# 输入数据
		x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

		y = inference.inference(x, None, input_node, output_node)

		# 计算正确率
		accuracy = compute_accuracy(y, y_)

		#给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
		variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(model_path)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					# 通过文件名得到模型保存时迭代的轮次。
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

					# print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
					print("After %s training step(s), validation accuracy = %g " % (global_step, accuracy_score))
				else:
					print ('No checkpoint file found')
					return
				time.sleep(time_interval)


# 计算正确率
def compute_accuracy(predict_result, truth_result):
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(predict_result, 1), tf.argmax(truth_result, 1))
		return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





