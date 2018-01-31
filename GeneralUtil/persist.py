#coding=utf-8
__author__ = 'duangan'


import tensorflow as tf
import os


# 初始化tf持久化类
saver = tf.train.Saver()

# 运行时记录信息的proto
run_metadata = tf.RunMetadata()

def init(model_path, log_path):
	global saver
	global run_metadata
	with tf.variable_scope('persist'):
		tf.get_variable("model_path", [1], initializer=tf.constant_initializer(model_path), dtype=tf.string, trainable=False)
		# tf.get_variable("model_file", [1], initializer=tf.constant_initializer(model_file), dtype=tf.string, trainable=False)
		tf.get_variable("log_path", [1], initializer=tf.constant_initializer(log_path), dtype=tf.string, trainable=False)

		# 配置运行时需要记录的信息
		run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
		writer = tf.summary.FileWriter(log_path, tf.get_default_graph())

		return saver, writer, run_metadata, run_options

# 完成持久化
def do(sess, writer, current_step, global_step):
	global saver
	global run_metadata
	with tf.variable_scope('persist', reuse=True):
		# 记录节点的时间和空间开销
		writer.add_run_metadata(run_metadata, "step%03d" % current_step)
		# writer.add_summary(step, i)

		model_path = tf.get_variable("model_path", [1], dtype=tf.string)[0]
		# model_file = tf.get_variable("model_file", [1], dtype=tf.string)[0]
		# with sess.as_default():
			# print("model_path=%s" % model_path.eval())
			# print("model_file=%s" % model_file.eval())
		# saver.save(sess, os.path.join(model_path, model_file), global_step=global_step)
		# saver.save(sess, model_path, global_step=global_step)

def close(writer):
	with tf.variable_scope('persist', reuse=True):
		writer.close()




