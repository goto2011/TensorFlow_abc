#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf

# 处理平滑
def get_average_op(base_variable, global_step):
    with tf.name_scope("moving_average"):
        # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
        variable_averages = tf.train.ExponentialMovingAverage(
        	base_variable.get_moving_average_decay(), global_step)

        # 在所有代表神经网络的参数的变量上使用滑动平均，其他辅助变量就不需要了
        return variable_averages.apply(tf.trainable_variables())

