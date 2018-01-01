#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import b_mnist_train as mnist_train
import GeneralUtil.accuracy as mnist_accuracy

MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率
MODEL_SAVE_PATH = "../model/"  # model文件的路径
EVAL_INTERVAL_SECS = 10  # 统计间隔，单位为秒

def main(argv=None):
	mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
	mnist_accuracy.evaluate(mnist, MOVING_AVERAGE_DECAY, MODEL_SAVE_PATH, EVAL_INTERVAL_SECS)

if __name__ == '__main__':
	tf.app.run()
