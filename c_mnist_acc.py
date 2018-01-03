#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import b_mnist_train as mnist_train
import GeneralUtil.accuracy as mnist_accuracy

EVAL_INTERVAL_SECS = 10  # 统计间隔，单位为秒

def main(argv=None):
	mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
	mnist_accuracy.evaluate(mnist, mnist_train.MOVING_AVERAGE_DECAY, mnist_train.MODEL_SAVE_PATH, EVAL_INTERVAL_SECS)

if __name__ == '__main__':
	tf.app.run()
