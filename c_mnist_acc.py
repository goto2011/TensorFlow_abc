#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import b_mnist_train as mnist_train
import GeneralUtil.accuracy as mnist_accuracy
import GeneralUtil.base_variable as mnist_variable

EVAL_INTERVAL_SECS = 10  # 统计间隔，单位为秒
MODEL_SAVE_PATH = "../model/"

#数据源文件夹
INPUT_DATA_PATCH="./MNIST_data/"

def main(argv=None):
    # 初始化 base variable
    mnist_variable.init_base_variable(784, 10, 100, 0.8, 0.99, 0.0001, 10000, 0.99)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        input_node = mnist_variable.get_input_node().eval()
        output_node = mnist_variable.get_output_node().eval()
        moving_average_decay = mnist_variable.get_moving_average_decay().eval()

	mnist = input_data.read_data_sets(INPUT_DATA_PATCH, one_hot=True)
	mnist_accuracy.evaluate(mnist, MODEL_SAVE_PATH, EVAL_INTERVAL_SECS, input_node, output_node, moving_average_decay)

if __name__ == '__main__':
	tf.app.run()
