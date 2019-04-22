#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
import GeneralUtil.base_variable

'''配置神经网络的参数'''
LAYER1_NODE = 500  # 隐藏层神经元个数

# 获取 weights 变量，并把变量的正则化损失加入损失集合。
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# 定义神经网络的前向传播过程。
def inference(input_tensor, regularizer, input_node, output_node):
    # 第一层
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([input_node, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer = tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    #第二层
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, output_node], regularizer)
        biases = tf.get_variable("biases", [output_node], initializer = tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2