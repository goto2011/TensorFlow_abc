#coding=utf-8
__author__ = 'duangan'


import tensorflow as tf
import numpy as np

import GeneralUtil.base_variable as variable

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512


# 获取上一层的输入数据的深度。
def get_input_depth(layer_index):
    if (layer_index == 0):
        # 返回输入
        return variable.get_input_depth()
    # 如果不是输入层，那么就要找最邻近的卷积层
    else:
        for ii in range(layer_index - 1, 0):
            layer_variable = variable.get_gived_layer(ii)
            if (layer_variable[0] == "conv"):
                return layer_variable[0][1];
    return -1


# 获取 other variable。
def get_other_variable(layer_variable):
    # 默认值
    if (len(layer_variable) == 2):
        return 'SAME', 1
    elif (len(layer_variable) == 3):
        return layer_variable[2][0], layer_variable[2][1]    


# 判断当前层是否为第一个全连通层。
def is_first_fc_layer(layer_index):
    layer_variable = variable.get_gived_layer(layer_index)
    if (layer_variable[0] != "fc"):
        return False
    for ii in range(layer_index - 1, 0):
        layer_variable = variable.get_gived_layer(ii)
        if (layer_variable[0] == "fc"):
            return False
    return True


# 使用变量化的元参数来生成神经网络结构。
def inference_ext(input_data, train, regularizer, layer_index):
    layer_variable = variable.get_gived_layer(layer_index)

    if (layer_variable[0] == "conv"):
        kernel_length = layer_variable[1][0]
        kernel_depth = layer_variable[1][1]
        input_depth = get_input_depth(layer_index)
        padding, step = get_other_variable(layer_variable)

        conv_weights = tf.get_variable(
            "weight", [kernel_length, kernel_length, input_depth, kernel_depth],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_biases = tf.get_variable("bias", [kernel_depth], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_data, conv_weights, strides=[1, step, step, 1], padding=padding)

        return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

    elif (layer_variable[0] == "max-pool"):
        kernel_length = layer_variable[1][0]
        padding, step = get_other_variable(layer_variable)

        return tf.nn.max_pool(input_data, ksize = [1,kernel_length,kernel_length,1], 
            strides=[1,step,step,1], padding=padding)        

    elif (layer_variable[0] == "fc"):
        # 如果是第一个全连通层，需要把上一层的输入数据拉直为向量。
        if (is_first_fc_layer(layer_index)):
            # 获取输入数据的维度
            pool_shape = input_data.get_shape().as_list()
            # 计算拉直后的向量的长度。
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            # 拉直。pool_shape[0]是一个 batch 中数据的个数。
            input_data = tf.reshape(input_data, [pool_shape[0], nodes])
        else:
            # 因为所有的全连通层都在一起，不是第一个时，直接取上一个。上一个应该是全连通层。
            nodes = variable.get_gived_layer(layer_index - 1)[1]

        current_nodes = layer_variable[1]

        fc_weights = tf.get_variable("weight", [nodes, current_nodes],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连通层需要加入正则化
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc_weights))
        fc_biases = tf.get_variable("bias", [current_nodes], initializer=tf.constant_initializer(0.1))



# 生成神经网络结构。
# 参数 train 表示当前是否在train过程中，以和测试过程分开。目的是使用 dropout 方法以防止过拟合。该方法不能用于测试过程。
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        # 32*32*1 ==》 5*5*1*32  ==》 28*28*32
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        # 28*28*32 ==> 2*2  ==> 14*14*32
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        # 14*14*32  ==》 5*5*32*64  ==》 14*14*64
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        # 14*14*64 ==> 2*2  ==> 7*7*64
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 为了准备第5层的数量，把矩阵拉直。 7*7*64 ==》 3136
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        # 3136  ==》 512
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        # tf.nn.relu是非线性激活函数
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        # 512  ==》 10
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit