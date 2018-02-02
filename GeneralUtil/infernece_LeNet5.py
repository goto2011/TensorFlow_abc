#coding=utf-8
__author__ = 'duangan'


import tensorflow as tf
import numpy as np

import GeneralUtil.base_variable as variable


# 模块级打印
DEBUG_FLAG = variable.get_debug_flag() or False
DEBUG_MODULE = "infernece_LeNet5"
# 打印例子：
# if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)

# 获取上一层的输入数据的深度。
def get_previous_depth(layer_index):
    if (layer_index == 1):
        # 返回输入数据的深度
        layer_variable = variable.get_gived_layer(0)
        return layer_variable[1][2]
    # 如果不是输入层，那么就要找最邻近的卷积层
    elif (layer_index > 1):
        for ii in range(0, layer_index)[::-1]:
            layer_variable = variable.get_gived_layer(ii)
            if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)
            if (layer_variable[0] == "conv"):
                return layer_variable[1][1];

    print(DEBUG_MODULE, "error: layer index error")
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
    for ii in range(0, layer_index)[::-1]:
        layer_variable = variable.get_gived_layer(ii)
        if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)
        if (layer_variable[0] == "fc"):
            return False
    return True


# 使用变量化的元参数来生成神经网络结构。
def inference_ext(input_data, train, regularizer, layer_index):
    with tf.variable_scope('layer' + bytes(layer_index)):
        layer_variable = variable.get_gived_layer(layer_index)
        print(DEBUG_MODULE, "***", layer_index, layer_variable)

        if (layer_variable[0] == "conv"):
            kernel_length = layer_variable[1][0]
            kernel_depth = layer_variable[1][1]
            input_depth = get_previous_depth(layer_index)
            padding, step = get_other_variable(layer_variable)

            if (DEBUG_FLAG): print(DEBUG_MODULE, kernel_length, kernel_depth, input_depth)

            conv_weights = tf.get_variable("weight", [kernel_length, kernel_length, input_depth, kernel_depth], initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv_biases = tf.get_variable("bias", [kernel_depth], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input_data, conv_weights, strides=[1, step, step, 1], padding=padding)

            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

        elif (layer_variable[0] == "max-pool"):
            kernel_length = layer_variable[1]
            padding, step = get_other_variable(layer_variable)

            return tf.nn.max_pool(input_data, ksize = [1,kernel_length,kernel_length,1], 
                strides=[1,step,step,1], padding=padding)        

        elif (layer_variable[0] == "fc"):
            # 如果是第一个全连通层，需要把上一层的输入数据拉直为向量。
            if (is_first_fc_layer(layer_index)):
                # 获取输入数据的维度
                pool_shape = input_data.get_shape().as_list()
                if (DEBUG_FLAG): print(DEBUG_MODULE, pool_shape)
                # 计算拉直后的向量的长度。
                nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
                # 拉直。pool_shape[0]是一个 batch 中数据的个数。
                input_data = tf.reshape(input_data, [pool_shape[0], nodes])
            else:
                # 如果不是第一个时，就直接取上一个。因为所有的全连通层都在一起，上一个也是全连通层。
                nodes = variable.get_gived_layer(layer_index - 1)[1]

            current_nodes = layer_variable[1]

            fc_weights = tf.get_variable("weight", [nodes, current_nodes],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            # 只有全连通层需要加入正则化
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc_weights))
            fc_biases = tf.get_variable("bias", [current_nodes], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(input_data, fc_weights) + fc_biases

            if (len(layer_variable) == 3):
                activity_index = layer_variable[2][0]
                enable_dropout = layer_variable[2][1]
                dropout_rate = layer_variable[2][2]

                # 如果不是最后一个全连接层，则需要激活函数
                if (activity_index == 1):
                    logit = tf.nn.relu(logit)
                elif (activity_index == 2):
                    logit = tf.sigmoid(logit)
                elif (activity_index == 3):
                    logit = tf.tanh(logit)
                elif (activity_index == 4):
                    logit = tf.nn.softplus(logit)
                elif (activity_index == 5):
                    logit = tf.nn.relu6(logit)

                if (train):
                    if (enable_dropout == 1):
                        logit = tf.nn.dropout(logit, dropout_rate)

            return logit
        else:
            print(DEBUG_MODULE, "error: layer variable error")
            return []

