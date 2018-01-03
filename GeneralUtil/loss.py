#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf

# 总损失函数
def get_total_loss(forecast, real):
    with tf.name_scope("loss_function"):
        # 计算交叉熵及其平均值。 这里tf.argmax(y_,1)表示在“行”这个维度上张量最大元素的索引号
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real, 1), logits=forecast)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        
        #总损失函数=交叉熵损失和正则化损失的和
        return cross_entropy_mean + tf.add_n(tf.get_collection('losses'))