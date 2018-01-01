#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import b_mnist_inference as mnist_inference
import os

'''配置神经网络的参数'''
BATCH_SIZE = 100  # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8    #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 8000   #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "../model/"
MODEL_NAME = "02_mnist.ckpt"

# 保存log文件
LOG_SAVE_PATH = "../log/"


'''训练模型的过程'''
def train(mnist):
    # 输入数据
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')      #维度可以自动算出，也就是样本数
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    # 损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  #正则化损失函数

    # 计算前向传播结果
    y = mnist_inference.inference(x, regularizer)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)   # 将训练轮数的变量指定为不参与训练的参数

    # 处理平滑
    with tf.name_scope("moving_average"):
        #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        #在所有代表神经网络的参数的变量上使用滑动平均，其他辅助变量就不需要了
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 处理损失函数
    with tf.name_scope("loss_function"):
        # 计算交叉熵及其平均值。 这里tf.argmax(y_,1)表示在“行”这个维度上张量最大元素的索引号
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        #总损失函数=交叉熵损失和正则化损失的和
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 处理学习率、优化方法以及每一轮训练需要的操作。
    with tf.name_scope("train_step"):
        # 设置指数衰减的学习率。
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,     #基础学习率
            global_step,            #迭代轮数
            mnist.train.num_examples / BATCH_SIZE,  #过完所有训练数据需要的迭代次数
            LEARNING_RATE_DECAY,    #学习率衰减速率
            staircase=True)
        # 优化损失函数，用梯度下降法来优化
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # 反向传播更新参数和更新每一个参数的滑动平均值
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
        # 初始化tf持久化类
        saver = tf.train.Saver()

        # 开始训练过程。
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # 测试数据的验证过程放在另外一个独立程序中进行
            for i in range(TRAINING_STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if i % 1000 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

                    # 每 1000轮 把模型保存一次。
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

    writer = tf.summary.FileWriter(LOG_SAVE_PATH, tf.get_default_graph())
    writer.close()


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    print('===begin===')
    main()
    print('===end===')


