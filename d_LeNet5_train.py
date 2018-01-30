#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import GeneralUtil.infernece_LeNet5 as inference
import GeneralUtil.average as average
import GeneralUtil.loss as loss
import GeneralUtil.learning_rate as learning_rate
import GeneralUtil.base_variable as variable


#数据源文件夹
INPUT_DATA_PATCH="./MNIST_data/"

'''训练模型的过程'''
def train_once(mnist):

    # 初始化输入数据参数
    variable.init_input_variable(28, 28, 1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        input_width = variable.get_input_width().eval()
        input_height = variable.get_input_height().eval()
        input_depth = variable.get_input_depth().eval()
        variable.input_variable_dump(sess)

    # 初始化 base variable
    # 1. input_node, 输入层节点数
    # 2. output_node, 输出层节点数
    # 3. batch_size, 每次batch打包的样本个数
    # 4. learning_rate_base, 基础学习learning_rate_base率
    # 5. learning_rate_decay, 学习率的衰减率
    # 6. regularization_rate, 描述模型复杂度的正则化项在损失函数中的系数
    # 7. training_steps, 训练轮数
    # 8. moving_average_decay, 滑动平均衰减率
    variable.init_base_variable(input_width*input_height*input_depth, 10, 100, 0.01, 0.99, 0.0001, 3001, 0.99)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        input_node = variable.get_input_node().eval()
        output_node = variable.get_output_node().eval()
        training_steps = variable.get_training_steps().eval()
        batch_size = variable.get_batch_size().eval()
        variable.base_variable_dump(sess)

    # 输入数据
    with tf.name_scope('input'):
        # 维度可以自动算出，也就是样本数
        x = tf.placeholder(tf.float32, [
            batch_size,         # 第一维度表示一个batch中样例的个数。
            input_width,  # 第二维和第三维表示图片的尺寸
            input_height,
            input_depth],  # 第四维表示图片的深度，黑白图片是1，RGB彩色是3.
            name='x-input')
        y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')

    # 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(variable.get_regularization_rate())

    # 初始化 layer variable
    variable.init_layer_variable([
        ["conv", [5, 32]], 
        ["max-pool", 2, ["SAME", 2]],
        ["conv", [5, 64]], 
        ["max-pool", 2, ["SAME", 2]],
        ["fc", 512, [1, 1, 0.5]],
        ["fc", 10]
        ])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        variable.layer_variable_dump(sess)

    # 计算前向传播结果
    l1_output = inference.inference_ext(x, False, regularizer, 0)
    l2_output = inference.inference_ext(l1_output, False, regularizer, 1)
    l3_output = inference.inference_ext(l2_output, False, regularizer, 2)
    l4_output = inference.inference_ext(l3_output, False, regularizer, 3)
    l5_output = inference.inference_ext(l4_output, False, regularizer, 4)
    y = inference.inference_ext(l5_output, False, regularizer, 5)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)   # 将训练轮数的变量指定为不参与训练的参数

    # 处理平滑
    variables_averages_op = average.get_average_op(global_step)

    # 处理损失函数
    my_loss = loss.get_total_loss(y, y_)

    # 处理学习率、优化方法等。
    train_op = learning_rate.get_train_op(global_step, mnist.train.num_examples, my_loss, variables_averages_op)

    # 开始训练过程。
    with tf.name_scope("train_step"):
        with tf.Session() as sess:
            import GeneralUtil.persist as persist
            saver, writer, run_metadata, run_options = persist.init("../model/02_mnist.ckpt", "../log/")
            tf.global_variables_initializer().run()

            # 测试数据的验证过程放在另外一个独立程序中进行
            for i in range(training_steps):
                xs, ys = mnist.train.next_batch(batch_size)
                reshaped_xs = np.reshape(xs,(
                    batch_size,         # 第一维度表示一个batch中样例的个数。
                    inference.IMAGE_SIZE,  # 第二维和第三维表示图片的尺寸
                    inference.IMAGE_SIZE,
                    inference.NUM_CHANNELS))  # 第四维表示图片的深度，黑白图片是1，RGB彩色是3.

                # 每1000轮做一次持久化
                if i % 1000 == 0:
                    # 将配置信息和记录运行的proto信息传入运行的线程，从而记录运行时每个节点的时间、空间开销。
                    _, loss_value, step = sess.run([train_op, my_loss, global_step], feed_dict={x: reshaped_xs, y_: ys}
                        , options=run_options, run_metadata=run_metadata)
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

                    persist.do(sess, writer, i, global_step)
                    saver.save(sess, "../model/02_mnist.ckpt", global_step=global_step)
                else:
                    _, loss_value, step = sess.run([train_op, my_loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            persist.close(writer)

def main(argv=None):
    mnist = input_data.read_data_sets(INPUT_DATA_PATCH, one_hot=True)
    train_once(mnist)


if __name__ == '__main__':
    print('===begin===')
    main()
    print('===end===')

