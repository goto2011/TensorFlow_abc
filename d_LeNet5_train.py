#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import GeneralUtil.average as average
import GeneralUtil.loss as loss
import GeneralUtil.learning_rate as learning_rate
import GeneralUtil.base_variable as variable
import GeneralUtil.accuracy as accuracy

# 模块级打印
DEBUG_FLAG = base_variable.get_debug_flag() or True
DEBUG_MODULE = "d_LeNet5_train"
# 打印例子：
# if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)

#数据源文件夹
INPUT_DATA_PATCH="./MNIST_data/"

'''训练模型的过程'''
def train_once(mnist):

    # 初始化输入数据参数
    # 1.输入数据之宽
    # 2.输入数据之高
    # 3.输入数据之深
    my_var = variable.init_input_variable(28, 28, 1)
    if (DEBUG_FLAG): my_var.input_variable_dump()

    # 初始化 base variable
    # 1. input_node, 输入层节点数
    # 2. output_node, 输出层节点数
    # 3. batch_size, 每次batch打包的样本个数
    # 4. learning_rate_base, 基础学习learning_rate_base率
    # 5. learning_rate_decay, 学习率的衰减率
    # 6. regularization_rate, 描述模型复杂度的正则化项在损失函数中的系数
    # 7. training_steps, 训练轮数
    # 8. moving_average_decay, 滑动平均衰减率
    input_node = my_var.get_input_width() * my_var.get_input_height() * my_var.get_input_depth()
    my_var.init_base_variable(input_node, 10, 100, 0.01, 0.99, 0.0001, 3001, 0.99)
    if (DEBUG_FLAG): my_var.base_variable_dump()

    # 初始化 layer variable
    my_var.init_layer_variable([
        ["input", [my_var.get_input_width(), my_var.get_input_height(), my_var.get_input_depth()]],
        ["conv", [5, 32]], 
        ["max-pool", 2, ["SAME", 2]],
        ["conv", [5, 64]], 
        ["max-pool", 2, ["SAME", 2]],
        ["fc", 512, [1, 1, 0.5]],
        ["fc", 10]
        ])
    if (DEBUG_FLAG): my_var.layer_variable_dump()

    # 输入数据
    with tf.name_scope('input'):
        # 维度可以自动算出，也就是样本数
        x = tf.placeholder(tf.float32, [
            my_var.get_batch_size(),     # 第一维度表示一个batch中样例的个数。
            my_var.get_input_width(),    # 第二维和第三维表示图片的尺寸
            my_var.get_input_height(),
            my_var.get_input_depth()],   # 第四维表示图片的深度，黑白图片是1，RGB彩色是3.
            name='x-input')
        y_ = tf.placeholder(tf.float32, [None, my_var.get_output_node()], name='y-input')

    # 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(my_var.get_regularization_rate())

    # 计算前向传播结果
    l1_output = my_var.inference_ext(x, False, regularizer, 1)
    l2_output = my_var.inference_ext(l1_output, False, regularizer, 2)
    l3_output = my_var.inference_ext(l2_output, False, regularizer, 3)
    l4_output = my_var.inference_ext(l3_output, False, regularizer, 4)
    l5_output = my_var.inference_ext(l4_output, False, regularizer, 5)
    y = my_var.inference_ext(l5_output, False, regularizer, 6)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)   # 将训练轮数的变量指定为不参与训练的参数

    # 处理平滑
    variables_averages_op = average.get_average_op(global_step)

    # 处理损失函数
    my_loss = loss.get_total_loss(y, y_)

    # 处理学习率、优化方法等。
    train_op = learning_rate.get_train_op(global_step, mnist.train.num_examples, my_loss, variables_averages_op)

    # 计算正确率
    evaluation_step = accuracy.compute_accuracy(y, y_)

    # 开始训练过程。
    with tf.name_scope("train_step"):
        with tf.Session() as sess:
            import GeneralUtil.persist as persist
            saver, writer, run_metadata, run_options = persist.init("../model/02_mnist.ckpt", "../log/")
            tf.global_variables_initializer().run()

            # 测试数据的验证过程放在另外一个独立程序中进行
            for i in range(variable.get_training_steps()):
                # 读取数据
                xs, ys = mnist.train.next_batch(variable.get_batch_size())
                reshaped_xs = np.reshape(xs,(
                    variable.get_batch_size(),      # 第一维度表示一个batch中样例的个数。
                    variable.get_input_width(),     # 第二维和第三维表示图片的尺寸
                    variable.get_input_height(),
                    variable.get_input_depth()))    # 第四维表示图片的深度，黑白图片是1，RGB彩色是3.

                # 每1000轮做一次持久化
                if i % 1000 == 0:
                    # 将配置信息和记录运行的proto信息传入运行的线程，从而记录运行时每个节点的时间、空间开销。
                    _, loss_value, step, acc = sess.run([train_op, my_loss, global_step, evaluation_step], feed_dict={x: reshaped_xs, y_: ys}
                        , options=run_options, run_metadata=run_metadata)
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    accuracy.print_accuracy(acc)

                    persist.do(sess, writer, i, global_step)
                    saver.save(sess, "../model/02_mnist.ckpt", global_step=global_step)
                else:
                    _, loss_value, step, acc = sess.run([train_op, my_loss, global_step, evaluation_step], feed_dict={x: reshaped_xs, y_: ys})

            persist.close(writer)



def main(argv=None):
    mnist = input_data.read_data_sets(INPUT_DATA_PATCH, one_hot=True)
    train_once(mnist)

if __name__ == '__main__':
    print('===begin===')
    main()
    print('===end===')

