#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

import GeneralUtil.inference_mnist as mnist_inference
import GeneralUtil.average as mnist_average
import GeneralUtil.loss as minist_loss
import GeneralUtil.learning_rate as mnist_learning_rate
import GeneralUtil.base_variable as variable

# 模块级打印
DEBUG_FLAG = variable.base_variable.get_debug_flag() or True
DEBUG_MODULE = "c_mnist_train"

#数据源文件夹
INPUT_DATA_PATCH="./MNIST_data/"

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "../model/"
MODEL_NAME = "c_mnist.ckpt"

# 保存log文件
LOG_SAVE_PATH = "../log/"


'''训练模型的过程'''
def train_once(mnist):
    # 1. 输入数据之宽
    # 2. 输入数据之高
    # 3. 输入数据之深
    # 4. 训练集的样本数量
    # 5. 每次batch打包的样本个数
    # 6. 计划的训练轮数
    my_var = variable.base_variable(28, 28, 1, mnist.train.num_examples, 100, 10000)
    if (DEBUG_FLAG):
        my_var.input_variable_dump()
    # 1. 输入层节点数
    # 2. 输出层节点数.
    # 3. 基础学习率. 0.8 是不是太高了, 之前是 0.01
    # 4. 学习率的衰减率
    # 5. 描述模型复杂度的正则化项在损失函数中的系数
    # 6. 滑动平均衰减率
    my_var.init_base_variable(
        my_var.get_input_node_count(), 10, 0.1, 0.99, 0.0001, 0.99)
    if (DEBUG_FLAG):
        my_var.base_variable_dump()

    # 输入数据
    with tf.name_scope('input'):
        # 维度可以自动算出，也就是样本数
        x = tf.placeholder(tf.float32, [None, my_var.get_input_node()], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, my_var.get_output_node()], name='y-input')

    # 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(my_var.get_regularization_rate())

    # 计算前向传播结果
    y = mnist_inference.inference(x, regularizer, my_var.get_input_node(), my_var.get_output_node())

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)   # 将训练轮数的变量指定为不参与训练的参数

    # 处理平滑
    variables_averages_op = mnist_average.get_average_op(my_var, global_step)

    # 处理损失函数
    loss = minist_loss.get_total_loss(y, y_)

    # 处理学习率、优化方法等。
    train_op = mnist_learning_rate.get_train_op(my_var, global_step, loss, variables_averages_op)

    # 训练
    with tf.name_scope("train_step"):
        # 初始化tf持久化类
        saver = tf.train.Saver()

        # 开始训练过程。
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # 持久化
            writer = tf.summary.FileWriter(LOG_SAVE_PATH, tf.get_default_graph())
            tf.global_variables_initializer().run()

            # 测试数据的验证过程放在另外一个独立程序中进行
            for i in range(my_var.get_training_steps()):
                xs, ys = mnist.train.next_batch(my_var.get_train_batch_size())

                if i % 1000 == 0:
                    # 配置运行时需要记录的信息
                    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                    # 运行时记录信息的proto
                    run_metadata = tf.RunMetadata()
                    # 将配置信息和记录运行的proto信息传入运行的线程，从而记录运行时每个节点的时间、空间开销。
                    _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys}
                        , options=run_options, run_metadata=run_metadata)
                    # 记录节点的时间和空间开销
                    writer.add_run_metadata(run_metadata, "step%03d" % i)
                    # writer.add_summary(step, i)

                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

                    # 每 1000轮 把模型保存一次。
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                else:
                    _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
    
    writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets(INPUT_DATA_PATCH, one_hot=True)
    train_once(mnist)


if __name__ == '__main__':
    print('===begin===')
    main()
    print('===end===')

