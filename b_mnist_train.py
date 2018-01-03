#coding=utf-8
__author__ = 'duangan'

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

import GeneralUtil.inference as mnist_inference
import GeneralUtil.average as mnist_average
import GeneralUtil.loss as minist_loss
import GeneralUtil.learning_rate as mnist_learning_rate
import GeneralUtil.base_variable as mnist_variable


#数据源文件夹
INPUT_DATA_PATCH="./MNIST_data/"

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "../model/"
MODEL_NAME = "02_mnist.ckpt"

# 保存log文件
LOG_SAVE_PATH = "../log/"


'''训练模型的过程'''
def train_once(mnist):
    # 初始化 base variable
    mnist_variable.init_base_variable(784, 10, 100, 0.8, 0.99, 0.0001, 10000, 0.99)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        input_node = mnist_variable.get_input_node().eval()
        output_node = mnist_variable.get_output_node().eval()
        training_steps = mnist_variable.get_training_steps().eval()
        batch_size = mnist_variable.get_batch_size().eval()

        mnist_variable.base_variable_dump(sess)

    # 输入数据
    with tf.name_scope('input'):
        # 维度可以自动算出，也就是样本数
        x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')

    # 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(mnist_variable.get_regularization_rate())

    # 计算前向传播结果
    y = mnist_inference.inference(x, regularizer, input_node, output_node)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)   # 将训练轮数的变量指定为不参与训练的参数

    # 处理平滑
    variables_averages_op = mnist_average.get_average_op(global_step)

    # 处理损失函数
    loss = minist_loss.get_total_loss(y, y_)

    # 处理学习率、优化方法等。
    train_op = mnist_learning_rate.get_train_op(global_step, mnist.train.num_examples, loss, variables_averages_op)

    # 训练
    with tf.name_scope("train_step"):
        # 初始化tf持久化类
        saver = tf.train.Saver()

        # 开始训练过程。
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_SAVE_PATH, tf.get_default_graph())
            tf.global_variables_initializer().run()

            # 测试数据的验证过程放在另外一个独立程序中进行
            for i in range(training_steps):
                xs, ys = mnist.train.next_batch(batch_size)

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

