# -*- coding: utf-8 -*-
"""
@author: duangan

卷积神经网络 Inception-v3模型 迁移学习
"""
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import GeneralUtil.file_system as file_system
import GeneralUtil.data_set as data_set
import GeneralUtil.accuracy as accuracy
import GeneralUtil.loss as loss
import GeneralUtil.infernece_LeNet5 as inference
import GeneralUtil.average as average
import GeneralUtil.learning_rate as learning_rate
import GeneralUtil.base_variable as variable
import GeneralUtil.image_pretrim as image_pretrim


# 模块级打印
DEBUG_FLAG = variable.get_debug_flag() or True
DEBUG_MODULE = "h_image_class_train"
# 打印例子：
# if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)

# inception-v3 模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# 输入数据。其中每个子文件夹代表一个需要分类的类比，分类的名称就是文件夹名。
INPUT_DATA = '/Volumes/Data/TensorFlow/datasets/person_photo'
# INPUT_DATA = '/Volumes/Data/TensorFlow/datasets/flower_photos'

# 保存训练数据通过瓶颈层后提取的特征向量。因为一个训练数据会被使用多次，所以将原始图像通过 inception-v3
# 模型计算出的特征向量存放在文件中，避免重复计算。
CACHE_DIR = '/Volumes/Data/TensorFlow/tmp/person'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTACE = 10


# 这个函数把图片文件任意分成训练，验证，测试三部分
# testing_percentage: 测试的数据百分比，是10%
# validation_percentage: 验证的数据百分比，是10%
def create_image_lists(testing_percentage, validation_percentage):
    # 返回值
    result = {}

    # 获取目录下所有子目录
    sub_dirs = file_system.get_dirs_list(INPUT_DATA)

    # 遍历目录数组，每次处理一种
    for sub_dir in sub_dirs:
        dir_name = os.path.basename(sub_dir)

        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jepg', 'JPG', 'JPEG']
        file_list = file_system.get_files_by_ext(
            os.path.join(INPUT_DATA, dir_name), extensions)

        training_images, testing_images, validation_images = data_set.random_alloc_train_set(
        	file_list, testing_percentage, validation_percentage)

        result[dir_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }
    return result


# 这个函数通过类别名称、所属数据集和图片编号获取一张图片的地址
# image_lists: 所有图片信息
# image_dir: 根目录 （ 图片特征向量根目录 CACHE_DIR | 图片原始路径根目录 INPUT_DATA ）
# label_name: 类别的名称（ daisy|dandelion|roses|sunflowers|tulips ）
# index: 编号
# category: 所属的数据集（ training|testing|validation ）
# return: 指定图片文件的地址
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取给定类别的图片集合
    label_lists = image_lists[label_name]
    # 获取这种类别的图片中，特定的数据集(base_name的一维数组)
    category_list = label_lists[category]
    mod_index = index % len(category_list)  # 图片的编号 % 此数据集中图片数量
    # 获取图片文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 拼接地址
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 获取图片对应的预处理后的图片。
def get_bottleneck_path(image_lists, label_name, index, category):
    # CACHE_DIR 特征向量的根地址
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.jpg'


# 对照片进行预处理。
# image_data: 图片数据
def run_bottleneck_on_image(sess, image_data):
    # 1. 在降低图片分辨率之前，先将图片进行亮度、对比度等处理，将图片标准化。
    new_image = image_pretrim.image_cleanup(image_data)

    # 2. 降低图片分辨率到神经网络需要的状态。
    return image_pretrim.image_resize(new_image, [variable.get_input_width(),
                                                  variable.get_input_height()], 0)


# 获取一张图片对应的特征向量的路径。它会先试图寻找已经计算并保存下来的特征向量，找不到再计算该特征向量，然后保存到文件中去。
def get_or_create_bottleneck(sess, image_lists, label_name, index, category):
    sub_dir_path = os.path.join(CACHE_DIR, image_lists[
                                label_name]['dir'])  # 到类别的文件夹
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

    # 获取图片预处理的路径
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # 特征向量文件不存在，则新建
    if not os.path.exists(bottleneck_path):
        # 获取图片原始路径
        image_path = get_image_path(
            image_lists, INPUT_DATA, label_name, index, category)
        if (DEBUG_FLAG): print(DEBUG_MODULE, image_path)
        # 读取图片内容
        image_data = image_pretrim.read_image(image_path)
        # if (DEBUG_FLAG): print(DEBUG_MODULE, image_data.get_shape())
        # 对照片进行预处理
        bottleneck_values = run_bottleneck_on_image(sess, image_data)
        # 将特征向量存储到文件
        image_pretrim.save_to_jpg_file(sess, bottleneck_values, bottleneck_path)
    else:
        # 读取预处理文件
        bottleneck_values = image_pretrim.read_image(bottleneck_path)
    return bottleneck_values


# 随机取一个batch的图片作为训练数据（特征向量，类别）
# sess:
# n_classes: 类别数量
# image_lists:
# category: 所属的数据集
# return: 特征向量列表，类别列表
def get_random_cached_bottlenecks(sess, image_lists, category):
    bottlenecks = []
    ground_truths = []
    # 排序
    label_lists = sorted(list(image_lists.keys()))
    for _ in range(variable.get_batch_size()):
        # 随机一个类别和图片编号加入当前的训练数据
        label_index = random.randrange(variable.get_output_node())
        # 随机图片的类别名
        label_name = label_lists[label_index]
        # 随机图片的编号
        image_index = random.randrange(65536)

        # 给 x 值赋值
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category)
        bottlenecks.append(bottleneck)

        # 给 y_ 值赋值
        # ground_truth = tf.zeros(variable.get_output_node(), dtype=tf.float32)
        ground_truth = np.zeros(variable.get_output_node(), dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    # ('', array([ 0.16851467,  0.66669655,  0.32118168, ...,  0.07431143, 0.54442036,  0.59177148], dtype=float32), 3)
    # ('', <tf.Tensor 'Squeeze:0' shape=(299, 299, ?) dtype=float32>, 3)
    return bottlenecks, ground_truths


# 获取全部的测试数据
def get_test_bottlenecks(sess, image_lists):
    bottlenecks = []
    ground_truths = []
    # ['dandelion', 'daisy', 'sunflowers', 'roses', 'tulips']
    label_name_list = sorted(list(image_lists.keys()))
    # 枚举每个类别,如:0 sunflowers
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        # 枚举此类别中的测试数据集中的每张图片
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index, category)
            ground_truth = tf.zeros(
                variable.get_output_node(), dtype=tf.float32)
            # 给y值赋值
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 读取所有图片，并随机分配到各个数据集
    image_lists = create_image_lists(TEST_PERCENTACE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    print(n_classes)

    # 初始化输入数据参数
    # 1.输入数据之宽
    # 2.输入数据之高
    # 3.输入数据之深
    variable.init_input_variable(299, 299, 3)
    if (DEBUG_FLAG):
        variable.input_variable_dump()

    # 初始化 base variable
    # 1. input_node, 输入层节点数
    # 2. output_node, 输出层节点数
    # 3. batch_size, 每次batch打包的样本个数
    # 4. learning_rate_base, 基础学习learning_rate_base率
    # 5. learning_rate_decay, 学习率的衰减率
    # 6. regularization_rate, 描述模型复杂度的正则化项在损失函数中的系数
    # 7. training_steps, 训练轮数
    # 8. moving_average_decay, 滑动平均衰减率
    input_node = variable.get_input_width() * variable.get_input_height() * \
        variable.get_input_depth()
    variable.init_base_variable(
        input_node, n_classes, 100, 0.01, 0.99, 0.0001, 500, 0.99)
    if (DEBUG_FLAG):
        variable.base_variable_dump()

    # 初始化 layer variable
    variable.init_layer_variable([
        ["input", [variable.get_input_width(), variable.get_input_height(),
                   variable.get_input_depth()]],
        ["conv", [5, 32]],
        ["max-pool", 2, ["SAME", 2]],
        ["conv", [5, 64]],
        ["max-pool", 2, ["SAME", 2]],
        ["fc", 512, [1, 1, 0.5]],
        ["fc", variable.get_output_node()]
    ])
    if (DEBUG_FLAG):
        variable.layer_variable_dump()

    # 输入数据
    with tf.name_scope('input'):
        # 维度可以自动算出，也就是样本数
        x = tf.placeholder(tf.float32, [
            variable.get_batch_size(),     # 第一维度表示一个batch中样例的个数。
            variable.get_input_width(),    # 第二维和第三维表示图片的尺寸
            variable.get_input_height(),
            variable.get_input_depth()],   # 第四维表示图片的深度，黑白图片是1，RGB彩色是3.
            name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, variable.get_output_node()], name='y-input')

    # 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(
        variable.get_regularization_rate())

    # 计算前向传播结果
    l1_output = inference.inference_ext(x, False, regularizer, 1)
    l2_output = inference.inference_ext(l1_output, False, regularizer, 2)
    l3_output = inference.inference_ext(l2_output, False, regularizer, 3)
    l4_output = inference.inference_ext(l3_output, False, regularizer, 4)
    l5_output = inference.inference_ext(l4_output, False, regularizer, 5)
    y = inference.inference_ext(l5_output, False, regularizer, 6)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)   # 将训练轮数的变量指定为不参与训练的参数

    # 处理平滑
    variables_averages_op = average.get_average_op(global_step)

    # 处理损失函数
    my_loss = loss.get_total_loss(y, y_)

    # 处理学习率、优化方法等。
    train_step = learning_rate.get_train_op(
        global_step, variable.get_training_steps(), my_loss, variables_averages_op)

    # 计算正确率
    evaluation_step = accuracy.compute_accuracy(y, y_)

    # 训练开始
    with tf.Session() as sess:
        # 初始化参数
        init = tf.global_variables_initializer()
        sess.run(init)

        # 训练开始
        for i in range(variable.get_training_steps()):
            # 每次随机获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, image_lists, 'training')

            # ('', 100, 100)
            if (DEBUG_FLAG):print(DEBUG_MODULE, len(train_bottlenecks), len(train_ground_truth))
            # ('', <tf.Tensor 'Squeeze:0' shape=(299, 299, ?) dtype=float32>, 3)
            if (DEBUG_FLAG):print(DEBUG_MODULE, train_bottlenecks[0], len(train_ground_truth[0]))
            # ('', array([ 1.,  0.,  0.], dtype=float32), 3)
            if (DEBUG_FLAG):print(DEBUG_MODULE, train_ground_truth[0], len(train_ground_truth[0]))

            # 训练
            # setting an array element with a sequence.  / 用序列设置数组元素。
            sess.run(train_step, feed_dict={
                     x: train_bottlenecks, y_: train_ground_truth})

            # 验证
            if i % 100 == 0 or i + 1 == variable.get_training_steps():
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess,
                                                                                                image_lists, 'validation')
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                                               x: validation_bottlenecks, y_: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % (
                    i, variable.get_batch_size(), validation_accuracy * 100))

        # 测试开始
        print("训练完成！")
        print("开始测试:")

        # ['dandelion', 'daisy', 'sunflowers', 'roses', 'tulips']
        label_name_list = sorted(list(image_lists.keys()))
        print(label_name_list)
        # 枚举每个类别,如:0 sunflowers
        for label_index, label_name in enumerate(label_name_list):
            # 枚举此类别中的测试数据集中的每张图片
            for index, unused_base_name in enumerate(image_lists[label_name]['testing']):

                bottlenecks = []
                ground_truths = []

                # 给 x 赋值
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, index, 'testing')
                bottlenecks.append(bottleneck)

                # 给 y_ 赋值
                ground_truth = tf.zeros(
                    variable.get_output_node(), dtype=tf.float32)
                ground_truth[label_index] = 1.0
                ground_truths.append(ground_truth)

                predict_tensor, test_accuracy = sess.run(
                    [final_tensor, evaluation_step], feed_dict={x: bottlenecks, y_: ground_truths})

                predict_index = predict_tensor.argmax()
                predict_name = label_name_list[predict_index]
                print("---------------------------------")
                print(index, label_name, unused_base_name)
                print(predict_tensor)

                if (test_accuracy < 0.01):
                    print('This image verify as \"%s\", but is \"%s\"' %
                          (predict_name, label_name))
                else:
                    print('This image verify as \"%s\", is OK!' %
                          (predict_name))

        # 累计测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists)
        test_accuracy = sess.run(evaluation_step, feed_dict={
                                 x: test_bottlenecks, y_: test_ground_truth})
        print('总体测试准确率是 %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
