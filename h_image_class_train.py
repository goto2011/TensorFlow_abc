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


# inception-v3 模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# inception-v3 模型中代表瓶颈层结果的张量名称。在训练模型时，可通过 tensor.name 来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的谷歌训练好的inception-v3模型文件目录
MODEL_DIR = '/Volumes/Data/TensorFlow/model/inception_dec_2015'
# 下载的谷歌训练好的inception-v3模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 保存训练数据通过瓶颈层后提取的特征向量。因为一个训练数据会被使用多次，所以将原始图像通过 inception-v3 模型计算出的特征向量存放在文件中，避免重复计算。
CACHE_DIR = '/Volumes/Data/TensorFlow/tmp/bottleneck'

# 图片数据的文件夹。其中每个子文件夹代表一个需要分类的类比，而且分类的名称就是文件夹名。
INPUT_DATA = '/Volumes/Data/TensorFlow/datasets/person_photo'
# INPUT_DATA = '/Volumes/Data/TensorFlow/datasets/flower_photos'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTACE = 10

# 定义神经网路的设置
LEARNING_RATE = 0.01
# STEPS = 4000
STEPS = 500
BATCH = 100


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
        file_list = file_system.get_files_by_ext(os.path.join(INPUT_DATA, dir_name), extensions)
        
        training_images, testing_images, validation_images = data_set.random_alloc_train_set(file_list, 
            testing_percentage, validation_percentage)

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


# 获取图片经过 inception-v3 模型处理后的特征向量的文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    # CACHE_DIR 特征向量的根地址
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


# 使用计算好的inception-v3模型处理一张照片，得到它的特征向量。
# image_data: 图片数据
# image_data_tensor: 
# bottleneck_tensor: 瓶颈张量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 将图片作为输入数据，注入瓶颈张量，得到该图片的特征向量。
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 通过瓶颈张量计算得来的是一个四维矩阵，需要压缩为特征向量（一维矩阵）。
    # np.squeeze(): 从数组的形状中删除单维条目，即把 shape 中为1的维度去掉
    return np.squeeze(bottleneck_values)


# 获取一张图片对应的特征向量的路径。它会先试图寻找已经计算并保存下来的特征向量，找不到再计算该特征向量，然后保存到文件中去。
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    sub_dir_path = os.path.join(CACHE_DIR, image_lists[label_name]['dir'])  # 到类别的文件夹
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)

    # 获取图片特征向量的路径
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # 特征向量文件不存在，则新建
    if not os.path.exists(bottleneck_path):
        # 获取图片原始路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 读取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 计算图片特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将特征向量存储到文件
        file_system.save_tensor_to_file(bottleneck_path, bottleneck_values)
    else:
        # 读取保存的特征向量文件
        bottleneck_values = file_system.read_tensor_from_file(bottleneck_path)
    return bottleneck_values


# 随机取一个batch的图片作为训练数据（特征向量，类别）
# sess:
# n_classes: 类别数量
# image_lists:
# how_many: 一个batch的数量
# category: 所属的数据集
# jpeg_data_tensor:
# bottleneck_tensor:
# return: 特征向量列表，类别列表
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_lists = sorted(list(image_lists.keys()))
    for _ in range(how_many):
        # 随机一个类别和图片编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        # 随机图片的类别名
        label_name = label_lists[label_index]
        # 随机图片的编号
        image_index = random.randrange(65536)
        # 计算此图片的特征向量
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        # 给y值赋值
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 获取全部的测试数据
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = sorted(list(image_lists.keys()))  # ['dandelion', 'daisy', 'sunflowers', 'roses', 'tulips']
    for label_index, label_name in enumerate(label_name_list):  # 枚举每个类别,如:0 sunflowers
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):  # 枚举此类别中的测试数据集中的每张图片
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            # 给y值赋值
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 读取所有图片，并分数据集
    image_lists = create_image_lists(TEST_PERCENTACE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    print(n_classes)

    # 初始化 base variable
    # 1. input_node, 输入层节点数
    # 2. output_node, 输出层节点数
    # 3. batch_size, 每次batch打包的样本个数
    # 4. learning_rate_base, 基础学习learning_rate_base率
    # 5. learning_rate_decay, 学习率的衰减率
    # 6. regularization_rate, 描述模型复杂度的正则化项在损失函数中的系数
    # 7. training_steps, 训练轮数
    # 8. moving_average_decay, 滑动平均衰减率
    variable.init_base_variable(2048, n_classes, 100, 0.01, 0.99, 0.0001, 500, 0.99)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        input_node = variable.get_input_node().eval()
        output_node = variable.get_output_node().eval()
        training_steps = variable.get_training_steps().eval()
        batch_size = variable.get_batch_size().eval()

        variable.base_variable_dump(sess)


    # 定义新的神经网络输入。这个输入是新的图片经过inception-v3 模型前向传播到瓶颈层时的节点取值。
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    # 定义新的y值
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义全连接层，用于解决新的图片分类问题。因为 inception-v3 模型已经把图片数据抽象为更加容易分类的特征向量了，所以不再需要训练复杂的神经网络来完成分类任务了。
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义损失函数
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=real, logits=forecast)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    # cross_entropy_mean = tf.reduce_mean(cross_entropy)
    cross_entropy_mean = loss.get_total_loss(logits, ground_truth_input, 1)
    # 优化
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    evaluation_step = accuracy.compute_accuracy(final_tensor, ground_truth_input)

    with tf.Session() as sess:
        # 初始化参数
        init = tf.global_variables_initializer()
        sess.run(init)

        # 训练开始
        for i in range(STEPS):
            # 每次随机获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            # 训练
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            # 验证
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % (
                    i, BATCH, validation_accuracy * 100))

        # 测试
        print("训练完成！")
        print("开始测试:")

        label_name_list = sorted(list(image_lists.keys()))  # ['dandelion', 'daisy', 'sunflowers', 'roses', 'tulips']
        print(label_name_list)
        for label_index, label_name in enumerate(label_name_list):  # 枚举每个类别,如:0 sunflowers
            for index, unused_base_name in enumerate(image_lists[label_name]['testing']):  # 枚举此类别中的测试数据集中的每张图片

                bottlenecks = []
                ground_truths = []

                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, index, 'testing', jpeg_data_tensor, bottleneck_tensor)
                ground_truth = np.zeros(n_classes, dtype=np.float32)
                # 给y值赋值
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)

                predict_tensor, test_accuracy = sess.run([final_tensor, evaluation_step], feed_dict={bottleneck_input: bottlenecks, ground_truth_input: ground_truths})

                predict_index = predict_tensor.argmax()
                predict_name = label_name_list[predict_index]
                print("---------------------------------")
                print(index, label_name, unused_base_name)
                print(predict_tensor)

                if (test_accuracy < 0.01):
                    print('This image verify as \"%s\", but is \"%s\"' % (predict_name, label_name))
                else:
                    print('This image verify as \"%s\", is OK!' % (predict_name))

        # 累计测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('总体测试准确率是 %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()