#coding=utf-8
# 输入数据处理。
__author__ = 'duangan'


import tensorflow as tf
import numpy as np

# 将输入数据随机分配为训练数据集、测试数据集、验证数据集
# test_percent: 测试的数据百分比。推荐10%。
# valid_percent: 验证的数据百分比。推荐10%。
def random_alloc_train_set(data_list, test_percent, valid_percent):
	train_set = []
	test_set = []
	valid_set = []

	for data_item in data_list:
		# 随机讲数据分到训练数据集、测试集和验证集
		chance = np.random.randint(100)
		if chance < valid_percent:
			valid_set.append(data_item)
		elif chance < (test_percent + valid_percent):
			test_set.append(data_item)
		else:
			train_set.append(data_item)

	return train_set, test_set, valid_set