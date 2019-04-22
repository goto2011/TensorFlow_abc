#coding=utf-8
# 文件系统相关
__author__ = 'duangan'


import tensorflow as tf
import os
import os.path
import glob
from pathlib import Path


# 获取指定目录的目录列。包括子目录。按字母排序。
def get_dirs_list(root_path):
	dir_list = []
	# 判断目录是否存在。
	if (os.path.exists(root_path) == False):
		return dir_list

	for (root, dirs, files) in os.walk(root_path):
		if (dirs != []):
			dir_list.extend(dirs)

	return sorted(dir_list)



# 获取指定目录的所有文件的列表。不包括子目录
def get_files_list(root_path):
	return get_files_list_by_ext(root_path, "*")


# 获取指定目录的指定扩展名的文件列表。不包括子目录
def get_files_by_ext(root_path, extensions):
	file_list = []
	file_list_no_path = []

	for extension in extensions:
		# file_list 中保存了图片文件的全路径的列表。
		file_glob = os.path.join(root_path, '*.' + extension)  # 将多个路径组合后返回
		file_list.extend(glob.glob(file_glob))

	for file_name in file_list:
		file_list_no_path.append(os.path.basename(file_name))
	return file_list_no_path



# 保存张量数据到文件系统
def save_tensor_to_file(file_name, tesor):
	# 不存在新建。存在则什么也不做。
	if not os.path.exists(file_name):
		saved_string = ','.join(str(x) for x in tesor)
		with open(file_name, 'w') as save_file:
			save_file.write(saved_string)

# 读取文件中的张量数据
def read_tensor_from_file(file_name):
	data_folder = Path(file_name)
	with open(data_folder, 'r') as read_file:
		read_string = read_file.read()

	# 字符串转float数组
	return [float(x) for x in read_string.split(',')]

