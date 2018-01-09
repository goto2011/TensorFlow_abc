#coding=utf-8
__author__ = 'duangan'


import tensorflow as tf
import os
import os.path
import glob

# 获取指定目录的所有文件的全路径的列表。不包括子目录
def get_files_list(root_path):
	return get_files_list_by_ext(root_path, "*")

# 获取指定目录的指定扩展名的文件的全路径的列表。不包括子目录
def get_files_by_ext(root_path, extensions):
	file_list = []
	for extension in extensions:
		# 一次追加一类扩展名的多个文件。最终 file_list 中保存了图片文件的全路径的列表。
		file_glob = os.path.join(root_path, '*.' + extension)  # 将多个路径组合后返回
		file_list.extend(glob.glob(file_glob))
	return file_list