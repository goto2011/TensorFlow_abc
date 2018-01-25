#coding=utf-8
# 图像预处理。

__author__ = 'duangan'

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import os.path


# 读取图像文件的原始数据。可传入任何格式的图像，包括jpeg、png、gif及bmp。
# 返回：BMP, JPEG, and PNG时返回[height, width, num_channels]，
# 而gif返回[num_frames, height, width, 3]，且他们的dtype都为uint8，这会导致dtype为uint16的png格式图像失真
def read_image(image_file):
	image_raw_data = tf.gfile.FastGFile(image_file, "rb").read()

	_, image_type = os.path.splitext(image_file)

	if (image_type == ".jpg" or image_type == ".jpeg"):
		img_data = tf.image.decode_jpeg(image_raw_data)
	elif (image_type ==".bmp"):
		img_data = tf.image.decode_bmp(image_raw_data)
	elif (image_type ==".git"):
		img_data = tf.image.decode_gif(image_raw_data)
	elif (image_type ==".png"):
		img_data = tf.image.decode_png(image_raw_data)
	else:
		img_data = tf.image.decode_image(image_raw_data)

	# 转化数据类型为 float32，方便后续处理。
	img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

	return img_data


# 显示图片
def show_image(sess, img_data):
	with sess.as_default():
		plt.imshow(img_data.eval())
		plt.show()

