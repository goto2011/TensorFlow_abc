#coding=utf-8
# 图像预处理。

__author__ = 'duangan'

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import os.path
import GeneralUtil.base_variable as variable


# 模块级打印
DEBUG_FLAG = variable.get_debug_flag() or True
DEBUG_MODULE = "image_pretrim"
# 打印例子：
# if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)


# 读取图像文件的原始数据。可传入任何格式的图像，包括jpeg、png、gif及bmp。
# 返回：BMP, JPEG, and PNG时返回[height, width, num_channels]，
# 而gif返回[num_frames, height, width, 3]，且他们的dtype都为uint8，这会导致dtype为uint16的png格式图像失真
def read_image(image_file):
	image_raw_data = tf.gfile.FastGFile(image_file, "rb").read()

	_, image_type = os.path.splitext(image_file)

	if (image_type == ".jpg" or image_type == ".jpeg" or image_type == ".JPG" or image_type == ".JPEG"):
		img_data = tf.image.decode_jpeg(image_raw_data)
	elif (image_type ==".bmp" or image_type ==".BMP"):
		img_data = tf.image.decode_bmp(image_raw_data)
	elif (image_type ==".git" or image_type ==".GIT"):
		img_data = tf.image.decode_gif(image_raw_data)
	elif (image_type ==".png" or image_type ==".PNG"):
		img_data = tf.image.decode_png(image_raw_data)
	else:
		img_data = tf.image.decode_image(image_raw_data)

	# 转化数据类型为 float32 ，方便后续处理。
	img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

	print img_data.eval()

	return img_data


# 显示图片
def show_image(sess, img_data):
	with sess.as_default():
		if (img_data == []):
			return
		plt.imshow(img_data.eval())
		plt.show()

# 把图片数据保存到文件，采用 JPG 格式。
def save_to_jpg_file(sess, img_data, img_file):
	with sess.as_default():
		# Input 'image' of 'EncodeJpeg' Op has type float32 that does not match expected type of uint8.
		img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
		encode_image = tf.image.encode_jpeg(img_data)
		with tf.gfile.GFile(img_file, "wb") as f:
			f.write(encode_image.eval())

# 重新设置图像的长和宽(但图像内容尽量保留)。
# img_data: 图像数据。
# image_new_size: 新的图像大小，格式是 [300, 300]
# method: 指定图像大小调整算法。 0 表示双线性插值法，1 表示最近邻法， 2 表示双三次插值法， 3 表示面积插值法。各有千秋。
def image_resize(img_data, image_new_size, method):
	if (DEBUG_FLAG): print(DEBUG_MODULE, image_new_size, method)
	resized_image = tf.image.resize_images(img_data, image_new_size, method=method)
	print(resized_image.get_shape())
	return resized_image


# 对图像进行裁剪或填充。
# 如果指定的 size 小于原图像的大小，则是裁剪图像居中的部分；如果大于则在图像四周填充0.
# img_data: 图像数据。
# image_new_size: 新的图像大小，格式是 [300, 300]
def image_resize_with_crop_or_pad(img_data, image_new_size):
	resized_image = tf.image.resize_image_with_crop_or_pad(img_data, image_new_size[0], image_new_size[1])
	print(resized_image.get_shape())
	return resized_image


# 按比例裁剪图像的中间部分。
# img_data: 图像数据。
# percent: 裁剪比例，为一个(0, 1]的实数。
def image_resize_by_percent(image_data, percent):
	resized_image = tf.image.central_crop(image_data, percent)
	print(resized_image.get_shape())
	return resized_image

# 图像按坐标区域裁剪。
# img_data: 图像数据。
# left_top: 左上角坐标
# image_new_size: 截取区域大小（注意：不是右下角坐标）
def image_resize_by_zone(sess, image_data, left_top, image_new_size):
	# 检查给定的图像大小的正确性。
	with sess.as_default():
		ori_size = image_data.eval().shape
		print(ori_size)
		if ((left_top[0] + image_new_size[0] > ori_size[0]) or (left_top[1] + image_new_size[1] > ori_size[1])):
			return []

	resized_image = tf.image.crop_to_bounding_box(image_data, left_top[0], left_top[1], image_new_size[0], image_new_size[1])
	print(resized_image.get_shape())
	return resized_image

# 图像翻转
# method: 翻转方式：0-上下翻转; 1-左右翻转；2-对角线翻转
# is_random: 是否是随机翻转。 =0，不是； =1，是。
def image_turn(image_data, method, is_random = 0):
	if (method == 0):
		if (is_random == 0):
			new_image = tf.image.flip_up_down(image_data)
		else:
			new_image = tf.image.random_flip_up_down(image_data)
	elif (method == 1):
		if (is_random == 0):
			new_image = tf.image.flip_left_right(image_data)
		else:
			new_image = tf.image.random_flip_left_right(image_data)
	elif (method == 2):
		if (is_random == 0):
			new_image = tf.image.transpose_image(image_data)
		else:
			print(DEBUG_MODULE, "Error: image turn method error.")
			return []
	else:
		print(DEBUG_MODULE, "Error: image turn method error.")
		return []

	print(new_image.get_shape())
	return new_image


# 图像色彩调整
# method: 色彩调整方式：0-亮度；1-对比度；2-色相；3-饱和度。
# new_value： 调整的目标值。可能是单个数字，也可能是两个数组成的向量。
# is_random: 是否是随机调整。 =0，不是； =1，是。
def image_color_adjust(image_data, method, new_value, is_random=0):
	if (DEBUG_FLAG): print(DEBUG_MODULE, method, new_value, is_random)
	if (method == 0):
		if (is_random == 0):
			new_image = tf.image.adjust_brightness(image_data, new_value)
		else:
			new_value = abs(new_value)
			new_image = tf.image.random_brightness(image_data, new_value)
	elif (method == 1):
		if (is_random == 0):
			new_image = tf.image.adjust_contrast(image_data, new_value)
		else:
			if ((len(new_value) != 2) or (new_value[0] >= new_value[1])):
				print(DEBUG_MODULE, "Error: image color adjust param error.")
				return []
			new_image = tf.image.random_contrast(image_data, new_value[0], new_value[1])
	elif (method == 2):
		if (is_random == 0):
			new_image = tf.image.adjust_hue(image_data, new_value)
		else:
			if ((len(new_value) != 2) or (new_value[0] >= new_value[1])):
				print(DEBUG_MODULE, "Error: image color adjust param error.")
				return []
			new_image = tf.image.random_hue(image_data, new_value[0], new_value[1])
	elif (method == 3):
		if (is_random == 0):
			new_image = tf.image.adjust_saturation(image_data, new_value)
		else:
			if ((len(new_value) != 2) or (new_value[0] >= new_value[1])):
				print(DEBUG_MODULE, "Error: image color adjust param error.")
				return []
			new_image = tf.image.random_saturation(image_data, new_value[0], new_value[1])
	else:
		print(DEBUG_MODULE, "Error: image color adjust method error.")
		return []

	print(new_image.get_shape())
	return new_image

# 图像标准化，即把图像的数据均值变为0，方差变为1.
def image_cleanup(image_data):
	return tf.image.per_image_standardization(image_data)





