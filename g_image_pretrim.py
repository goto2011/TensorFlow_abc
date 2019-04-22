# -*- coding: utf-8 -*-
"""
@author: duangan

测试照片预处理.
"""

import tensorflow as tf
import GeneralUtil.image_pretrim as image


def main(_):
	with tf.Session() as sess:
		image_data = image.read_image("../datasets/cat.jpg")
		print("***", image_data.get_shape())
		new_image = image.image_cleanup(image_data)
		print("***", new_image.get_shape())
		new_image2 = image.image_resize(new_image, [299, 299], 0)
		print("***", new_image2.get_shape())

		image.show_image(sess, new_image2)

if __name__ == '__main__':
    tf.app.run()