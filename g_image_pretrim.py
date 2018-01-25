# -*- coding: utf-8 -*-
"""
@author: duangan

测试照片预处理.
"""

import tensorflow as tf
import GeneralUtil.image_pretrim as image


def main(_):
	with tf.Session() as sess:
		image.show_image(sess, image.read_image("/Volumes/Data/TensorFlow/datasets/cat.jpg"))

if __name__ == '__main__':
    tf.app.run()