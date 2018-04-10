#coding=utf-8
__author__ = 'duangan'


import tensorflow as tf
import numpy as np

import GeneralUtil.base_variable as variable


# 模块级打印
DEBUG_FLAG = variable.get_debug_flag() or False
DEBUG_MODULE = "infernece_LeNet5"
# 打印例子：
# if (DEBUG_FLAG): print(DEBUG_MODULE, ii, layer_variable)


