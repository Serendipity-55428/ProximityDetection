#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: test
@time: 2018/11/14 21:36
@desc:
'''
import numpy as np
import tensorflow as tf
import os

pb_file_path = os.getcwd()

with tf.Session() as sess:
    #Loads the model from a SavedModel as specified by tags
    tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path+'savemodel')

    #Returns the Tensor with the given name
    #名称都为'{name}:0'格式
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')
    op = sess.graph.get_tensor_by_name('op_to_store:0')

    #测试代码
    ret = sess.run(op, feed_dict={input_x: 5, input_y: 5})
    print(ret)
