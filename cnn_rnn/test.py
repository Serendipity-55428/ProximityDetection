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

#pb文件路径（在当前目录下生成）
pb_file_path = os.getcwd()

g1 = tf.Graph()
with g1.as_default():
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config= tf.ConfigProto(gpu_options= gpu_options), graph= g1) as sess:
    sess.run(init)
    #Loads the model from a SavedModel as specified by tags
    tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path+'savemodel1')

    #Returns the Tensor with the given name
    #名称都为'{name}:0'格式
    x = sess.graph.get_tensor_by_name('x:0')
    y = sess.graph.get_tensor_by_name('y:0')
    op = sess.graph.get_tensor_by_name('op_to_store:0')

    #测试代码
    ret = sess.run(op, feed_dict={x: 5, y: 5})
    print(ret)

g2 = tf.Graph()
with g2.as_default():
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.333)
with tf.Session(config=tf.ConfigProto(gpu_options= gpu_options), graph= g2) as sess:
    sess.run(init)
    tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path+'savemodel2')
    x_1 = sess.graph.get_tensor_by_name('x:0')
    y_1 = sess.graph.get_tensor_by_name('y:0')
    # w_1 = sess.graph.get_tensor_by_name('w_1:0')
    # b_1 = sess.graph.get_tensor_by_name('b_1:0')
    ops_x = sess.graph.get_tensor_by_name('ops_x:0')
    ops_s = sess.graph.get_tensor_by_name('ops_s:0')

    ret_1, ret_2 = sess.run([ops_x, ops_s], feed_dict= {x_1: np.arange(1, 7).reshape(2, 3), y_1: np.arange(2, 22).reshape(4, 5)})
    print(ret_1, ret_2)
    print(ret_1 + ret)




