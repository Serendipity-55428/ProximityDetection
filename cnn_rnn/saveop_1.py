#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: saveop_1
@time: 2018/12/18 11:08
@desc:
'''
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()
x = tf.placeholder(dtype= tf.float32, shape= [2, 3], name= 'x')
y = tf.placeholder(dtype= tf.float32, shape= [4, 5], name= 'y')
w = tf.Variable(tf.truncated_normal(shape= [3, 4], mean= 0, stddev= 1.0), dtype= tf.float32, name= 'w')
b = tf.Variable(tf.truncated_normal(shape= [5], mean= 0, stddev= 1.0), dtype= tf.float32, name= 'b')
ops = tf.matmul(x, w, name= 'ops')
ops_1 = tf.matmul(ops, y, name= 'ops_2')
ops_x = tf.nn.bias_add(ops_1, b, name= 'ops_x')
ops_s = tf.nn.bias_add(ops_1, b, name= 'ops_s')

init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options= gpu_options)) as sess:
    sess.run(init)
    ops_x = sess.run(ops_x, feed_dict= {x: np.arange(6, dtype= np.float32).reshape(2, 3), y: np.arange(20, dtype= np.float32).reshape(4, 5)})

    print(ops_x)

    # Replaces all the variables in a graph with constants of the same values
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['ops_x', 'ops_s'])
    # 写入序列化的pb文件
    with tf.gfile.FastGFile(pb_file_path + 'model2.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # Builds the SavedModel protocol buffer and saves variables and assets
    # 在和project相同层级目录下产生带有savemodel名称的文件夹
    builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path + 'savemodel2')
    # Adds the current meta graph to the SavedModel and saves variables
    # 第二个参数为字符列表形式的tags – The set of tags with which to save the meta graph
    builder.add_meta_graph_and_variables(sess, ['cpu_server_1'])
    # Writes a SavedModel protocol buffer to disk
    # 此处p值为生成的文件夹路径
    p = builder.save()
    print(p)