#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: saveop
@time: 2018/11/16 22:13
@desc:
'''

import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

def fun(x, y, name):
    with tf.name_scope('opsss'):
        with tf.name_scope('sec'):
            x = tf.Variable(x, name='x')
            y = tf.Variable(y, name='y')
            b = tf.Variable(1, name='b')
            xy = x * y
            op_1 = tf.add(xy, b, name='op_to_store_1')
            op_2 = tf.add(op_1, tf.Variable(3, name='b_2'), name=name)


    return op_2

op_a = fun(2, 3, 'op_1')
op_b = fun(4, 5, 'op_2')
init = tf.global_variables_initializer()

print(op_a.op.name, op_b.op.name)

with tf.Session() as sess:
    sess.run(init)

    #Replaces all the variables in a graph with constants of the same values
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['{name}'.format(name=op_a.op.name), '{name}'.format(name=op_b.op.name)])

    #测试代码
    # a = 10
    # b = 3
    op_out = sess.run(op_a)
    op_out_1 = sess.run(op_b)
    print(op_out, op_out_1)

    #写入序列化的pb文件
    with tf.gfile.FastGFile(pb_file_path + 'model1.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    #Builds the SavedModel protocol buffer and saves variables and assets
    #在和project相同层级目录下产生带有savemodel名称的文件夹
    builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path + 'savemodel1')
    #Adds the current meta graph to the SavedModel and saves variables
    #第二个参数为字符列表形式的tags – The set of tags with which to save the meta graph
    builder.add_meta_graph_and_variables(sess, ['cpu_server_1'])
    #Writes a SavedModel protocol buffer to disk
    #此处p值为生成的文件夹路径
    p = builder.save()
    print(p)