#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: Model_test
@time: 2018/12/18 14:57
@desc:
'''
import numpy as np
import tensorflow as tf
import os

#pb文件路径（在当前目录下生成）
pb_file_path = os.getcwd()

#建立两个计算图，分别用于载入初级学习器和次级学习器的所有计算节点,并将两个计算图的读取放在两个不同的会话中
g1 = tf.Graph()
with g1.as_default():
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config= tf.ConfigProto(gpu_options= gpu_options), graph= g1) as sess:
    sess.run(init)
    # Loads the model from a SavedModel as specified by tags
    tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path+'pri_savemodel')

    # Returns the Tensor with the given name
    # 名称都为'{output_name}: output_index'格式
    x = sess.graph.get_tensor_by_name('x-y/x:0')
    # y = sess.graph.get_tensor_by_name('y:0')
    cnn_op = sess.graph.get_tensor_by_name('cnn_ops/fc_layer/cnn_op/mul:0')
    gru_op = sess.graph.get_tensor_by_name('gru_ops/fc/Relu_2:0')

    #读入的数据需要用户自行输入
    input = np.arange(80*24, dtype= np.float32).reshape(80, 24)
    predict_cnn, predict_gru = sess.run([cnn_op, gru_op], feed_dict={x: input[:, 4:]})
    #初级学习器输出特征（需要输入次级学习器）
    pri_input = np.hstack((input[:, :4], predict_cnn, predict_gru))
    print('初级学习器预测特征向量为:', pri_input)

g2 = tf.Graph()
with g2.as_default():
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.333)
    #将初级学习器得到的进一步特征定义在另一张计算图中
    sec_input = pri_input

with tf.Session(config= tf.ConfigProto(gpu_options= gpu_options), graph= g2) as sess:
    sess.run(init)
    # Loads the model from a SavedModel as specified by tags
    tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path + 'sec_savemodel')

    # Returns the Tensor with the given name
    # 名称都为'{output_name}: output_index'格式
    x = sess.graph.get_tensor_by_name('x-y/x:0')
    fc_op = sess.graph.get_tensor_by_name('fc_ops/Relu_2:0')

    #输出最优半径预测值
    r = sess.run(fc_op, feed_dict= {x: sec_input})
    print('最优半径为: %s' % r)




