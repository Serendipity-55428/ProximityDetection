#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: model_test
@time: 2019/2/26 22:54
@desc:
'''
import tensorflow as tf
import numpy as np
import os
from Routine_operation import SaveImport_model

def cnn_pred():
    '''
    将待预测数据放入训练好的cnn模型中进行预测
    :return: cnn模型预测结果
    '''
    cnn_graph = tf.Graph()
    with cnn_graph.as_default():
        # 导入数据(实例)
        xs = np.arange(25).reshape(1, 25)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=cnn_graph) as sess:
        sess.run(init)
        savemodel = SaveImport_model(sess_ori=None, file_suffix='\\cnn_model', ops=None, usefulplaceholder_count=3)
        savemodel.use_pb(sess_new=sess)
        # 导入待使用节点
        x = savemodel.import_ops(sess_new=sess, op_name='placeholder/xt')
        bn_istraining = savemodel.import_ops(sess_new=sess, op_name='placeholder/bn_istraining')
        cnn_op = savemodel.import_ops(sess_new=sess, op_name='cnn_output/Relu')
        pred = sess.run(cnn_op, feed_dict={x: xs, bn_istraining: False})
        print(pred)
    return pred

def rnn_pred():
    '''
    将待预测数据放入训练好的rnn模型中进行预测
    :return: rnn模型预测结果
    '''
    rnn_graph = tf.Graph()
    with rnn_graph.as_default():
        # 导入数据(实例)
        xs = np.arange(20).reshape(1, 20)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=rnn_graph) as sess:
        sess.run(init)
        savemodel = SaveImport_model(sess_ori=None, file_suffix='\\rnn_model', ops=None, usefulplaceholder_count=2)
        savemodel.use_pb(sess_new=sess)
        # 导入待使用节点
        x = savemodel.import_ops(sess_new=sess, op_name='placeholder/xt')
        rnn_op = savemodel.import_ops(sess_new=sess, op_name='rnn-output/Relu')
        pred = sess.run(rnn_op, feed_dict={x: xs})
        print(pred)
    return pred

def fnn_pred():
    '''
    将待预测数据放入训练好的fnn模型中进行预测
    :return: fnn模型预测结果
    '''
    fnn_graph = tf.Graph()
    with fnn_graph.as_default():
        # 导入数据(实例)
        xs = np.arange(6).reshape(1, 6)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=fnn_graph) as sess:
        sess.run(init)
        savemodel = SaveImport_model(sess_ori=None, file_suffix='\\fc_model', ops=None, usefulplaceholder_count=2)
        savemodel.use_pb(sess_new=sess)
        # 导入待使用节点
        x = savemodel.import_ops(sess_new=sess, op_name='placeholder/xt')
        rnn_op = savemodel.import_ops(sess_new=sess, op_name='fc_output/Relu_2')
        pred = sess.run(rnn_op, feed_dict={x: xs})
        print(pred)
    return pred

if __name__ == '__main__':
    #先进行cnn和rnn模型预测
    # cnn_result = cnn_pred()
    # rnn_result = rnn_pred()
    #最后进行fnn模型预测
    fnn_result = fnn_pred()



