#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: FIRST_GRAPH
@time: 2019/2/20 21:48
@desc:
'''
from AllNet import CNN, RNN, FNN
from TestEvaluation import Evaluation
from DataGenerate import data_stepone, data_steptwo, second_dataset
from Routine_operation import SaveFile, LoadFile
import tensorflow as tf
import numpy as np
import os
import pickle

#cnn模块################################################################
def cnn_mode():
    '''
    cnn计算图
    :return: None
    '''
    CNN_graph = tf.Graph()
    with CNN_graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(None, 20), name='xt')
        y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='yt')
        bn_istraining = tf.placeholder(dtype=tf.bool, name='bn_istraining')
        learning_rate_cnn = tf.placeholder(dtype=tf.float32, name='learning_rate_cnn')

        # 核尺寸
        kernel_size = {
            'w1_edge': 3,
            'w1_deep': 96,
            'w2_edge': 3,
            'w2_deep': 256,
            'w3_edge': 3,
            'w3_deep': 384,
            'w4_edge': 2,
            'w4_deep': 384,
            'w5_edge': 1,
            'w5_deep': 256,
            'w6_edge': 1,
            'w6_deep': 96
        }
        # 核张量
        kernel_para = {
            'w1': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w1_edge'], kernel_size['w1_edge'], 1, kernel_size['w1_deep']),
                mean=0, stddev=1), dtype=tf.float32, name='w1'),
            'w2': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w2_edge'], kernel_size['w2_edge'], kernel_size['w1_deep'], kernel_size['w2_deep']),
                mean=0, stddev=1), dtype=tf.float32, name='w2'),
            'w3': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w3_edge'], kernel_size['w3_edge'], kernel_size['w2_deep'], kernel_size['w3_deep']),
                mean=0, stddev=1), dtype=tf.float32, name='w3'),
            'w4': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w4_edge'], kernel_size['w4_edge'], kernel_size['w3_deep'], kernel_size['w4_deep']),
                mean=0, stddev=1), dtype=tf.float32, name='w4'),
            'w5': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w5_edge'], kernel_size['w5_edge'], kernel_size['w4_deep'], kernel_size['w5_deep']),
                mean=0, stddev=1), dtype=tf.float32, name='w5'),
            'w6': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w6_edge'], kernel_size['w6_edge'], kernel_size['w5_deep'], kernel_size['w6_deep']),
                mean=0, stddev=1), dtype=tf.float32, name='w6'),
        }

        # 卷积层(数据特征维度：20->5*5)
        cnn_1 = CNN(x=x, w_conv=kernel_para['w1'], stride_conv=1, stride_pool=2)
        # 将向量x转换为5*5方形张量
        x_reshape = CNN.reshape(f_vector=x, new_shape=(-1, 5, 5, 1))
        # 1
        layer_1 = cnn_1.convolution(input=x_reshape)
        relu1 = tf.nn.relu(layer_1)
        bn1 = cnn_1.batch_normoalization(input=relu1, is_training=bn_istraining)
        # 2
        cnn_2 = CNN(x=bn1, w_conv=kernel_para['w2'], stride_conv=1, stride_pool=2)
        layer_2 = cnn_2.convolution(input=bn1)
        relu2 = tf.nn.relu(layer_2)
        bn2 = cnn_2.batch_normoalization(input=relu2, is_training=bn_istraining)
        # 3
        cnn_3 = CNN(x=bn2, w_conv=kernel_para['w3'], stride_conv=1, stride_pool=2)
        layer_3 = cnn_3.convolution(input=bn2)
        relu3 = tf.nn.relu(layer_3)
        bn3 = cnn_3.batch_normoalization(input=relu3, is_training=bn_istraining)
        # pool
        pool1 = cnn_3.pooling(pool_fun=tf.nn.max_pool, input=bn3)
        # 4
        cnn_4 = CNN(x=pool1, w_conv=kernel_para['w4'], stride_conv=1, stride_pool=2)
        layer_4 = cnn_4.convolution()
        relu4 = tf.nn.relu(layer_4)
        bn4 = cnn_4.batch_normoalization(input=relu4, is_training=bn_istraining)
        # 5
        cnn_5 = CNN(x=bn4, w_conv=kernel_para['w5'], stride_conv=1, stride_pool=2)
        layer_5 = cnn_5.convolution()
        relu5 = tf.nn.relu(layer_5)
        bn5 = cnn_5.batch_normoalization(input=relu5, is_training=bn_istraining)
        # pool
        pool2 = cnn_5.pooling(pool_fun=tf.nn.max_pool, input=bn5)
        # 6
        cnn_6 = CNN(x=pool2, w_conv=kernel_para['w6'], stride_conv=1, stride_pool=2)
        layer_6 = cnn_6.convolution()
        relu6 = tf.nn.relu(layer_6)
        bn6 = cnn_6.batch_normoalization(input=relu6, is_training=bn_istraining)
        # 经过6层卷积运算后张量维度为：(-1, 2, 2, 96)
        # flat
        bn6_x, bn6_y, bn6_z = bn6.get_shape().as_list()[1:]
        cnn_output = CNN.reshape(f_vector=bn6, new_shape=(-1, bn6_x * bn6_y * bn6_z))
        # cnn投影核
        w_cnn = tf.Variable(initial_value=tf.truncated_normal(shape=(cnn_output.get_shape().as_list()[-1], 1),
                                                              mean=0, stddev=1), dtype=tf.float32, name='w_cnn')
        b_cnn = tf.Variable(initial_value=tf.truncated_normal(shape=([1]), mean=0, stddev=1), dtype=tf.float32,
                            name='b_cnn')
        # 预测半径
        r_cnn = tf.matmul(cnn_output, w_cnn) + b_cnn
        relu_r_cnn = tf.nn.relu(r_cnn)
        # 代价函数、优化函数、评价指标
        loss_cnn = tf.reduce_mean(tf.square(relu_r_cnn - y))
        optimize_cnn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_cnn).minimize(loss_cnn)
        evaluation_cnn = Evaluation(one_hot=False, logit=None, label=None, regression_label=y,
                                    regression_pred=relu_r_cnn)
        acc_cnn = evaluation_cnn.acc_regression(Threshold=0.1)  # 评价指标中的阈值可以修改

        # 变量初始化节点,显卡占用率分配节点
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=CNN_graph) as sess:
        sess.run(init)
        # 导入数据
        p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data.pickle'
        dataset_ori = LoadFile(p=p_dataset_ori)
        # 记录折数
        fold = 0
        # 总训练轮数
        epoch_all = 10000
        # 设定所有折在cnn子学习器最终的预测值
        first_sub_cnn_pred = np.zeros(dtype=np.float32, shape=([1]))
        # 将数据集划分为训练集和测试集
        for train, test in data_stepone(p_dataset_ori=p_dataset_ori, padding=True, proportion=5):
            for epoch in range(epoch_all):
                # 以一定批次读入某一折数据进行训练
                for batch_x, batch_y in data_steptwo(train_data=train, batch_size=500):
                    _ = sess.run(optimize_cnn,
                                 feed_dict={x: batch_x, y: batch_y, bn_istraining: True, learning_rate_cnn: 1e-4})
                    # 设定标志在100的倍数epoch时只输出一次结果
                    flag = 1
                    if (epoch % 100) == 0 and flag == 1:
                        loss_cnn_ = sess.run(loss_cnn, feed_dict={x: batch_x, y: batch_y, bn_istraining: True})
                        acc_cnn_ = sess.run(acc_cnn, feed_dict={x: test[:, :-1], y: test[:, -1], bn_istraining: False})
                        print('第%s轮后训练集损失为: %s, 第 %s 折预测准确率为: %s' % (epoch, loss_cnn_, fold, acc_cnn_))
                        flag = 0

            # 记录循环多次后该折子学习器最终预测结果
            first_sub_cnn_pred = acc_cnn_ if first_sub_cnn_pred.any() == 0 else np.vstack(
                (first_sub_cnn_pred, acc_cnn_))
            fold += 1

    p_cnn = r'F:\ProximityDetection\Stacking\cnn_pred.pickle'
    SaveFile(data=first_sub_cnn_pred, savepickle_p=p_cnn)


#rnn模块################################################################
def rnn_mode():
    '''
    rnn计算图
    :return: None
    '''
    RNN_graph = tf.Graph()
    with RNN_graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(None, 20), name='xt')
        y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='yt')
        learning_rate_rnn = tf.placeholder(dtype=tf.float32, name='learning_rate_rnn')

        rnn = RNN(x=x, max_time=5, num_units=128)
        rnn_outputs, _ = rnn.dynamic_rnn(style='LSTM', output_keep_prob=0.8)
        rnn_output = rnn_outputs[:, -1, :]
        # rnn投影核
        w_rnn = tf.Variable(initial_value=tf.truncated_normal(shape=(rnn_output.get_shape().as_list()[-1], 1),
                                                              mean=0, stddev=1), dtype=tf.float32, name='w_rnn')
        b_rnn = tf.Variable(initial_value=tf.truncated_normal(shape=([1]), mean=0, stddev=1), dtype=tf.float32,
                            name='b_rnn')
        # 预测半径
        r_rnn = tf.matmul(rnn_output, w_rnn) + b_rnn
        relu_r_rnn = tf.nn.relu(r_rnn)
        # 代价函数、优化函数、评价指标
        loss_rnn = tf.reduce_mean(tf.square(relu_r_rnn - y))
        optimize_rnn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_rnn).minimize(loss_rnn)
        evaluation_rnn = Evaluation(one_hot=False, logit=None, label=None, regression_label=y,
                                    regression_pred=relu_r_rnn)
        acc_rnn = evaluation_rnn.acc_regression(Threshold=0.1)  # 评价指标中的阈值可以修改

        # 变量初始化节点,显卡占用率分配节点
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=RNN_graph) as sess:
        sess.run(init)
        # 导入数据
        p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data.pickle'
        dataset_ori = LoadFile(p=p_dataset_ori)
        # 记录折数
        fold = 0
        # 总训练轮数
        epoch_all = 10000
        # 设定所有折在cnn子学习器最终的预测值
        first_sub_rnn_pred = np.zeros(dtype=np.float32, shape=([1]))
        # 将数据集划分为训练集和测试集
        for train, test in data_stepone(p_dataset_ori=p_dataset_ori, padding=False, proportion=5):
            for epoch in range(epoch_all):
                # 以一定批次读入某一折数据进行训练
                for batch_x, batch_y in data_steptwo(train_data=train, batch_size=500):
                    _ = sess.run(optimize_rnn, feed_dict={x: batch_x, y: batch_y, learning_rate_rnn: 1e-4})
                    # 设定标志在100的倍数epoch时只输出一次结果
                    flag = 1
                    if (epoch % 100) == 0 and flag == 1:
                        loss_rnn_ = sess.run(loss_rnn, feed_dict={x: batch_x, y: batch_y})
                        acc_rnn_ = sess.run(acc_rnn, feed_dict={x: test[:, :-1], y: test[:, -1]})
                        print('第%s轮后训练集损失为: %s, 第 %s 折预测准确率为: %s' % (epoch, loss_rnn_, fold, acc_rnn_))
                        flag = 0

            # 记录循环多次后该折子学习器最终预测结果
            first_sub_rnn_pred = acc_rnn_ if first_sub_rnn_pred.any() == 0 else np.vstack(
                (first_sub_rnn_pred, acc_rnn_))
            fold += 1

    p_rnn = r'F:\ProximityDetection\Stacking\rnn_pred.pickle'
    SaveFile(data=first_sub_rnn_pred, savepickle_p=p_rnn)

#fnn模块################################################################
def fnn_mode():
    '''
    fnn计算图
    :return: None
    '''
    FNN_graph = tf.Graph()
    with FNN_graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(None, 6), name='xt')
        y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='yt')
        learning_rate_fnn = tf.placeholder(dtype=tf.float32, name='learning_rate_fnn')

        h_size = {
            'w1_insize': 6,
            'w1_outsize': 256,
            'w_2_insize': 256,
            'w_2_outsize': 96,
            'w_3_insize': 96,
            'w_3_outsize': 1
        }

        h_para = (
            (tf.Variable(initial_value= tf.truncated_normal(shape= ([h_size['w1_insize1'], h_size['w1_outsize']]),
                                                            mean= 0, stddev= 1), dtype= tf.float32, name= 'w1'),
             tf.Variable(initial_value= tf.truncated_normal(shape= ([h_size['w_1_outsize']]),
                                                            mean= 0, stddev= 1), dtype= tf.float32, name= 'b1')),
            (tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w2_insize1'], h_size['w2_outsize']]),
                                                           mean=0, stddev=1), dtype=tf.float32, name='w2'),
             tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w_2_outsize']]),
                                                           mean=0, stddev=1), dtype=tf.float32, name='b2')),
            (tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w3_insize1'], h_size['w3_outsize']]),
                                                           mean=0, stddev=1), dtype=tf.float32, name='w3'),
             tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w_3_outsize']]),
                                                           mean=0, stddev=1), dtype=tf.float32, name='b3')),
        )

        fnn = FNN(x= x, w= h_para)
        fc_output = fnn.fc_concat(keep_prob= 0.8)
        # 代价函数、优化函数、评价指标
        loss_fnn = tf.reduce_mean(tf.square(fc_output - y))
        optimize_rnn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_fnn).minimize(loss_fnn)
        evaluation_rnn = Evaluation(one_hot=False, logit=None, label=None, regression_label=y,
                                    regression_pred=fc_output)
        acc_fnn = evaluation_rnn.acc_regression(Threshold=0.1)  # 评价指标中的阈值可以修改

        # 变量初始化节点,显卡占用率分配节点
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=FNN_graph) as sess:
        sess.run(init)
        # 导入数据
        p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data.pickle' #改
        dataset_ori = LoadFile(p=p_dataset_ori)
        # 记录折数
        fold = 0
        # 总训练轮数
        epoch_all = 10000
        # 将数据集划分为训练集和测试集
        for train, test in data_stepone(p_dataset_ori=p_dataset_ori, padding=False, proportion=5):
            for epoch in range(epoch_all):
                # 以一定批次读入某一折数据进行训练
                for batch_x, batch_y in data_steptwo(train_data=train, batch_size=500):
                    _ = sess.run(optimize_rnn, feed_dict={x: batch_x, y: batch_y, learning_rate_fnn: 1e-4})
                    # 设定标志在100的倍数epoch时只输出一次结果
                    flag = 1
                    if (epoch % 100) == 0 and flag == 1:
                        loss_rnn_ = sess.run(loss_fnn, feed_dict={x: batch_x, y: batch_y})
                        acc_rnn_ = sess.run(acc_fnn, feed_dict={x: test[:, :-1], y: test[:, -1]})
                        print('第%s轮后训练集损失为: %s, 第 %s 折预测准确率为: %s' % (epoch, loss_rnn_, fold, acc_rnn_))
                        flag = 0

            fold += 1

if __name__ == '__main__':
    pass