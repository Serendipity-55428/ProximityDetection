#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: CombinationTest
@time: 2018/12/5 21:51
@desc:
'''

import tensorflow as tf
import numpy as np
from HyMultiNN import RecurrentNeuralNetwork, FCNN, CNN
import time

def variable_summaries(var, name):
    '''监控指标可视化函数'''
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

def SAlstm(x, num_units, arg_dict):
    '''
    SA-LSTM模型由一个单向LSTM结点级联一个双向LSTM结点对级联两个全连接层构成
    :param x: type= 'ndarray'
    :param num_units: lstm隐层神经元数量
    :param arg_dict: 全连接层权重以及偏置量矩阵散列
    :return: SA-LSTM模型最终输出
    '''
    with tf.name_scope('single_lstm'):
        # 生成RecurrentNeuralNetwork对象
        recurrentnn = RecurrentNeuralNetwork(x, keep_prob= 0.8)
        # 添加单层LSTM的cell
        cell = recurrentnn.get_a_cell(net_name='LSTM', num_units=num_units)
        # dropout
        cell_dropout = recurrentnn.dropout_rnn(cell=cell)
        outputs, _ = recurrentnn.dynamic_rnn(cell, x, max_time=5)  # outputs.shape= [batch_size, max_time, hide_size]

    with tf.name_scope('bi_lstm'):
        # 添加双向LSTM的结点并输出各个时刻输出和最终隐藏层输出
        cell_fw = recurrentnn.get_a_cell(net_name='LSTM', num_units=num_units)
        cell_bw = recurrentnn.get_a_cell(net_name='LSTM', num_units=num_units)
        output_bi, state_bi = RecurrentNeuralNetwork.bidirectional_dynamic_rnn(cell_fw, cell_bw, outputs,
                                                                               batch_size=x.shape[0])
        result = tf.concat([state_bi[0].h, state_bi[-1].h], -1)

    with tf.name_scope('FC'):
        # 生成FCNN对象
        fcnn = FCNN(result, keep_prob= 1.0)
        net_1 = fcnn.per_layer(arg_dict['w_1'], arg_dict['b_1'])
        net_2 = fcnn.per_layer(arg_dict['w_2'], arg_dict['b_2'], param=net_1)
        return net_1, net_2


def main_1():
    '''
    SA-LSTM主函数
    :return: None
    '''
    # 全连接层参数散列
    weight = {
        'w_1_insize': 512,
        'w_1_outsize': 1000,
        'w_2_insize': 1000,
        'w_2_outsize': 1,
        'b_1': 1000,
        'b_2': 1}

    with tf.name_scope('weights'):
        # 权重以及激活值矩阵
        arg_dict = {
            'w_1': tf.Variable(
                tf.truncated_normal(shape=[weight['w_1_insize'], weight['w_1_outsize']], mean=0, stddev=0.1,
                                    dtype=tf.float32), name='w_1'),
            'w_2': tf.Variable(
                tf.truncated_normal(shape=[weight['w_2_insize'], weight['w_2_outsize']], mean=0, stddev=0.1,
                                    dtype=tf.float32), name='w_2'),
            'b_1': tf.Variable(tf.truncated_normal(shape=[weight['b_1']], mean=0, stddev=0.1, dtype=tf.float32),
                               name='b_1'),
            'b_2': tf.Variable(tf.truncated_normal(shape=[weight['b_2']], mean=0, stddev=0.1, dtype=tf.float32),
                               name='b_2')}
        variable_summaries(arg_dict['w_1'], 'w_1')
        variable_summaries(arg_dict['w_2'], 'w_2')
        variable_summaries(arg_dict['b_1'], 'b_1')
        variable_summaries(arg_dict['b_2'], 'b_2')

    x = np.arange(80, dtype= np.float32).reshape(4, 20)
    y = np.arange(4, dtype= np.float32)
    y_T = tf.placeholder(dtype= tf.float32, shape= [4])
    net_1, net_2 = SAlstm(x, num_units= 256, arg_dict= arg_dict)
    with tf.name_scope('loss-optimize'):
        loss = tf.reduce_mean(tf.square(net_2 - y_T))
        tf.summary.scalar('loss', loss)
        optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    init = tf.global_variables_initializer()
    # 摘要汇总
    merged = tf.summary.merge_all()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options= gpu_options)) as sess:
        sess.run(init)
        # 摘要文件
        summary_writer = tf.summary.FileWriter('logs/', sess.graph)
        for  i in range(1000):
            summary = sess.run(merged, feed_dict= {y_T: y})
            _, loss_s = sess.run([optimize, loss], feed_dict= {y_T: y})
            print(loss_s)
            # 添加摘要
            summary_writer.add_summary(summary, i)
        summary_writer.close()

def MUlstm(x, num_units, arg_dict):
    '''
    MU-LSTM模型由两层lstm结点级联两层全连接层构成
    :param x: type= 'ndarray'
    :param num_units: lstm/gru隐层神经元数量
    :param arg_dict: 全连接层权重以及偏置量矩阵散列
    :return: MULSTM模型最终输出
    '''
    with tf.name_scope('multi_lstm'):
        # 生成RecurrentNeuralNetwork对象
        recurrentnn = RecurrentNeuralNetwork(x, keep_prob=0.8)
        # 添加layer_num层LSTM结点组合
        # LSTM
        # cells = recurrentnn.multiLSTM(net_name='LSTM', num_unit=num_units, layer_num=2)
        # GRU
        cells = recurrentnn.multiLSTM(net_name='GRU', num_unit=num_units, layer_num= 3)
        # outputs.shape= [batch_size, max_time, hide_size]
        # multi_state= ((h, c), (h, c)), h.shape= [batch_size, hide_size]
        outputs, multi_state = recurrentnn.dynamic_rnn(cells, x, max_time=5)
        # LSTM
        # result = multi_state[-1].h
        # GRU
        result = multi_state[-1]
        # 生成FCNN对象

    with tf.name_scope('fc'):
        fcnn = FCNN(result, keep_prob=1.0)
        net_1 = fcnn.per_layer(arg_dict['w_1'], arg_dict['b_1'])

    return net_1

def main_2():
    '''
    MU-LSTM主函数
    :return: None
    '''
    # 全连接层参数散列
    weight = {
        'w_1_insize': 256,
        'w_1_outsize': 1,
        'w_2_insize': 1000,
        'w_2_outsize': 1,
        'b_1': 1,
        'b_2': 1}

    with tf.name_scope('weights'):
        # 权重以及激活值矩阵
        arg_dict = {
            'w_1': tf.Variable(
                tf.truncated_normal(shape=[weight['w_1_insize'], weight['w_1_outsize']], mean=0, stddev=0.1,
                                    dtype=tf.float32), name='w_1'),
            'w_2': tf.Variable(
                tf.truncated_normal(shape=[weight['w_2_insize'], weight['w_2_outsize']], mean=0, stddev=0.1,
                                    dtype=tf.float32), name='w_2'),
            'b_1': tf.Variable(tf.truncated_normal(shape=[weight['b_1']], mean=0, stddev=0.1, dtype=tf.float32),
                               name='b_1'),
            'b_2': tf.Variable(tf.truncated_normal(shape=[weight['b_2']], mean=0, stddev=0.1, dtype=tf.float32),
                               name='b_2')}

        variable_summaries(arg_dict['w_1'], 'w_1')
        # variable_summaries(arg_dict['w_2'], 'w_2')
        variable_summaries(arg_dict['b_1'], 'b_1')
        # variable_summaries(arg_dict['b_2'], 'b_2')


    x = np.arange(80, dtype= np.float32).reshape(4, 20)
    y = np.arange(4, dtype= np.float32)
    y_T = tf.placeholder(dtype= tf.float32, shape= [4])
    net_1 = MUlstm(x, num_units= 256, arg_dict= arg_dict)

    with tf.name_scope('loss-optimize'):
        loss = tf.reduce_mean(tf.square(net_1 - y_T))
        tf.summary.scalar('loss', loss)
        optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    init = tf.global_variables_initializer()
    # 摘要汇总
    merged = tf.summary.merge_all()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options= gpu_options)) as sess:
        sess.run(init)
        # 摘要文件
        summary_writer = tf.summary.FileWriter('logs/', sess.graph)
        start = time.time()
        for  i in range(10000):
            summary = sess.run(merged, feed_dict={y_T: y})
            _, loss_s = sess.run([optimize, loss], feed_dict= {y_T: y})
            print(loss_s)
            summary_writer.add_summary(summary, i)
        stop = time.time()
        summary_writer.close()
        print('GRU网络运行时间为: %s' % (stop - start))

def stacking_CNN(x, arg_dict, keep_prob):
    '''
    stacking策略中CNN子学习器
    :param x: Tensor
    :param arg_dict: cnn和fc所需所有权重和偏置值散列表
    :param keep_prob: dropout参数
    :return: 全连接层最后输出一个最优半径值
    '''
    cnn = CNN()
    #两个维度一样的卷积层，一个池化层，一个全连接层
    with tf.name_scope('conv_layer'):
        conv_1, training_1, extra_update_ops_1 = cnn.conv2d(x, arg_dict['wc1'], arg_dict['bc1'], strides=1, use_bn='no')
        conv_2, training_2, extra_update_ops_2 = cnn.conv2d(conv_1, arg_dict['wc2'], arg_dict['bc2'], strides=1, use_bn='no')
        pooling_2 = cnn.pooling(style=tf.nn.max_pool, x=conv_2, k=2)
    with tf.name_scope('fc_layer'):
        fc1_input = tf.reshape(pooling_2, [-1, arg_dict['wd1'].get_shape().as_list()[0]])
        if (training_1 or training_2) == None:
            keep_prob = 0.8
        fcnn = FCNN(fc1_input, keep_prob)
        fc_1 = fcnn.per_layer(arg_dict['wd1'], arg_dict['bd1'])
        fc_2 = fcnn.per_layer(arg_dict['wd2'], arg_dict['bd2'], param=fc_1)
        out = tf.nn.bias_add(tf.matmul(fc_2, arg_dict['w_out']), arg_dict['b_out'])

    return out

def stacking_GRU(x, num_units, arg_dict):
    '''
    stacking策略中的RNN子学习器
    :param x: type= 'ndarray'
    :param num_units: lstm/gru隐层神经元数量
    :param arg_dict: 全连接层权重以及偏置量矩阵散列
    :return: MULSTM模型最终输出
    '''
    with tf.name_scope('multi_LSTM(GRU)'):
        # 生成RecurrentNeuralNetwork对象

        #一层一对一输出隐层状态的GRU/LSTM,一层多对一输出隐层状态的GRU/LSTM,
        # 衔接一层神经元结点为上一层一半的fc层，再衔接一层神经元数量为上一层一半的fc层
        recurrentnn = RecurrentNeuralNetwork(x, keep_prob=0.8)
        # 添加layer_num层LSTM结点组合
        # LSTM
        # cells = recurrentnn.multiLSTM(net_name='LSTM', num_unit=num_units, layer_num=2)
        # GRU
        cells = recurrentnn.multiLSTM(net_name='GRU', num_unit= num_units, layer_num= 2)
        # outputs.shape= [batch_size, max_time, hide_size]
        # multi_state= ((h, c), (h, c)), h.shape= [batch_size, hide_size]
        outputs, multi_state = recurrentnn.dynamic_rnn(cells, x, max_time= 6) #原始数据样本特征为24，被平均分为6份输入
        # LSTM
        # result = multi_state[-1].h
        # GRU
        result = multi_state[-1]
        # 生成FCNN对象

    with tf.name_scope('fc'):
        fcnn = FCNN(result, keep_prob=1.0)
        net_1 = fcnn.per_layer(arg_dict['w_1'], arg_dict['b_1'])
        net_2 = fcnn.per_layer(arg_dict['w_2'], arg_dict['b_2'], param= net_1)
    return net_2

def stacking_FC(x, arg_dict):
    '''
    元学习器为两层全连接层
    :param x: Tensor, 所有子学习器生成的数据集
    :param arg_dict: 权重矩阵以及偏置值散列表
    :return: 全连接网络输出， shape= [1]
    '''
    #生成FCNN对象
    fcnn = FCNN(x, arg_dict)
    net_1 = fcnn.per_layer(arg_dict['w_sub_1'], arg_dict['b_sub_1'])
    net_2 = fcnn.per_layer(arg_dict['w_sub_2'], arg_dict['b_sub_2'], param= net_1)
    net_3 = fcnn.per_layer(arg_dict['w_sub_3'], arg_dict['b_sub_3'], param= net_2)
    return net_3

def sub_LossOpitimize(net, target, optimize_function, learning_rate):
    '''
    对子学习器做损失函数的优化过程
    :param net: 网络最终的ops
    :param target: 批次数据标签
    :param optimize_function: 自选优化函数
    :param learning_rate: 学习率
    :return: 损失函数和优化损失函数的操作结点ops
    '''
    with tf.name_scope('loss_optimize'):
        loss = tf.reduce_mean(tf.square(net - target))
        optimize = optimize_function(learning_rate= learning_rate).minimize(loss)
    return optimize, loss

def stacking_main():
    ''''''


if __name__ == '__main__':
 ''''''

