#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: sub_learning
@time: 2018/12/11 20:14
@desc:
'''

import tensorflow as tf
import numpy as np
from cnn_rnn.HyMultiNN import RecurrentNeuralNetwork, FCNN, CNN
import time

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
        pooling_2 = cnn.pooling(style=tf.nn.max_pool, x=conv_2, k= 2)
    with tf.name_scope('fc_layer'):
        fc1_input = tf.reshape(pooling_2, [-1, arg_dict['wd1'].get_shape().as_list()[0]])
        if (training_1 or training_2) == None:
            keep_prob = 0.8
        fcnn = FCNN(fc1_input, keep_prob)
        fc = fcnn.per_layer(arg_dict['wd1'], arg_dict['bd1'])
        out = fcnn.per_layer(arg_dict['wd2'], arg_dict['bd2'], param= fc, name= 'cnn_ops')

    return out

def stacking_GRU(x, num_units, arg_dict):
    '''
    stacking策略中的RNN子学习器
    :param x: type= 'ndarray' / 'Tensor'
    :param num_units: lstm/gru隐层神经元数量
    :param arg_dict: 全连接层权重以及偏置量矩阵散列
    :return: MULSTM模型最终输出
    '''
    with tf.name_scope('multi_LSTMorGRU'):
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
        # (lstm)multi_state= ((h, c), (h, c)), (gru)multi_state= (h, h) h.shape= [batch_size, hide_size]
        outputs, multi_state = recurrentnn.dynamic_rnn(cells, x, max_time= 5) #若特征数24则分成6份，若特征数20则分成5份
        # LSTM
        # result = multi_state[-1].h
        # GRU
        result = multi_state[-1]
        # 生成FCNN对象

    with tf.name_scope('fc'):
        fcnn = FCNN(result, keep_prob=1.0)
        net_1 = fcnn.per_layer(arg_dict['w_1'], arg_dict['b_1'])
        net_2 = fcnn.per_layer(arg_dict['w_2'], arg_dict['b_2'], param= net_1, name= 'gru_ops')
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