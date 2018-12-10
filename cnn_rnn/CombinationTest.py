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
from cnn_rnn.HyMultiNN import RecurrentNeuralNetwork, FCNN, CNN
from cnn_rnn.Fmake2read import FileoOperation
import time

def variable_summaries(var, name):
    '''监控指标可视化函数'''
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

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
    '''
    stacking策略
    :param files: ParseDequeue函数所需参数
    :return: 多线程生成特征矩阵和标签向量
    '''
    p_in = None
    filename = None
    read_in_fun = None
    num_shards = None
    instance_per_shard = None
    ftype = None
    ttype = None
    fshape = None
    tshape = None
    batch_size = None
    capacity = None
    batch_fun = None
    batch_step = None
    files = None

    #定义读取数据类对象
    fileoperation = FileoOperation(p_in, filename, read_in_fun, num_shards, instance_per_shard, ftype, ttype, fshape, tshape,
                 batch_size, capacity, batch_fun, batch_step)

    feature_batch, target_batch = fileoperation.ParseDequeue(files)
    with tf.Session() as sess:
        # 在使用tf.train。match_filenames_once函数时需要初始化一些变量
        sess.run(tf.local_variables_initializer())
        # sess.run(tf.global_variables_initializer())

        # 线程调配管理器
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 获取并打印组合之后的样例
        # 由于tf.train。match_filenames_once函数机制:
        # The returned operation is a dequeue operation and will throw
        # tf.errors.OutOfRangeError if the input queue is exhausted.If
        # this operation is feeding another input queue, its queue runner
        # will catch this exception, however, if this operation is used
        # in your main thread you are responsible for catching this yourself.
        # 故需要在循环读取时及时捕捉异常
        train_steps = batch_step
        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                cur_feature_batch, cur_target_batch = sess.run([feature_batch, target_batch])
                print(cur_feature_batch, cur_target_batch)

                train_steps -= 1
                if train_steps <= 0:
                    coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            # When done, ask the threads to stop. 请求该线程停止
            coord.request_stop()
            # And wait for them to actually do it. 等待被指定的线程终止
            coord.join(threads)


if __name__ == '__main__':
 ''''''

