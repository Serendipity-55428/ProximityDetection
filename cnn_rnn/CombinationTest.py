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
from cnn_rnn.Sub_learning import stacking_CNN, stacking_GRU, stacking_FC
import time

def variable_summaries(var, name):
    '''监控指标可视化函数'''
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

def sub_LossOptimize(net, target, optimize_function, learning_rate):
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
    #训练集数据所需参数
    tr_p_in = None
    tr_filename = None
    tr_read_in_fun = None
    tr_num_shards = None
    tr_instance_per_shard = None
    tr_ftype = None
    tr_ttype = None
    tr_fshape = None
    tr_tshape = None
    tr_batch_size = None
    tr_capacity = None
    tr_batch_fun = None
    tr_batch_step = None
    tr_files = None
    tr_num_epochs = None

    #测试集数据所需参数
    te_p_in = None
    te_filename = None
    te_read_in_fun = None
    te_num_shards = None
    te_instance_per_shard = None
    te_ftype = None
    te_ttype = None
    te_fshape = None
    te_tshape = None
    te_batch_size = None
    te_capacity = None
    te_batch_fun = None
    te_batch_step = None
    te_files = None
    te_num_epochs = None

    #定义读取训练集数据对象
    train_fileoperation = FileoOperation(tr_p_in, tr_filename, tr_read_in_fun, tr_num_shards, tr_instance_per_shard,
                                         tr_ftype, tr_ttype, tr_fshape, tr_tshape, tr_batch_size, tr_capacity, tr_batch_fun, tr_batch_step)

    train_feature_batch, train_target_batch = train_fileoperation.ParseDequeue(tr_files, num_epochs= tr_num_epochs)

    #定义读取测试集数据对象
    test_fileoperation = FileoOperation(te_p_in, te_filename, te_read_in_fun, te_num_shards, te_instance_per_shard,
                                        te_ftype, te_ttype, te_fshape, te_tshape, te_batch_size, te_capacity, te_batch_fun, te_batch_step)

    test_feature_batch, test_feature_batch = test_fileoperation.ParseDequeue(te_files, num_epochs= te_num_epochs)

    #训练数据批次占位符,占位符读入数据形状和一个批次的数据特征矩阵形状相同
    x = tf.placeholder(dtype= tf.float32, shape= [tr_batch_size, tr_fshape])
    y = tf.placeholder(dtype= tf.float32, shape= [tr_batch_size, tr_tshape])

    ############################CNN############################
    #定义cnn子学习器中卷积核,全连接层参数矩阵以及偏置量尺寸
    cnn_weights = {
        'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 50], mean= 0, stddev= 1.0), dtype= tf.float32),
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 50, 50], mean= 0, stddev= 1.0), dtype= tf.float32),
        'bc1': tf.Variable(tf.truncated_normal([50], mean= 0, stddev= 1.0), dtype= tf.float32),
        'bc2': tf.Variable(tf.truncated_normal([50], mean= 0, stddev= 1.0), dtype= tf.float32),
        'wd1': tf.Variable(tf.truncated_normal([2*3*50, 100], mean= 0, stddev= 1.0), dtype= tf.float32),
        'bd1': tf.Variable(tf.truncated_normal([100], mean= 0, stddev= 1.0), dtype= tf.float32),
        'wd2': tf.Variable(tf.truncated_normal([100, 1], mean= 0, stddev= 1.0), dtype= tf.float32),
        'bd2': tf.Variable(tf.truncated_normal([1], mean= 0, stddev= 1.0), dtype= tf.float32)
    }

    #定义CNN类对象用以对数据Tensor进行改变形状
    cnn = CNN()
    #将每个样本特征向量由一维转化为二维shape= [batch_size, h, v, 1],原始数据有24个特征，转化为4*6维'picture'特征输入卷积神经网络
    x = cnn.d_one2d_two(x, 4, 6)
    #输出单个值的Variable
    out = stacking_CNN(x= x, arg_dict= cnn_weights, keep_prob= 1.0)
    #定义CNN自学习器的损失函数和优化器
    cnn_optimize, cnn_loss = sub_LossOptimize(out, y, optimize_function= tf.train.RMSPropOptimizer, learning_rate= 1e-4)

    ##############################RNN##############################
    #定义GRU子学习器中全连接层参数矩阵以及偏置量尺寸
    gru_weights = {
        'w_1': tf.Variable(tf.truncated_normal([256, 128], mean= 0, stddev= 1.0), dtype= tf.float32), #256为GRU网络最终输出的隐藏层结点数量
        'w_2': tf.Variable(tf.truncated_normal([128, 64], mean= 0, stddev= 1.0), dtype= tf.float32),
        'b_1': tf.Variable(tf.truncated_normal([128], mean= 0, stddev= 1.0), dtype= tf.float32),
        'b_2': tf.Variable(tf.truncated_normal([64], mean= 0, stddev= 1.0), dtype= tf.float32)
    }
    #定义多层GRU的最终输出ops
    net_2 = stacking_GRU(x= x, num_units= 256, arg_dict= gru_weights)
    #定义GRU自学习器的损失函数和优化器
    gru_optimize, gru_loss = sub_LossOptimize(net_2, y, optimize_function= tf.train.RMSPropOptimizer, learning_rate= 1e-4)

    #############################FC###############################
    #定义fc次级学习器中全连接层参数、
    fc_weights = {
        'w_sub_1': tf.Variable(tf.truncated_normal([2, 128], mean= 0, stddev= 1.0), dtype= tf.float32),
        'w_sub_2': tf.Variable(tf.truncated_normal([128, 64], mean= 0, stddev= 1.0), dtype= tf.float32),
        'w_sub_3': tf.Variable(tf.truncated_normal([64, 1], mean= 0, stddev= 1.0), dtype= tf.float32),
        'b_sub_1': tf.Variable(tf.truncated_normal([128], mean= 0, stddev= 1.0), dtype= tf.float32),
        'b_sub_2': tf.Variable(tf.truncated_normal([64], mean= 0, stddev= 1.0), dtype= tf.float32),
        'b_sub_3': tf.Variable(tf.truncated_normal([1], mean= 0, stddev= 1.0), dtype= tf.float32)
    }

    #定义FC次级学习器的最终输出ops
    net_3 = stacking_FC(x= x, arg_dict= fc_weights)
    #定义次级学习器的损失函数和优化器
    fc_optimize, fc_loss = sub_LossOptimize(net_3, y, optimize_function= tf.train.RMSPropOptimizer, learning_rate= 1e-4)

    with tf.Session() as sess:
        # 在使用tf.train。match_filenames_once函数时需要初始化一些变量
        sess.run(tf.local_variables_initializer())
        # sess.run(tf.global_variables_initializer())

        # 线程调配管理器
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph.
        threads = tf.train.start_queue_runners(sess= sess, coord= coord)

        train_steps = tr_batch_step #对于stacking策略，使用5折交叉验证，该参数设置为: 5折 * 5组 = 25
        #训练100000个epoch
        for epoch in range(100000):
            try:
                while not coord.should_stop():  # 如果线程应该停止则返回True
                    tr_feature_batch, tr_target_batch = sess.run([train_feature_batch, train_target_batch])
                    # print(cur_feature_batch, cur_target_batch)


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

