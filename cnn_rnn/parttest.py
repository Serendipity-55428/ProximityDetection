#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: parttest
@time: 2018/12/16 23:13
@desc:
'''
import tensorflow as tf
import numpy as np
from cnn_rnn.HyMultiNN import RecurrentNeuralNetwork, FCNN, CNN
from cnn_rnn.Fmake2read import FileoOperation
from cnn_rnn.sub_learning import stacking_CNN, stacking_GRU, stacking_FC
import pickle
import os.path
import time
import os
from tensorflow.python.framework import graph_util

def fun(p):
    '''不对数据做任何处理'''
    return np.arange(400*24).reshape(400, 24), np.arange(400)

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
        # tf.summary.scalar('loss', loss)
        optimize = optimize_function(learning_rate= learning_rate).minimize(loss)
    return optimize, loss

def coord_threads(sess):
    '''
    生成线程调配管理器和线程队列
    :param sess: 会话参数
    :return: coord, threads
    '''
    # 线程调配管理器
    coord = tf.train.Coordinator()
    # Starts all queue runners collected in the graph.
    threads = tf.train.start_queue_runners(sess= sess, coord= coord)
    return coord, threads

# 得到pb文件保存路径（当前目录下）
pb_file_path = os.getcwd()

#训练集数据所需参数
tr_p_in = 0
tr_filename = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\outputtrain.tfrecords-%.5d-of-%.5d'
tr_read_in_fun = fun
tr_num_shards = 5
tr_instance_per_shard = 80
tr_ftype = tf.float64
tr_ttype = tf.float64
tr_fshape = 24
tr_tshape = 1
tr_batch_size = 40
tr_capacity = 400 + 40 * 40
tr_batch_fun = tf.train.batch
tr_batch_step = 4
tr_files = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\outputtrain.tfrecords-*'
tr_num_epochs = 5000000

#测试集数据所需参数
te_p_in = 0
te_filename = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\outputtest.tfrecords-%.5d-of-%.5d'
te_read_in_fun = fun
te_num_shards = 5
te_instance_per_shard = 80
te_ftype = tf.float64
te_ttype = tf.float64
te_fshape = 24
te_tshape = 1
te_batch_size = 40
te_capacity = 400 + 40 * 40
te_batch_fun = tf.train.batch
te_batch_step = 4
te_files = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\outputtest.tfrecords-*'
te_num_epochs= 5000000

train_steps = tr_batch_step  # 对于stacking策略，使用5折交叉验证，该参数设置为4（5折，计数从0开始）
test_steps = te_batch_step  # 子学习器中测试集分批次预测

with tf.name_scope('data_batch'):
    # 定义读取训练集数据对象
    train_fileoperation = FileoOperation(tr_p_in, tr_filename, tr_read_in_fun, tr_num_shards, tr_instance_per_shard,
                                         tr_ftype, tr_ttype, tr_fshape, tr_tshape, tr_batch_size, tr_capacity,
                                         tr_batch_fun, tr_batch_step)

    train_feature_batch, train_target_batch = train_fileoperation.ParseDequeue(tr_files, num_epochs=tr_num_epochs)

    # 定义读取测试集数据对象
    test_fileoperation = FileoOperation(te_p_in, te_filename, te_read_in_fun, te_num_shards, te_instance_per_shard,
                                        te_ftype, te_ttype, te_fshape, te_tshape, te_batch_size, te_capacity,
                                        te_batch_fun, te_batch_step)

    test_feature_batch, test_target_batch = test_fileoperation.ParseDequeue(te_files, num_epochs=te_num_epochs)

with tf.name_scope('x-y'):
    # 训练数据批次占位符,占位符读入数据形状和一个批次的数据特征矩阵形状相同
    x = tf.placeholder(dtype=tf.float32, shape=[tr_batch_size, tr_fshape])
    y = tf.placeholder(dtype=tf.float32, shape=[tr_batch_size, tr_tshape])

############################CNN############################
# 定义cnn子学习器中卷积核,全连接层参数矩阵以及偏置量尺寸
with tf.name_scope('cnn_weights'):
    cnn_weights = {
        'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 50], mean=0, stddev=1.0), dtype=tf.float32),
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 50, 50], mean=0, stddev=1.0), dtype=tf.float32),
        'bc1': tf.Variable(tf.truncated_normal([50], mean=0, stddev=1.0), dtype=tf.float32),
        'bc2': tf.Variable(tf.truncated_normal([50], mean=0, stddev=1.0), dtype=tf.float32),
        'wd1': tf.Variable(tf.truncated_normal([2 * 3 * 50, 100], mean=0, stddev=1.0), dtype=tf.float32),
        'bd1': tf.Variable(tf.truncated_normal([100], mean=0, stddev=1.0), dtype=tf.float32),
        'wd2': tf.Variable(tf.truncated_normal([100, 1], mean=0, stddev=1.0), dtype=tf.float32),
        'bd2': tf.Variable(tf.truncated_normal([1], mean=0, stddev=1.0), dtype=tf.float32)
    }

    # variable_summaries(cnn_weights['wc1'], 'wc1')
    # variable_summaries(cnn_weights['wc2'], 'wc2')
    # variable_summaries(cnn_weights['bc1'], 'bc1')
    # variable_summaries(cnn_weights['bc2'], 'bc2')
    # variable_summaries(cnn_weights['wd1'], 'wd1')
    # variable_summaries(cnn_weights['bd1'], 'bd1')
    # variable_summaries(cnn_weights['wd2'], 'wd2')
    # variable_summaries(cnn_weights['bd2'], 'bd2')

# 定义CNN类对象用以对数据Tensor进行改变形状
cnn = CNN()
# 将每个样本特征向量由一维转化为二维shape= [batch_size, h, v, 1],原始数据有24个特征，转化为4*6维'picture'特征输入卷积神经网络
x = cnn.d_one2d_two(x, 4, 6)

with tf.name_scope('cnn_ops'):
    # 输出单个值的Variable
    cnn_ops = stacking_CNN(x=x, arg_dict=cnn_weights, keep_prob=1.0)

with tf.name_scope('cnn_optimize-loss'):
     # 定义CNN自学习器的损失函数和优化器
    cnn_optimize, cnn_loss = sub_LossOptimize(cnn_ops, y, optimize_function=tf.train.RMSPropOptimizer,
                                                  learning_rate=1e-4)

##############################RNN##############################

init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # 在使用tf.train。match_filenames_once函数时需要初始化一些变量
    sess.run(tf.local_variables_initializer())
    sess.run(init)
    # 摘要文件
    # summary_writer = tf.summary.FileWriter('logs/', sess.graph)

    # 训练集5折，每折分别作为测试样本
    for group in range(5):

        # 导入前一次训练保存的计算图
        if group:
            # Loads the model from a SavedModel as specified by tags
            tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path + 'savemodel')
            # Returns the Tensor with the given name
            # 名称都为'{name}:0'格式
            x = sess.graph.get_tensor_by_name('x:0')
            y = sess.graph.get_tensor_by_name('y:0')
            cnn_ops = sess.graph.get_tensor_by_name('cnn_ops:0')
            gru_ops = sess.graph.get_tensor_by_name('gru_ops:0')

        # 训练100000个epoch
        for epoch in range(100000):

            coord, threads = coord_threads(sess=sess)

            # summary = sess.run(merged, feed_dict={x: train_feature_batch, y: train_target_batch})
            try:
                while not coord.should_stop():  # 如果线程应该停止则返回True
                    tr_feature_batch, tr_target_batch = sess.run([train_feature_batch, train_target_batch])
                    # print(cur_feature_batch, cur_target_batch)
                    if not train_steps != (4 - group):
                        #######################
                        # 读入后20个特征
                        _ = sess.run(cnn_optimize, feed_dict={x: tr_feature_batch[:, 4:], y: tr_target_batch})
                        # _ = sess.run(gru_optimize, feed_dict={x: tr_feature_batch[:, 4:], y: tr_target_batch})
                        if not (epoch % 1000):
                            loss_cnn = sess.run(cnn_loss, feed_dict={x: tr_feature_batch[:, 4:], y: tr_target_batch})
                            # loss_gru = sess.run(gru_loss, feed_dict={x: tr_feature_batch[:, 4:], y: tr_target_batch})
                            print('CNN子学习器损失函在第 %s 个epoch的数值为: %s' % (epoch, loss_cnn))
                            # print('GRU子学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_gru))

                        #######################
                        #######################
                        # 读入所有特征
                        # _, loss_cnn = sess.run([cnn_optimize, cnn_loss],
                        #                        feed_dict={x: tr_feature_batch, y: tr_target_batch})
                        # print('CNN子学习器损失函在第 %s 个epoch的数值为: %s' % (epoch, loss_cnn))
                        # _, loss_gru = sess.run([gru_optimize, gru_loss],
                        #                        feed_dict={x: tr_feature_batch, y: tr_target_batch})
                        # print('GRU子学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_gru))
                        #######################
                    elif not epoch:  # 循环训练时只需在第一次提取训练集中待预测批次
                        # 将序号等于group的一折（批次）数据存入cross_tr_feature, cross_tr_target中
                        cross_tr_feature_batch, super_tr_target_batch = tr_feature_batch, tr_target_batch

                    train_steps -= 1
                    if train_steps <= 0:
                        coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

            except tf.errors.OutOfRangeError:
                print('第 %s 轮, 将训练集第 %s 折作为测试样本' % (epoch, group))
            finally:
                # When done, ask the threads to stop. 请求该线程停止
                coord.request_stop()
                # And wait for them to actually do it. 等待被指定的线程终止
                coord.join(threads)

                # summary_writer.add_summary(summary, epoch)

                # 重置训练批次数为最大
                train_steps = tr_batch_step

            if not (epoch % 100):
                # 对1折测试集输出均方差
                loss_cnn_test = sess.run(cnn_loss,
                                         feed_dict={x: cross_tr_feature_batch[:, 4:], y: super_tr_target_batch})
                # loss_gru_test = sess.run(gru_loss,
                #                          feed_dict={x: cross_tr_feature_batch[:, 4:], y: super_tr_target_batch})
                print('CNN子学习器损失函在第 %s 个epoch的数值为: %s' % (epoch, loss_cnn_test))
                # print('GRU子学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_gru_test))


