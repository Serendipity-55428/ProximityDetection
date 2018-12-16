#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: SecondLearner
@time: 2018/12/13 17:10
@desc:
'''

import tensorflow as tf
import numpy as np
from cnn_rnn.FirstLearner import variable_summaries, sub_LossOptimize
from cnn_rnn.HyMultiNN import RecurrentNeuralNetwork, FCNN, CNN
from cnn_rnn.Fmake2read import FileoOperation
from cnn_rnn.sub_learning import stacking_CNN, stacking_GRU, stacking_FC
import time

def stacking_second_main():
    '''
    stacking策略次级学习器训练和交叉预测
    :param files: ParseDequeue函数所需参数
    :return: None
    '''
    # 训练集数据所需参数
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

    # 测试集数据所需参数
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

    #############################FC###############################
    # 定义fc次级学习器中全连接层参数
    with tf.name_scope('fc_weights'):
        fc_weights = {
            'w_sub_1': tf.Variable(tf.truncated_normal([2, 128], mean=0, stddev=1.0), dtype=tf.float32),
            'w_sub_2': tf.Variable(tf.truncated_normal([128, 64], mean=0, stddev=1.0), dtype=tf.float32),
            'w_sub_3': tf.Variable(tf.truncated_normal([64, 1], mean=0, stddev=1.0), dtype=tf.float32),
            'b_sub_1': tf.Variable(tf.truncated_normal([128], mean=0, stddev=1.0), dtype=tf.float32),
            'b_sub_2': tf.Variable(tf.truncated_normal([64], mean=0, stddev=1.0), dtype=tf.float32),
            'b_sub_3': tf.Variable(tf.truncated_normal([1], mean=0, stddev=1.0), dtype=tf.float32)
        }

        variable_summaries(fc_weights['w_sub_1'], 'w_sub_1')
        variable_summaries(fc_weights['w_sub_2'], 'w_sub_2')
        variable_summaries(fc_weights['w_sub_3'], 'w_sub_3')
        variable_summaries(fc_weights['b_sub_1'], 'b_sub_1')
        variable_summaries(fc_weights['b_sub_2'], 'b_sub_2')
        variable_summaries(fc_weights['b_sub_3'], 'b_sub_3')

    with tf.name_scope('fc_ops'):
        # 定义FC次级学习器的最终输出ops
        fc_ops = stacking_FC(x=x, arg_dict=fc_weights)
    with tf.name_scope('fc_optimize-loss'):
        # 定义次级学习器的损失函数和优化器
        fc_optimize, fc_loss = sub_LossOptimize(fc_ops, y, optimize_function=tf.train.RMSPropOptimizer,
                                                learning_rate=1e-4)

    # 摘要汇总
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #############################Session###########################
    with tf.Session(config=tf.ConfigProto(gpu_options= gpu_options)) as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # 摘要文件
        summary_writer = tf.summary.FileWriter('logs/', sess.graph)

        # 线程调配管理器
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_steps = tr_batch_step  # 对于stacking策略，使用5折交叉验证，该参数设置为4（5折，计数从0开始）
        test_steps = te_batch_step  # 子学习器中测试集分批次预测

        for epoch in range(100000):
            summary = sess.run(merged, feed_dict={x: tr_feature_batch, y: tr_target_batch})
            try:
                while not coord.should_stop():  # 如果线程应该停止则返回True
                    #批量读取次级学习器训练集特征和标签数据
                    tr_feature_batch, tr_target_batch = sess.run(train_feature_batch, train_target_batch)
                    _, loss_fc = sess.run([fc_optimize, fc_loss], feed_dict= {x: tr_feature_batch, y: tr_target_batch})
                    print('FC次级学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_fc))

                    test_steps -= 1
                    if test_steps <= 0:
                        coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

            except tf.errors.OutOfRangeError:
                print('第 %s 轮训练结束' % epoch)
            finally:
                # When done, ask the threads to stop. 请求该线程停止
                coord.request_stop()
                # And wait for them to actually do it. 等待被指定的线程终止
                coord.join(threads)

                summary_writer.add_summary(summary, epoch)

            try:
                while not coord.should_stop():  # 如果线程应该停止则返回True
                    #批量读取次级学习器测试集特征和标签数据
                    te_feature_batch, te_target_batch = sess.run(test_feature_batch, test_target_batch)
                    loss_fc = sess.run(fc_loss, feed_dict= {x: tr_feature_batch, y: tr_target_batch})
                    print('FC次级学习器预测损失在第 %s 个epoch的数值为: %s' % (epoch, loss_fc))

                    test_steps -= 1
                    if test_steps <= 0:
                        coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

            except tf.errors.OutOfRangeError:
                print('第 %s 轮测试结束' % epoch)
            finally:
                # When done, ask the threads to stop. 请求该线程停止
                coord.request_stop()
                # And wait for them to actually do it. 等待被指定的线程终止
                coord.join(threads)

        # 关闭摘要
        summary_writer.close()

if __name__ == '__main__':
    stacking_second_main()





