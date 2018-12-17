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
from cnn_rnn.FirstLearner import variable_summaries, sub_LossOptimize, coord_threads
from cnn_rnn.HyMultiNN import RecurrentNeuralNetwork, FCNN, CNN
from cnn_rnn.Fmake2read import FileoOperation
from cnn_rnn.sub_learning import stacking_CNN, stacking_GRU, stacking_FC
import time
import pickle

def LoadFile(p):
    '''读取文件'''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data[:, :7], data[:, -1]

def stacking_second_main():
    '''
    stacking策略次级学习器训练和交叉预测
    :param files: ParseDequeue函数所需参数
    :return: None
    '''
    # 训练集数据所需参数
    tr_p_in = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\cnn_rnn\TRAIN.pickle'
    tr_filename = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\train.tfrecords-%.5d-of-%.5d'
    tr_read_in_fun = LoadFile
    tr_num_shards = 5
    tr_instance_per_shard = 80
    tr_ftype = tf.float64
    tr_ttype = tf.float64
    tr_fshape = 6
    tr_tshape = 1
    tr_batch_size = 80
    tr_capacity = 400 + 40 * 40
    tr_batch_fun = tf.train.batch
    tr_batch_step = 1
    tr_files = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\train.tfrecords-*'

    # 测试集数据所需参数
    te_p_in = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\cnn_rnn\TEST.pickle'
    te_filename = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\test.tfrecords-%.5d-of-%.5d'
    te_read_in_fun = LoadFile
    te_num_shards = 5
    te_instance_per_shard = 80
    te_ftype = tf.float64
    te_ttype = tf.float64
    te_fshape = 6
    te_tshape = 1
    te_batch_size = 80
    te_capacity = 400 + 40 * 40
    te_batch_fun = tf.train.batch
    te_batch_step = 1
    te_files = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\test.tfrecords-*'

    with tf.name_scope('data_batch'):
        # 定义读取训练集数据对象
        train_fileoperation = FileoOperation(tr_p_in, tr_filename, tr_read_in_fun, tr_num_shards, tr_instance_per_shard,
                                             tr_ftype, tr_ttype, tr_fshape, tr_tshape, tr_batch_size, tr_capacity,
                                             tr_batch_fun, tr_batch_step)

        train_feature_batch, train_target_batch = train_fileoperation.ParseDequeue(tr_files)

        # 定义读取测试集数据对象
        test_fileoperation = FileoOperation(te_p_in, te_filename, te_read_in_fun, te_num_shards, te_instance_per_shard,
                                            te_ftype, te_ttype, te_fshape, te_tshape, te_batch_size, te_capacity,
                                            te_batch_fun, te_batch_step)

        test_feature_batch, test_target_batch = test_fileoperation.ParseDequeue(te_files)

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

        # variable_summaries(fc_weights['w_sub_1'], 'w_sub_1')
        # variable_summaries(fc_weights['w_sub_2'], 'w_sub_2')
        # variable_summaries(fc_weights['w_sub_3'], 'w_sub_3')
        # variable_summaries(fc_weights['b_sub_1'], 'b_sub_1')
        # variable_summaries(fc_weights['b_sub_2'], 'b_sub_2')
        # variable_summaries(fc_weights['b_sub_3'], 'b_sub_3')

    with tf.name_scope('fc_ops'):
        # 定义FC次级学习器的最终输出ops
        fc_ops = stacking_FC(x= x, arg_dict= fc_weights)
    with tf.name_scope('fc_optimize-loss'):
        # 定义次级学习器的损失函数和优化器
        fc_optimize, fc_loss = sub_LossOptimize(fc_ops, y, optimize_function= tf.train.RMSPropOptimizer,
                                                learning_rate=1e-4)

    train_steps = tr_batch_step  # 对于stacking策略，使用5折交叉验证
    # 交叉检验折数
    fold = 1
    # 循环训练次数和总轮数
    epoch, loop = 1, 1000

    # 摘要汇总
    # merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #############################Session###########################
    with tf.Session(config=tf.ConfigProto(gpu_options= gpu_options)) as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # 摘要文件
        # summary_writer = tf.summary.FileWriter('logs/', sess.graph)

        # 线程调配管理器
        coord, threads = coord_threads(sess= sess)

        # summary = sess.run(merged, feed_dict={x: tr_feature_batch, y: tr_target_batch})
        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                # 批量读取次级学习器训练集特征和标签数据
                tr_feature_batch, tr_target_batch = sess.run(train_feature_batch, train_target_batch)
                _ = sess.run(fc_optimize, feed_dict={x: tr_feature_batch, y: tr_target_batch})
                if not (epoch % 100):
                    loss_fc = sess.run(fc_loss, feed_dict={x: tr_feature_batch, y: tr_target_batch})
                    print('FC次级学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_fc))

                # _, loss_fc = sess.run([fc_optimize, fc_loss], feed_dict= {x: tr_feature_batch, y: tr_target_batch})
                # print('FC次级学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_fc))

                train_steps += 1

                # 在train_steps为5的倍数时更新
                if train_steps % 5 == 0:
                    epoch += 1

                if train_steps > 5*5*loop:
                    coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

        except tf.errors.OutOfRangeError:
            print('次级训练器训练结束!')
        finally:
            # When done, ask the threads to stop. 请求该线程停止
            coord.request_stop()
            # And wait for them to actually do it. 等待被指定的线程终止
            coord.join(threads)

            # summary_writer.add_summary(summary, epoch)




            #更新线程调配器
            coord, threads = coord_threads(sess= sess)

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
        # summary_writer.close()

if __name__ == '__main__':
    stacking_second_main()





