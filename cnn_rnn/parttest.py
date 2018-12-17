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

def variable_summaries(var, name):
    '''
    监控指标可视化函数
    :param var: Variable 类型变量
    :param name: 变量名
    :return: None
    '''
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

def SaveFile(data, p):
    '''
    存储整理好的数据
    :param data: 待存入该文件的数据
    :param p: 新生成的文件绝对路径
    :return: None
    '''
    if not os.path.exists(p):
        with open(p, 'wb') as file:
            pickle.dump(data, file)

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

def stacking_first_main():
    '''
    stacking策略初级学习器训练和交叉预测
    :param files: ParseDequeue函数所需参数
    :return: None
    '''

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
    tr_batch_size = 80
    tr_capacity = 400 + 40 * 40
    tr_batch_fun = tf.train.batch
    tr_batch_step = 1
    tr_files = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\outputtrain.tfrecords-*'

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
    te_batch_size = 80
    te_capacity = 400 + 40 * 40
    te_batch_fun = tf.train.batch
    te_batch_step = 1
    te_files = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\outputtest.tfrecords-*'

    with tf.name_scope('data_batch'):
        # 定义读取训练集数据对象
        train_fileoperation = FileoOperation(tr_p_in, tr_filename, tr_read_in_fun, tr_num_shards, tr_instance_per_shard,
                                             tr_ftype, tr_ttype, tr_fshape, tr_tshape, tr_batch_size, tr_capacity,
                                             tr_batch_fun, tr_batch_step)
        train_fileoperation.file2TFRecord()

        train_feature_batch, train_target_batch = train_fileoperation.ParseDequeue(tr_files)

        # 定义读取测试集数据对象
        test_fileoperation = FileoOperation(te_p_in, te_filename, te_read_in_fun, te_num_shards, te_instance_per_shard,
                                            te_ftype, te_ttype, te_fshape, te_tshape, te_batch_size, te_capacity,
                                            te_batch_fun, te_batch_step)
        test_fileoperation.file2TFRecord()

        test_feature_batch, test_target_batch = test_fileoperation.ParseDequeue(te_files)

    with tf.name_scope('x-y'):
        # 训练数据批次占位符,占位符读入数据形状和一个批次的数据特征矩阵形状相同
        x = tf.placeholder(dtype=tf.float32, shape=[tr_batch_size, tr_fshape-4])
        y = tf.placeholder(dtype=tf.float32, shape=[tr_batch_size, tr_tshape])

    ############################CNN############################
    #定义cnn子学习器中卷积核,全连接层参数矩阵以及偏置量尺寸
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

    #定义CNN类对象用以对数据Tensor进行改变形状
    cnn = CNN()
    #将每个样本特征向量由一维转化为二维shape= [batch_size, h, v, 1],原始数据有24个特征，转化为4*6维'picture'特征输入卷积神经网络
    x_cnn = cnn.d_one2d_two(x, 5, 4)

    with tf.name_scope('cnn_ops'):
        # 输出单个值的Variable
        cnn_ops = stacking_CNN(x= x_cnn, arg_dict= cnn_weights, keep_prob= 1.0)

    with tf.name_scope('cnn_optimize-loss'):
        # 定义CNN自学习器的损失函数和优化器
        cnn_optimize, cnn_loss = sub_LossOptimize(cnn_ops, y, optimize_function= tf.train.RMSPropOptimizer, learning_rate= 1e-4)

    ##############################RNN##############################
    #定义GRU子学习器中全连接层参数矩阵以及偏置量尺寸
    with tf.name_scope('gru_weights'):
        gru_weights = {
            'w_1': tf.Variable(tf.truncated_normal([256, 128], mean=0, stddev=1.0), dtype=tf.float32),
        # 256为GRU网络最终输出的隐藏层结点数量
            'w_2': tf.Variable(tf.truncated_normal([128, 64], mean=0, stddev=1.0), dtype=tf.float32),
            'b_1': tf.Variable(tf.truncated_normal([128], mean=0, stddev=1.0), dtype=tf.float32),
            'b_2': tf.Variable(tf.truncated_normal([64], mean=0, stddev=1.0), dtype=tf.float32)
        }

        # variable_summaries(gru_weights['w_1'], 'w_1')
        # variable_summaries(gru_weights['w_2'], 'w_2')
        # variable_summaries(gru_weights['b_1'], 'b_1')
        # variable_summaries(gru_weights['b_2'], 'b_2')

    with tf.name_scope('gru_ops'):
        # 定义多层GRU的最终输出ops
        gru_ops = stacking_GRU(x= x, num_units= 256, arg_dict=gru_weights)

    with tf.name_scope('gru_optimize-loss'):
        # 定义GRU自学习器的损失函数和优化器
        gru_optimize, gru_loss = sub_LossOptimize(gru_ops, y, optimize_function=tf.train.RMSPropOptimizer,
                                                  learning_rate= 1e-4)

    train_steps = tr_batch_step  # 对于stacking策略，使用5折交叉验证
    #交叉检验折数
    fold = 1
    #循环训练次数和总轮数
    epoch, loop = 1, 1000

    # 初级学习器预测得到的两个特征进行组合后作为次级学习器的特征向量，起始值为空
    super_tr_feature = np.array(None)
    # 初级学习器读取文件中训练集标签直接传到次级学习器中，起始值为空
    super_tr_target_batch_all = np.array(None)
    # 测试样本标签在初级学习器读取文件后直接传到次级学习器中，起始值为空
    super_te_target_batch_all = np.array(None)
    # 得到的次级学习器测试集特征取5次平均值
    super_te_feature_ave = np.array(None)



    # 摘要汇总
    # merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #############################Session###########################
    with tf.Session(config=tf.ConfigProto(gpu_options= gpu_options)) as sess:
        # 在使用tf.train。match_filenames_once函数时需要初始化一些变量
        sess.run(tf.local_variables_initializer())
        sess.run(init)
        # 摘要文件
        # summary_writer = tf.summary.FileWriter('logs/', sess.graph)
        # coord, threads = train_fileoperation.coord_threads(sess=sess)
        # 线程调配管理器
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # summary = sess.run(merged, feed_dict={x: train_feature_batch, y: train_target_batch})
        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                tr_feature_batch, tr_target_batch = sess.run([train_feature_batch, train_target_batch])
                # print(tr_feature_batch, tr_target_batch)
                if (train_steps % 5) != fold:
                    #######################
                    # 读入后20个特征
                    _ = sess.run(cnn_optimize, feed_dict={x: tr_feature_batch[:, 4:], y: tr_target_batch})
                    _ = sess.run(gru_optimize, feed_dict={x: tr_feature_batch[:, 4:], y: tr_target_batch})
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
                elif (train_steps == ((fold-1) * 5 * loop + fold)):  # 循环训练时只需在第一次提取训练集中待预测批次
                    # 将序号等于group的一折（批次）数据存入cross_tr_feature, cross_tr_target中
                    cross_tr_feature_batch, super_tr_target_batch = tr_feature_batch, tr_target_batch

                if not (epoch % 100):
                    loss_cnn = sess.run(cnn_loss, feed_dict={x: tr_feature_batch[:, 4:], y: tr_target_batch})
                    loss_gru = sess.run(gru_loss, feed_dict={x: tr_feature_batch[:, 4:], y: tr_target_batch})
                    print('CNN子学习器损失函在第 %s 个epoch的数值为: %s' % (epoch, loss_cnn))
                    print('GRU子学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_gru))

                    # 对1折测试集输出均方差
                    loss_cnn_test = sess.run(cnn_loss,
                                             feed_dict={x: cross_tr_feature_batch[:, 4:], y: super_tr_target_batch})
                    loss_gru_test = sess.run(gru_loss,
                                             feed_dict={x: cross_tr_feature_batch[:, 4:], y: super_tr_target_batch})
                    print('---CNN子学习器损失函在第 %s 个epoch的数值为: %s' % (epoch, loss_cnn_test))
                    print('---GRU子学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_gru_test))

                train_steps += 1
                #在一次五折交叉验证循环loop次后更新fold值
                if not (train_steps % (5 * loop)):
                    fold += 1

                    ################################对训练集数据进行次级特征预测##########################
                    # 输出特定批次在两个子学习器中的预测值，predict_cnn.shape= predict_gru.shape= [batch_size, 1]
                    ######################
                    # 读入后20个特征
                    predict_cnn, predict_gru = sess.run([cnn_ops, gru_ops],
                                                        feed_dict={x: cross_tr_feature_batch[:, 4:]})

                    # (读入后20个特征情况)组合特征得到次级学习器训练集特征矩阵（需要存储至新文件）
                    super_tr_feature = np.hstack((predict_cnn, predict_gru, cross_tr_feature_batch[:, :5])) \
                        if super_tr_feature.any() == None else \
                        np.vstack((super_tr_feature, np.hstack((predict_cnn, predict_gru, cross_tr_feature_batch[:, :5]))))
                    print('正在预测第 %s 折样本的部分次级学习特征值' % fold)
                    ######################
                    ######################
                    # 读入全部特征
                    # predict_cnn, predict_gru = sess.run([cnn_ops, gru_ops], feed_dict={x: cross_tr_feature_batch})

                    # # (读入全部特征情况)组合特征得到次级学习器训练集特征矩阵（需要存储至新文件）
                    # super_tr_feature = np.hstack((predict_cnn, predict_gru)) if super_tr_feature.any() == None else \
                    #     np.vstack((super_tr_feature, np.hstack((predict_cnn, predict_gru))))
                    #####################

                    # 组合标签得到次级学习器训练集标签矩阵（需要存储至新文件）
                    super_tr_target_batch_all = super_tr_target_batch if super_tr_target_batch_all.any() == None else \
                        np.vstack((super_tr_target_batch_all, super_tr_target_batch))

                    ############################对测试集数据进行初级预测得到次级特征#########################
                    # test_steps 为测试集按照每个批次和训练集相同大小时一共分几组读入
                    test_steps = 2

                    # 测试样本经过初级学习器预测后得到两个特征组合为次级学习器所需特征向量，起始值为空
                    super_te_feature = np.array(None)

                    for i in range(test_steps):
                        # 读取测试集一个批次的特征和标签
                        te_feature_batch, te_target_batch = sess.run([test_feature_batch, test_target_batch])

                        # 对原始测试集应用训练好的两个子学习器做两个次级学习器所需特征预测，并组合为特征向量
                        #######################
                        # 读入后20个特征情况
                        te_sufeature_cnn, te_sufeature_gru = sess.run([cnn_ops, gru_ops],
                                                                      feed_dict={x: te_feature_batch[:, 4:]})

                        # (读入后20个特征情况)组合特征得到次级学习器测试集特征矩阵(取5次预测的平均值)
                        super_te_feature = np.hstack((te_sufeature_cnn, te_sufeature_gru, te_feature_batch[:, :5])) \
                            if super_te_feature.any() == None else np.vstack((super_te_feature,
                                                                              np.hstack((te_sufeature_cnn,
                                                                                         te_sufeature_gru,
                                                                                         te_feature_batch[:, :5]))))
                        #######################
                        #######################
                        # 读入所有特征情况
                        # te_sufeature_cnn, te_sufeature_gru = sess.run([cnn_ops, gru_ops], feed_dict={x: te_feature_batch})
                        #
                        # # (读入所有特征情况)组合特征得到次级学习器测试集特征矩阵(取5次预测的平均值)
                        # super_te_feature = np.hstack((te_sufeature_cnn, te_sufeature_gru)) if super_te_feature.any() == None else \
                        #     np.vstack((super_tr_feature, np.hstack((te_sufeature_cnn, te_sufeature_gru))))
                        #######################

                        if fold == 2:
                            # 组合标签得到次级学习器测试集标签矩阵（需要存储至新文件）
                            super_te_target_batch_all = te_target_batch if super_te_target_batch_all.any() == None else \
                                np.vstack((super_te_target_batch_all, te_target_batch))


                    # 次级学习器测试集特征矩阵，递归取平均（需要存储至新文件）
                    super_te_feature_ave = super_te_feature if super_te_feature_ave.any() == None else \
                        ((fold - 1) * super_te_feature_ave + super_te_feature) / fold



                # 在train_steps为5的倍数时更新
                if train_steps % 5 == 0:
                    epoch += 1

                #fold更新至5且最后一次五折交叉验证完成后请求所有线程结束
                if train_steps > 5*5*loop:
                    coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

        except tf.errors.OutOfRangeError:
            print('初级学习器训练结束!')
        finally:
            # When done, ask the threads to stop. 请求该线程停止
            coord.request_stop()
            # And wait for them to actually do it. 等待被指定的线程终止
            coord.join(threads)

            # summary_writer.add_summary(summary, epoch)

            # 关闭摘要
            # summary_writer.close()

            #存储计算图为pb格式
            # Replaces all the variables in a graph with constants of the same values
            # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])
            # 写入序列化的pb文件
            # with tf.gfile.FastGFile(pb_file_path + 'model.pb', mode='wb') as f:
            #     f.write(constant_graph.SerializeToString())

            # Builds the SavedModel protocol buffer and saves variables and assets
            # 在和project相同层级目录下产生带有savemodel名称的文件夹
            # builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path + 'savemodel')
            # Adds the current meta graph to the SavedModel and saves variables
            # 第二个参数为字符列表形式的tags – The set of tags with which to save the meta graph
            # builder.add_meta_graph_and_variables(sess, ['cpu_server_1'])
            # Writes a SavedModel protocol buffer to disk
            # 此处p值为生成的文件夹路径
            # p = builder.save()


    #将次级学习器训练集和测试集特征和标签合并存入各自的文件
    train_database = np.hstack((super_tr_feature, super_tr_target_batch_all))
    print(super_te_feature_ave.shape, super_te_target_batch_all.shape)
    test_database = np.hstack((super_te_feature_ave, super_te_target_batch_all))
    #文件路径
    p_train = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\cnn_rnn\TRAIN.pickle'
    p_test = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\cnn_rnn\TEST.pickle'
    SaveFile(train_database, p_train)
    SaveFile(test_database, p_test)

    # 关闭摘要
    # summary_writer.close()

def fun(p):
    '''不对数据做任何处理'''
    return np.arange(400*24, dtype= np.float64).reshape(400, 24), np.arange(400, dtype= np.float64)


if __name__ == '__main__':
    stacking_first_main()



