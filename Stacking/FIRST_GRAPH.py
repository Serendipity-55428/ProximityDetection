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
from Routine_operation import SaveFile, LoadFile, Summary_Visualization, SaveImport_model, SaveRestore_model
import tensorflow as tf
import numpy as np
import os
import pickle

#cnn模块################################################################
def cnn_mode(training_time, is_finishing):
    '''
    cnn计算图
    :return: None
    '''
    CNN_graph = tf.Graph()
    with CNN_graph.as_default():
        summary_visualization = Summary_Visualization()
        with tf.name_scope('placeholder'):
            x = tf.placeholder(dtype=tf.float32, shape=(None, 25), name='xt')
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
        with tf.name_scope('w'):
            # 核张量
            kernel_para = {
                'w1_size': tf.Variable(initial_value= tf.truncated_normal(
                    shape= (kernel_size['w1_edge'], kernel_size['w1_edge'], 1, kernel_size['w1_deep']),
                    mean= 0, stddev= 1), dtype= tf.float32),
                'w2_size': tf.Variable(initial_value= tf.truncated_normal(
                    shape= (kernel_size['w2_edge'], kernel_size['w2_edge'], kernel_size['w1_deep'], kernel_size['w2_deep']),
                    mean= 0, stddev= 1), dtype= tf.float32),
                'w3_size': tf.Variable(initial_value=tf.truncated_normal(
                    shape=(kernel_size['w3_edge'], kernel_size['w3_edge'], kernel_size['w2_deep'], kernel_size['w3_deep']),
                    mean=0, stddev=1), dtype=tf.float32),
                'w4_size': tf.Variable(initial_value=tf.truncated_normal(
                    shape=(kernel_size['w4_edge'], kernel_size['w4_edge'], kernel_size['w3_deep'], kernel_size['w4_deep']),
                    mean=0, stddev=1), dtype=tf.float32),
                'w5_size': tf.Variable(initial_value=tf.truncated_normal(
                    shape=(kernel_size['w5_edge'], kernel_size['w5_edge'], kernel_size['w4_deep'], kernel_size['w5_deep']),
                    mean=0, stddev=1), dtype=tf.float32),
                'w6_size': tf.Variable(initial_value=tf.truncated_normal(
                    shape=(kernel_size['w6_edge'], kernel_size['w6_edge'], kernel_size['w5_deep'], kernel_size['w6_deep']),
                    mean=0, stddev=1), dtype=tf.float32),
            }
            #对所有卷积核w和b写入文件摘要
            for i in (i for i in range(6)):
                summary_visualization.variable_summaries(var= kernel_para['w%s_size' % (i+1)], name= 'w%s' % (i+1))

        # 卷积层(数据特征维度：20->5*5)
        with tf.name_scope('cnn'):
            cnn_1 = CNN(x=x, w_conv=kernel_para['w1_size'], stride_conv=1, stride_pool=2)
            # 将向量x转换为5*5方形张量
            x_reshape = CNN.reshape(f_vector=x, new_shape=(-1, 5, 5, 1))
            # 1
            layer_1 = cnn_1.convolution(input=x_reshape)
            relu1 = tf.nn.relu(layer_1)
            bn1 = cnn_1.batch_normoalization(input=relu1, is_training=bn_istraining)
            # 2
            cnn_2 = CNN(x=bn1, w_conv=kernel_para['w2_size'], stride_conv=1, stride_pool=2)
            layer_2 = cnn_2.convolution(input=bn1)
            relu2 = tf.nn.relu(layer_2)
            bn2 = cnn_2.batch_normoalization(input=relu2, is_training=bn_istraining)
            # 3
            cnn_3 = CNN(x=bn2, w_conv=kernel_para['w3_size'], stride_conv=1, stride_pool=2)
            layer_3 = cnn_3.convolution(input=bn2)
            relu3 = tf.nn.relu(layer_3)
            bn3 = cnn_3.batch_normoalization(input=relu3, is_training=bn_istraining)
            # pool
            pool1 = cnn_3.pooling(pool_fun=tf.nn.max_pool, input=bn3)
            # 4
            cnn_4 = CNN(x=pool1, w_conv=kernel_para['w4_size'], stride_conv=1, stride_pool=2)
            layer_4 = cnn_4.convolution()
            relu4 = tf.nn.relu(layer_4)
            bn4 = cnn_4.batch_normoalization(input=relu4, is_training=bn_istraining)
            # 5
            cnn_5 = CNN(x=bn4, w_conv=kernel_para['w5_size'], stride_conv=1, stride_pool=2)
            layer_5 = cnn_5.convolution()
            relu5 = tf.nn.relu(layer_5)
            bn5 = cnn_5.batch_normoalization(input=relu5, is_training=bn_istraining)
            # pool
            pool2 = cnn_5.pooling(pool_fun=tf.nn.max_pool, input=bn5)
            # 6
            cnn_6 = CNN(x=pool2, w_conv=kernel_para['w6_size'], stride_conv=1, stride_pool=2)
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
        with tf.name_scope('cnn_output'):
            # 预测半径
            r_cnn = tf.matmul(cnn_output, w_cnn) + b_cnn
            relu_r_cnn = tf.nn.relu(r_cnn) #此处可以根据实验结果去掉或者添加relu，如果去掉，pb保存代码中相应也需要修改
        with tf.name_scope('loss-optimize-evaluation-acc'):
            # 代价函数、优化函数、评价指标
            loss_cnn = tf.reduce_mean(tf.square(relu_r_cnn - y))
            # 添加摘要loss
            summary_visualization.scalar_summaries(arg={'loss': loss_cnn})
            optimize_cnn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_cnn).minimize(loss_cnn)
            evaluation_cnn = Evaluation(one_hot=False, logit=None, label=None, regression_label=y,
                                        regression_pred=relu_r_cnn)
            acc_cnn = evaluation_cnn.acc_regression(Threshold=0.1)  # 评价指标中的阈值可以修改

        # 摘要汇总
        merge = summary_visualization.summary_merge()
        # 变量初始化节点,显卡占用率分配节点
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=CNN_graph) as sess:
        sess.run(init)
        # 建立checkpoint节点保存对象
        saverestore_model = SaveRestore_model(sess=sess, save_file_name='cnn', max_to_keep=1)
        saver = saverestore_model.saver_build()
        if training_time != 0:
            # 导入checkpoint节点，继续训练
            saverestore_model.restore_checkpoint(saver=saver)
        # 摘要文件
        summary_writer = summary_visualization.summary_file(p='logs/', graph=sess.graph)
        # 导入数据
        p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data.pickle'
        dataset_ori = LoadFile(p=p_dataset_ori)
        # 记录折数
        fold = 0
        # 总训练轮数
        epoch_all = 1
        # 设定所有折在cnn子学习器最终的预测值
        first_sub_cnn_pred = np.zeros(dtype=np.float32, shape=([1]))
        # 将数据集划分为训练集和测试集
        for train, test in data_stepone(p_dataset_ori=p_dataset_ori, padding=True, proportion=5):
            for epoch in range(epoch_all):
                # 设定标志在100的倍数epoch时只输出一次结果
                flag = 1
                # 以一定批次读入某一折数据进行训练
                for batch_x, batch_y in data_steptwo(train_data=train, batch_size=500):
                    # 所有训练数据每折各个批次的模型参数摘要汇总
                    summary = sess.run(merge, feed_dict={x: batch_x, y: batch_y[:, np.newaxis], bn_istraining: True,
                                                         learning_rate_cnn: 1e-4})
                    _ = sess.run(optimize_cnn,
                                 feed_dict={x: batch_x, y: batch_y[:, np.newaxis], bn_istraining: True, learning_rate_cnn: 1e-4})
                    summary_visualization.add_summary(summary_writer=summary_writer, summary=summary,
                                                      summary_information=epoch)
                    if (epoch % 100) == 0 and flag == 1:
                        loss_cnn_ = sess.run(loss_cnn, feed_dict={x: batch_x, y: batch_y[:, np.newaxis], bn_istraining: True})
                        acc_cnn_ = sess.run(acc_cnn, feed_dict={x: test[:, :-1], y: test[:, -1][:, np.newaxis], bn_istraining: False})
                        print('第%s轮后训练集损失为: %s, 第 %s 折预测准确率为: %s' % (epoch, loss_cnn_, fold, acc_cnn_))
                        flag = 0
                # 保存checkpoint节点
                saverestore_model.save_checkpoint(saver=saver, epoch=epoch, is_recording_max_acc=False)

            # 记录循环多次后该折子学习器最终预测结果
            first_sub_cnn_pred = acc_cnn_ if first_sub_cnn_pred.any() == 0 else np.vstack(
                (first_sub_cnn_pred, acc_cnn_))
            fold += 1
        summary_visualization.summary_close(summary_writer=summary_writer)
        if is_finishing:
            # 将最终训练好的模型保存为pb文件
            savemodel = SaveImport_model(sess_ori=sess, file_suffix='\\cnn_model', ops=(relu_r_cnn, x, bn_istraining),
                                         usefulplaceholder_count=2)
            savemodel.save_pb()

    p_cnn = r'F:\ProximityDetection\Stacking\cnn_pred.pickle'
    SaveFile(data=first_sub_cnn_pred, savepickle_p=p_cnn)

#rnn模块################################################################
def rnn_mode(training_time, is_finishing):
    '''
    rnn计算图
    :return: None
    '''
    RNN_graph = tf.Graph()
    with RNN_graph.as_default():
        summary_visualization = Summary_Visualization()
        with tf.name_scope('placeholder'):
            x = tf.placeholder(dtype=tf.float32, shape=(None, 20), name='xt')
            y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='yt')
            learning_rate_rnn = tf.placeholder(dtype=tf.float32, name='learning_rate_rnn')
        with tf.name_scope('rnn'):
            rnn = RNN(x=x, max_time=5, num_units=128)
            rnn_outputs, _ = rnn.dynamic_rnn(style='LSTM', output_keep_prob=0.8)
            rnn_output = rnn_outputs[:, -1, :]
        with tf.name_scope('rnn-output'):
            # rnn投影核
            w_rnn = tf.Variable(initial_value=tf.truncated_normal(shape=(rnn_output.get_shape().as_list()[-1], 1),
                                                                  mean=0, stddev=1), dtype=tf.float32, name='w_rnn')
            b_rnn = tf.Variable(initial_value=tf.truncated_normal(shape=([1]), mean=0, stddev=1), dtype=tf.float32,
                                name='b_rnn')
            #对w和b参数写入文件摘要
            summary_visualization.variable_summaries(var= w_rnn, name= 'w')
            summary_visualization.variable_summaries(var= b_rnn, name= 'b')
            # 预测半径
            r_rnn = tf.matmul(rnn_output, w_rnn) + b_rnn
            relu_r_rnn = tf.nn.relu(r_rnn) #此处可以根据实验结果添加或者去掉relu，如果去掉，需要在pb保存时变换相应的保存节点
        with tf.name_scope('loss-optimize-evaluation-acc'):
            # 代价函数、优化函数、评价指标
            loss_rnn = tf.reduce_mean(tf.square(relu_r_rnn - y))
            # 添加摘要loss
            summary_visualization.scalar_summaries(arg={'loss': loss_rnn})
            optimize_rnn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_rnn).minimize(loss_rnn)
            evaluation_rnn = Evaluation(one_hot=False, logit=None, label=None, regression_label=y,
                                        regression_pred=relu_r_rnn)
            acc_rnn = evaluation_rnn.acc_regression(Threshold=0.1)  # 评价指标中的阈值可以修改

        # 摘要汇总
        merge = summary_visualization.summary_merge()
        # 变量初始化节点,显卡占用率分配节点
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=RNN_graph) as sess:
        sess.run(init)
        # 建立checkpoint节点保存对象
        saverestore_model = SaveRestore_model(sess=sess, save_file_name='rnn', max_to_keep=1)
        saver = saverestore_model.saver_build()
        if training_time != 0:
            # 导入checkpoint节点，继续训练
            saverestore_model.restore_checkpoint(saver=saver)
        # 摘要文件
        summary_writer = summary_visualization.summary_file(p='logs/', graph=sess.graph)
        # 导入数据
        p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data.pickle'
        dataset_ori = LoadFile(p=p_dataset_ori)
        # 记录折数
        fold = 0
        # 总训练轮数
        epoch_all = 1
        # 设定所有折在cnn子学习器最终的预测值
        first_sub_rnn_pred = np.zeros(dtype=np.float32, shape=([1]))
        # 将数据集划分为训练集和测试集
        for train, test in data_stepone(p_dataset_ori=p_dataset_ori, padding=False, proportion=5):
            for epoch in range(epoch_all):
                # 设定标志在100的倍数epoch时只输出一次结果
                flag = 1
                # 以一定批次读入某一折数据进行训练
                for batch_x, batch_y in data_steptwo(train_data=train, batch_size=500):
                    # 所有训练数据每折各个批次的模型参数摘要汇总
                    summary = sess.run(merge, feed_dict={x: batch_x, y: batch_y[:, np.newaxis], learning_rate_rnn: 1e-4})
                    _ = sess.run(optimize_rnn, feed_dict={x: batch_x, y: batch_y[:, np.newaxis], learning_rate_rnn: 1e-4})
                    summary_visualization.add_summary(summary_writer=summary_writer, summary=summary,
                                                      summary_information=epoch)
                    if (epoch % 100) == 0 and flag == 1:
                        loss_rnn_ = sess.run(loss_rnn, feed_dict={x: batch_x, y: batch_y[:, np.newaxis]})
                        acc_rnn_ = sess.run(acc_rnn, feed_dict={x: test[:, :-1], y: test[:, -1][:, np.newaxis]})
                        print('第%s轮后训练集损失为: %s, 第 %s 折预测准确率为: %s' % (epoch, loss_rnn_, fold, acc_rnn_))
                        flag = 0
                # 保存checkpoint节点
                saverestore_model.save_checkpoint(saver=saver, epoch=epoch, is_recording_max_acc=False)

            # 记录循环多次后该折子学习器最终预测结果
            first_sub_rnn_pred = acc_rnn_ if first_sub_rnn_pred.any() == 0 else np.vstack(
                (first_sub_rnn_pred, acc_rnn_))
            fold += 1
        summary_visualization.summary_close(summary_writer=summary_writer)
        if is_finishing:
            # 将最终训练好的模型保存为pb文件
            savemodel = SaveImport_model(sess_ori=sess, file_suffix='\\rnn_model', ops=(relu_r_rnn, x),
                                         usefulplaceholder_count=1)
            savemodel.save_pb()


    p_rnn = r'F:\ProximityDetection\Stacking\rnn_pred.pickle'
    SaveFile(data=first_sub_rnn_pred, savepickle_p=p_rnn)

#fnn模块################################################################
def fnn_mode(training_time, is_finishing):
    '''
    fnn计算图
    :param training_time: int, 计算图间断训练次数
    :param is_finishing: bool, 是否结束训练并输出pb文件
    :return: None
    '''
    FNN_graph = tf.Graph()
    with FNN_graph.as_default():
        summary_visualization = Summary_Visualization()
        with tf.name_scope('placeholder'):
            x = tf.placeholder(dtype=tf.float32, shape=(None, 6), name='xt')
            y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='yt')
            learning_rate_fnn = tf.placeholder(dtype=tf.float32, name='learning_rate_fnn')

        h_size = {
            'w1_insize': 6,
            'w1_outsize': 100,
            'w2_insize': 100,
            'w2_outsize': 200,
            'w3_insize': 200,
            'w3_outsize': 1
        }

        with tf.name_scope('w-b'):
            h_para = (
                (tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w1_insize'], h_size['w1_outsize']]),
                                                               mean=0, stddev=1), dtype=tf.float32, name='w1'),
                 tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w1_outsize']]),
                                                               mean=0, stddev=1), dtype=tf.float32, name='b1')),
                (tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w2_insize'], h_size['w2_outsize']]),
                                                               mean=0, stddev=1), dtype=tf.float32, name='w2'),
                 tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w2_outsize']]),
                                                               mean=0, stddev=1), dtype=tf.float32, name='b2')),
                (tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w3_insize'], h_size['w3_outsize']]),
                                                               mean=0, stddev=1), dtype=tf.float32, name='w3'),
                 tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w3_outsize']]),
                                                               mean=0, stddev=1), dtype=tf.float32, name='b3')),
            )
            #对所有连接参数和偏置量制作摘要
            for i in (i for i in range(3)):
                w, b = h_para[i]
                summary_visualization.variable_summaries(var= w, name= 'w%s' % (i+1))
                summary_visualization.variable_summaries(var= b, name= 'b%s' % (i+1))


        fnn = FNN(x= x, w= h_para)
        with tf.name_scope('fc_output'):
            fc_output = fnn.fc_concat(keep_prob= 0.8)
        with tf.name_scope('loss-potimize-evaluation-acc'):
            # 代价函数、优化函数、评价指标
            loss_fnn = tf.reduce_mean(tf.square(fc_output - y))
            #添加摘要loss
            summary_visualization.scalar_summaries(arg= {'loss': loss_fnn})
            optimize_fnn = tf.train.AdamOptimizer(learning_rate=learning_rate_fnn).minimize(loss_fnn)
            evaluation_fnn = Evaluation(one_hot=False, logit=None, label=None, regression_label=y,
                                        regression_pred=fc_output)
            acc_fnn = evaluation_fnn.acc_regression(Threshold=0.9)  # 评价指标中的阈值可以修改
            #添加摘要acc
            # summary_visualization.scalar_summaries(arg= {'acc': acc_fnn})

        #摘要汇总
        merge = summary_visualization.summary_merge()
        # 变量初始化节点,显卡占用率分配节点
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=FNN_graph) as sess:
        sess.run(init)
        # 建立checkpoint节点保存对象
        saverestore_model = SaveRestore_model(sess=sess, save_file_name='fnn', max_to_keep=1)
        saver = saverestore_model.saver_build()
        if training_time != 0:
            # 导入checkpoint节点，继续训练
            saverestore_model.restore_checkpoint(saver=saver)
        #摘要文件
        summary_writer = summary_visualization.summary_file(p= 'logs/', graph= sess.graph)
        # 导入数据
        p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data_2.pickle'
        # 记录折数
        fold = 0
        # 总训练轮数
        epoch_all = 1
        # 将数据集划分为训练集和测试集
        for train, test in data_stepone(p_dataset_ori=p_dataset_ori, padding=False, proportion=5):
            for epoch in range(epoch_all):
                # 设定标志在100的倍数epoch时只输出一次结果
                flag = 1
                # 以一定批次读入某一折数据进行训练
                for batch_x, batch_y in data_steptwo(train_data=train, batch_size=1000):
                    #所有训练数据每折各个批次的模型参数摘要汇总
                    summary = sess.run(merge, feed_dict= {x: batch_x, y: batch_y[:, np.newaxis], learning_rate_fnn: 1e-4})
                    _ = sess.run(optimize_fnn, feed_dict={x: batch_x, y: batch_y[:, np.newaxis], learning_rate_fnn: 1e-4})
                    summary_visualization.add_summary(summary_writer= summary_writer, summary= summary,
                                                      summary_information= epoch)
                    if (epoch % 100) == 0 and flag == 1:
                        loss_fnn_ = sess.run(loss_fnn, feed_dict={x: batch_x, y: batch_y[:, np.newaxis]})
                        acc_fnn_ = sess.run(acc_fnn, feed_dict={x: test[:, :-1], y: test[:, -1][:, np.newaxis]})
                        acc_fnn_train = sess.run(acc_fnn, feed_dict={x:train[:, :-1], y:train[:, -1][:, np.newaxis]})
                        print('训练集精度为: %s' % acc_fnn_train)
                        print('第%s轮后训练集损失为: %s, 第 %s 折预测准确率为: %s' % (epoch, loss_fnn_, fold, acc_fnn_))
                        flag = 0
                #保存checkpoint节点
                saverestore_model.save_checkpoint(saver= saver, epoch= epoch, is_recording_max_acc= False)


            fold += 1
        summary_visualization.summary_close(summary_writer= summary_writer)
        if is_finishing:
            # 将最终训练好的模型保存为pb文件
            savemodel = SaveImport_model(sess_ori=sess, file_suffix='\\fc_model', ops=(fc_output, x),
                                         usefulplaceholder_count=1)
            savemodel.save_pb()


if __name__ == '__main__':
    cnn_mode(training_time= 0, is_finishing= False)
    # rnn_mode(training_time= 1, is_finishing= False)
    # fnn_mode(training_time= 3, is_finishing= False)