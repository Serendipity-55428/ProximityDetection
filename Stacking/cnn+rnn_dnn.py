#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: cnn+rnn_dnn
@time: 2019/3/20 22:30
@desc:
'''
from Stacking.AllNet import CNN, RNN, FNN
from Stacking.TestEvaluation import Evaluation
from Stacking.DataGenerate import data_stepone, data_steptwo, second_dataset
from Stacking.Routine_operation import SaveFile, LoadFile, Summary_Visualization, SaveImport_model, SaveRestore_model
import tensorflow as tf
import numpy as np

def cnn(x, is_training, summary_visualization):
    '''
    卷积网络部分
    :param x: placeholder, 数据特征, 维度为以为向量
    :param is_training: placeholder, 标记是否正在训练模型中
    :param summary_visualization: 摘要类型对象
    :return: cnn网络返回节点
    '''
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
            'w1_size': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w1_edge'], kernel_size['w1_edge'], 1, kernel_size['w1_deep']),
                mean=0, stddev=1), dtype=tf.float32),
            'w2_size': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w2_edge'], kernel_size['w2_edge'], kernel_size['w1_deep'], kernel_size['w2_deep']),
                mean=0, stddev=1), dtype=tf.float32),
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
        # 对所有卷积核w和b写入文件摘要
        for i in (i for i in range(6)):
            summary_visualization.variable_summaries(var=kernel_para['w%s_size' % (i + 1)], name='w%s' % (i + 1))

    # 卷积层(数据特征维度：20->5*5)
    with tf.name_scope('cnn'):
        cnn_1 = CNN(x=x, w_conv=kernel_para['w1_size'], stride_conv=1, stride_pool=2)
        # 将向量x转换为5*5方形张量
        x_reshape = CNN.reshape(f_vector=x, new_shape=(-1, 5, 5, 1))
        # 1
        layer_1 = cnn_1.convolution(input=x_reshape)
        relu1 = tf.nn.relu(layer_1)
        bn1 = cnn_1.batch_normoalization(input=relu1, is_training=is_training)
        # 2
        cnn_2 = CNN(x=bn1, w_conv=kernel_para['w2_size'], stride_conv=1, stride_pool=2)
        layer_2 = cnn_2.convolution(input=bn1)
        relu2 = tf.nn.relu(layer_2)
        bn2 = cnn_2.batch_normoalization(input=relu2, is_training=is_training)
        # 3
        cnn_3 = CNN(x=bn2, w_conv=kernel_para['w3_size'], stride_conv=1, stride_pool=2)
        layer_3 = cnn_3.convolution(input=bn2)
        relu3 = tf.nn.relu(layer_3)
        bn3 = cnn_3.batch_normoalization(input=relu3, is_training=is_training)
        # pool
        pool1 = cnn_3.pooling(pool_fun=tf.nn.max_pool, input=bn3)
        # 4
        cnn_4 = CNN(x=pool1, w_conv=kernel_para['w4_size'], stride_conv=1, stride_pool=2)
        layer_4 = cnn_4.convolution()
        relu4 = tf.nn.relu(layer_4)
        bn4 = cnn_4.batch_normoalization(input=relu4, is_training=is_training)
        # 5
        cnn_5 = CNN(x=bn4, w_conv=kernel_para['w5_size'], stride_conv=1, stride_pool=2)
        layer_5 = cnn_5.convolution()
        relu5 = tf.nn.relu(layer_5)
        bn5 = cnn_5.batch_normoalization(input=relu5, is_training=is_training)
        # pool
        pool2 = cnn_5.pooling(pool_fun=tf.nn.max_pool, input=bn5)
        # 6
        cnn_6 = CNN(x=pool2, w_conv=kernel_para['w6_size'], stride_conv=1, stride_pool=2)
        layer_6 = cnn_6.convolution()
        relu6 = tf.nn.relu(layer_6)
        bn6 = cnn_6.batch_normoalization(input=relu6, is_training=is_training)
        # 经过6层卷积运算后张量维度为：(-1, 2, 2, 96)
        # flat
        bn6_x, bn6_y, bn6_z = bn6.get_shape().as_list()[1:]
        cnn_output = CNN.reshape(f_vector=bn6, new_shape=(-1, bn6_x * bn6_y * bn6_z))

    return cnn_output

def rnn(x, summary_visualization):
    '''
    rnn网络部分
    :param x: Variable, 输入特征, 维度为一维向量
    :param summary_visualization: 摘要类型对象
    :return: rnn网络输出部分
    '''
    with tf.name_scope('rnn'):
        rnn = RNN(x=x, max_time=5, num_units=128)
        rnn_outputs, _ = rnn.dynamic_rnn(style='LSTM', output_keep_prob=1.0)
        rnn_output = rnn_outputs[:, -1, :]

    return rnn_output

def dnn(x, summary_visualization):
    '''
    dnn网络部分
    :param x: Variable, 输入特征, 维度为1维向量
    :param summary_visualization: 摘要类型对象
    :return: dnn网络输出部分
    '''
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
        # 对所有连接参数和偏置量制作摘要
        for i in (i for i in range(3)):
            w, b = h_para[i]
            summary_visualization.variable_summaries(var=w, name='w%s' % (i + 1))
            summary_visualization.variable_summaries(var=b, name='b%s' % (i + 1))

    fnn = FNN(x=x, w=h_para)
    with tf.name_scope('fc_output'):
        fc_output = fnn.fc_concat(keep_prob=0.8)

    return fc_output

def train(training_time, is_finishing):
    '''
    训练模型
    :param training_time: 标记训练次数
    :param is_finishing: 标记是否已完成训练
    :return: None
    '''
    FEATURE_DIM = 105
    LABEL_DIM = 1
    NN_graph = tf.Graph()
    with NN_graph.as_default():
        summary_visualization = Summary_Visualization()
        with tf.name_scope('placeholder'):
            x = tf.placeholder(shape=[None, FEATURE_DIM], dtype=tf.float32, name='x')
            y = tf.placeholder(shape=[None, LABEL_DIM], dtype=tf.float32, name='y')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            learning_rate_fnn = tf.placeholder(dtype=tf.float32, name='learning_rate')
        with tf.name_scope('NN'):
            cnn_op = cnn(x=x, is_training=is_training, summary_visualization=summary_visualization)
            rnn_op = rnn(x=cnn_op, summary_visualization=summary_visualization)
            fnn_op = dnn(x=rnn_op, summary_visualization=summary_visualization)
        with tf.name_scope('loss-potimize-evaluation-acc'):
            # 代价函数、优化函数、评价指标
            loss_fnn = tf.reduce_mean(tf.square(fnn_op - y))
            #添加摘要loss
            summary_visualization.scalar_summaries(arg= {'loss': loss_fnn})
            optimize_fnn = tf.train.AdamOptimizer(learning_rate=learning_rate_fnn).minimize(loss_fnn)
            evaluation_fnn = Evaluation(one_hot=False, logit=None, label=None, regression_label=y,
                                        regression_pred=fnn_op)
            acc_fnn = evaluation_fnn.acc_regression(Threshold=0.1)  # 评价指标中的阈值可以修改
            #添加摘要acc
            # summary_visualization.scalar_summaries(arg= {'acc': acc_fnn})
        with tf.name_scope('etc'):
            # 摘要汇总
            merge = summary_visualization.summary_merge()
            # 变量初始化节点,显卡占用率分配节点
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=NN_graph) as sess:
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
            savemodel = SaveImport_model(sess_ori=sess, file_suffix='\\fc_model', ops=(fnn_op, x),
                                         usefulplaceholder_count=1)
            savemodel.save_pb()


if __name__ == '__main__':
    train(training_time=0, is_finishing=False)
