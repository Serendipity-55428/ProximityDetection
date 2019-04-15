#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: classifier2
@time: 2019/4/11 16:26
@desc:
'''
from bisect import bisect
from Stacking.AllNet import CNN, RNN, FNN
from Stacking.TestEvaluation import Evaluation
from Stacking.DataGenerate import data_stepone, data_stepone_1, data_steptwo, second_dataset
from Stacking.Routine_operation import SaveFile, LoadFile, Summary_Visualization, SaveImport_model, SaveRestore_model
import tensorflow as tf
import numpy as np
from collections import Counter

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
        'w1_edge': 5,
        'w1_deep': 96,
        'w2_edge': 3,
        'w2_deep': 256,
        'w3_edge': 2,
        'w3_deep': 384,
    }
    with tf.name_scope('w'):
        # 核张量
        kernel_para = {
            'w1_size': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w1_edge'], kernel_size['w1_edge'], 1, kernel_size['w1_deep']),
                mean=0, stddev=0.1), dtype=tf.float32),
            'w2_size': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w2_edge'], kernel_size['w2_edge'], kernel_size['w1_deep'], kernel_size['w2_deep']),
                mean=0, stddev=0.1), dtype=tf.float32),
            'w3_size': tf.Variable(initial_value=tf.truncated_normal(
                shape=(kernel_size['w3_edge'], kernel_size['w3_edge'], kernel_size['w2_deep'], kernel_size['w3_deep']),
                mean=0, stddev=0.1), dtype=tf.float32),
        }
        # 对所有卷积核w和b写入文件摘要
        for i in (i for i in range(3)):
            summary_visualization.variable_summaries(var=kernel_para['w%s_size' % (i + 1)], name='w%s' % (i + 1))

    # 卷积层(数据特征维度：20->5*5)
    with tf.name_scope('cnn'):
        cnn_1 = CNN(x=x, w_conv=kernel_para['w1_size'], stride_conv=1, stride_pool=2)
        # 将向量x转换为5*5方形张量
        x_reshape = CNN.reshape(f_vector=x, new_shape=(-1, 10, 10, 1))
        # 1
        layer_1 = cnn_1.convolution(input=x_reshape)
        relu1 = tf.nn.relu(layer_1)
        bn1 = cnn_1.batch_normoalization(input=relu1, is_training=is_training)
        pool1 = cnn_1.pooling(pool_fun=tf.nn.max_pool, input=bn1) #(-1, 5, 5, 96)
        # 2
        cnn_2 = CNN(x=pool1, w_conv=kernel_para['w2_size'], stride_conv=1, stride_pool=2)
        layer_2 = cnn_2.convolution(input=bn1)
        relu2 = tf.nn.relu(layer_2)
        bn2 = cnn_2.batch_normoalization(input=relu2, is_training=is_training)
        pool2 = cnn_2.pooling(pool_fun=tf.nn.max_pool, input=bn2) #(-1, 3, 3, 256)
        # 3
        cnn_3 = CNN(x=bn2, w_conv=kernel_para['w3_size'], stride_conv=1, stride_pool=2)
        layer_3 = cnn_3.convolution(input=bn2)
        relu3 = tf.nn.relu(layer_3)
        bn3 = cnn_3.batch_normoalization(input=relu3, is_training=is_training)
        pool3 = cnn_3.pooling(pool_fun=tf.nn.max_pool, input=bn3) #(-1, 2, 2, 384)
        # flat
        pool3_x, pool3_y, pool3_z = pool3.get_shape().as_list()[1:]
        cnn_output = CNN.reshape(f_vector=pool3, new_shape=(-1, pool3_x * pool3_y * pool3_z))
        print(cnn_output.shape)
    return cnn_output

def rnn(x, summary_visualization):
    '''
    rnn网络部分
    :param x: Variable, 输入特征, 维度为一维向量
    :param summary_visualization: 摘要类型对象
    :return: rnn网络输出部分
    '''
    with tf.name_scope('rnn'):
        rnn = RNN(x=x, max_time=16, num_units=192)
        rnn_outputs, _ = rnn.dynamic_multirnn(style='LSTM', layers_num=2, output_keep_prob=0.8, is_reshape='yes')
        rnn_output = rnn_outputs[:, -1, :]

    return rnn_output #(-1, 192)

def fnn(x, summary_visualization):
    '''
    dnn网络部分
    :param x: Variable, 输入特征, 维度为1维向量
    :param summary_visualization: 摘要类型对象
    :return: dnn网络输出部分
    '''
    h_size = {
        'w1_insize': 192+4,
        'w1_outsize': 200,
        'w2_insize': 200,
        'w2_outsize': 100,
        'w3_insize': 100,
        'w3_outsize': 4
    }

    with tf.name_scope('w-b'):
        h_para = (
            (tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w1_insize'], h_size['w1_outsize']]),
                                                           mean=0, stddev=0.1), dtype=tf.float32, name='w1'),
             tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w1_outsize']]),
                                                           mean=0, stddev=0.1), dtype=tf.float32, name='b1')),
            (tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w2_insize'], h_size['w2_outsize']]),
                                                           mean=0, stddev=0.1), dtype=tf.float32, name='w2'),
             tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w2_outsize']]),
                                                           mean=0, stddev=0.1), dtype=tf.float32, name='b2')),
            (tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w3_insize'], h_size['w3_outsize']]),
                                                           mean=0, stddev=0.1), dtype=tf.float32, name='w3'),
             tf.Variable(initial_value=tf.truncated_normal(shape=([h_size['w3_outsize']]),
                                                           mean=0, stddev=0.1), dtype=tf.float32, name='b3')),
        )
        # 对所有连接参数和偏置量制作摘要
        for i in (i for i in range(3)):
            w, b = h_para[i]
            summary_visualization.variable_summaries(var=w, name='w%s' % (i + 1))
            summary_visualization.variable_summaries(var=b, name='b%s' % (i + 1))
    fnn = FNN(x=x, w=h_para)
    with tf.name_scope('fc_output'):
        fc_output = fnn.fc_concat(keep_prob=0.8)

    return fc_output #(-1, 1)

def train(training_time, is_finishing):
    '''
    训练模型
    :param training_time: 标记训练次数
    :param is_finishing: 标记是否已完成训练
    :return: None
    '''
    FEATURE_DIM = 104
    LABEL_DIM = 4
    LEARNING_RATE = 1e-3
    NN_graph = tf.Graph()
    with NN_graph.as_default():
        summary_visualization = Summary_Visualization()
        with tf.name_scope('placeholder'):
            x = tf.placeholder(shape=[None, FEATURE_DIM-4], dtype=tf.float32, name='x') #fft变换取模后结果
            x_add = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='x_add') #剩余特征
            y = tf.placeholder(shape=[None, LABEL_DIM], dtype=tf.float32, name='y')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            learning_rate_fnn = tf.placeholder(dtype=tf.float32, name='learning_rate')
        with tf.name_scope('NN'):
            cnn_op = cnn(x=x, is_training=is_training, summary_visualization=summary_visualization)
            rnn_op = rnn(x=cnn_op, summary_visualization=summary_visualization)
            x_in = tf.concat(values=[rnn_op, x_add], axis=1)
            print(x_in.shape)
            fnn_op = fnn(x=x_in, summary_visualization=summary_visualization)
        with tf.name_scope('loss-potimize-evaluation-acc'):
            # 定义softmax交叉熵和损失函数以及精确度函数
            loss_fnn = -tf.reduce_mean(y * tf.log(fnn_op), name='loss')
            #添加摘要loss
            summary_visualization.scalar_summaries(arg= {'loss': loss_fnn})
            optimize_fnn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_fnn).minimize(loss_fnn)
            evaluation_fnn = Evaluation(one_hot= True, logit= tf.nn.softmax(fnn_op), label= y,
                                        regression_pred= None, regression_label= None)
            acc_fnn = evaluation_fnn.acc_classification()
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
        p_dataset_ori = r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_fft_cl.pickle'
        # 记录折数
        fold = 0
        # 总训练轮数
        epoch_all = 10000
        # 将数据集划分为训练集和测试集
        for train, test in data_stepone_1(p_dataset_ori=p_dataset_ori, proportion=4, is_shuffle=True):
            # print(train.shape, test.shape)
            for epoch in range(epoch_all):
                # 设定标志在100的倍数epoch时只输出一次结果
                flag = 1
                # 以一定批次读入某一折数据进行训练
                for batch_x, batch_y in data_steptwo(train_data=train, batch_size=5*190):
                    # print(batch_x.shape, batch_y.shape)
                    #所有训练数据每折各个批次的模型参数摘要汇总
                    summary = sess.run(merge, feed_dict= {x: batch_x[:, :100], x_add: batch_x[:, 100:],
                                                          y: batch_y, learning_rate_fnn: LEARNING_RATE,
                                                          is_training: True})
                    _ = sess.run(optimize_fnn, feed_dict={x: batch_x[:, :100], x_add: batch_x[:, 100:],
                                                          y: batch_y, learning_rate_fnn: LEARNING_RATE,
                                                          is_training: True})
                    summary_visualization.add_summary(summary_writer= summary_writer, summary= summary,
                                                      summary_information= epoch)
                    if (epoch % 100) == 0 and flag == 1:
                        loss_fnn_ = sess.run(loss_fnn, feed_dict={x: batch_x[:, :100], x_add: batch_x[:, 100:],
                                                                  y: batch_y,
                                                                  is_training: False})
                        acc_fnn_ = sess.run(acc_fnn, feed_dict={x: test[:, :100], x_add: test[:, 100:-4],
                                                                y: test[:, -4:],
                                                                is_training: False})
                        # acc_fnn_train = sess.run(acc_fnn, feed_dict={x:train[:, :100], x_add:train[:, 100:-1],
                        #                                              y:train[:, -1][:, np.newaxis]})
                        # print('训练集精度为: %s' % acc_fnn_train)
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

        # 定义Evaluation类对象
        # evalulation_2 = Evaluation(one_hot=True, logit=op_logit, label=y, regression_pred=None, regression_label=None)
        # PRF_dict_ = evalulation_2.PRF_tables(mode_num=6)
        # # _, PRF_dict_ = evalulation_2.session_PRF(acc= None, prf_dict= PRF_dict)
        # print(PRF_dict_)


if __name__ == '__main__':
    train(training_time=1, is_finishing=False)