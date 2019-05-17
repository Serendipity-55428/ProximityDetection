#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: testt
@time: 2019/3/14 18:44
@desc:
'''
import os.path
import pickle
import numpy as np
import xlwt
import pandas as pd
from Stacking.Routine_operation import LoadFile, SaveFile, Summary_Visualization, SaveImport_model, SaveRestore_model
from collections import Counter
import tensorflow as tf
from Stacking.AllNet import CNN
from sklearn.metrics import classification_report
from Stacking.DataGenerate import data_stepone_1, data_steptwo

def data_make():
    '''
    制作均衡分类数据
    :return: None
    '''
    rng = np.random.RandomState(0)
    # 制作fft均分类数据
    data_PNY_fft = LoadFile(p=r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_data_fft.pickle')
    PNY_fft_features = data_PNY_fft[:, :-1]
    # 归一化
    PNY_fft_features = (PNY_fft_features - np.min(PNY_fft_features, axis=0)) / \
                       (np.max(PNY_fft_features, axis=0) - np.min(PNY_fft_features, axis=0))
    # 组合归一化后的特征和标签
    PNY_fft_data = np.hstack((PNY_fft_features, data_PNY_fft[:, -1][:, np.newaxis]))
    PNY_fft_data = pd.DataFrame(PNY_fft_data, columns=[i for i in range(1, PNY_fft_data.shape[-1] + 1)])
    divided = [(0, 10), (10, 20), (20, 100), (100, 300)]
    num_per_group = 1900
    indexx = 0
    PNY_data_fft_classifier = np.zeros(shape=[1])
    for i, j in divided:
        per_group = PNY_fft_data[PNY_fft_data[PNY_fft_data.shape[-1]] > i]
        per_group = per_group[per_group[PNY_fft_data.shape[-1]] <= j]
        per_group = np.array(per_group)
        rng.shuffle(per_group)
        per_group = per_group[:num_per_group, :]
        one_hot_label = np.zeros(shape=[num_per_group, 4], dtype=np.float32)
        one_hot_label[:, indexx] = 1
        print(np.sum(one_hot_label, axis=0))
        per_group = np.hstack((per_group[:, :-1], one_hot_label))
        PNY_data_fft_classifier = np.vstack((PNY_data_fft_classifier, per_group)) if PNY_data_fft_classifier.any() else \
            per_group
        indexx += 1

    rng.shuffle(PNY_data_fft_classifier)
    SaveFile(data=PNY_data_fft_classifier, savepickle_p=r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_fft_cl.pickle')
    print(PNY_data_fft_classifier.shape)

def classific_report(y_true, y_pred):
    '''
    生成precision, recall, f1
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: DataFrame
    '''
    return classification_report(y_true=y_true, y_pred=y_pred)

def batch_normoalization(input, is_training, moving_decay= 0.9, eps= 1e-5):
    '''
    批处理层操作
    :param input: Tensor/Variable, 输入张量
    :param is_training: type= tf.placeholder, (True/False)指示当前模型是处在训练还是测试时段
    :param moving_decay: 滑动平均所需的衰减率
    :param eps: 防止bn操作时出现分母病态条件
    :return: BN层输出节点
    '''
    #获取张量维度元组
    input_shape = input.get_shape().as_list()
    #BN公式中的期望和方差学习参数
    beta = tf.Variable(tf.zeros(shape= ([input_shape[-1]])), dtype= tf.float32)
    gamma = tf.Variable(tf.ones(shape= ([input_shape[-1]])), dtype= tf.float32)
    axes = list(range(len(input_shape) - 1))
    #计算各个批次的均值和方差节点
    batch_mean, batch_var = tf.nn.moments(x= input, axes= axes)
    #滑动平均处理各个批次的均值和方差
    ema = tf.train.ExponentialMovingAverage(moving_decay)

    def mean_var_with_update():
        #设置应用滑动平均的张量节点
        ema_apply_op = ema.apply([batch_mean, batch_var])
        #明确控制依赖
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    #训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
    mean, var = tf.cond(tf.equal(is_training, True), mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # 最后执行batch normalization
    return tf.nn.batch_normalization(input, mean, var, beta, gamma, eps)

def network(x_p, x_f, is_training):
    '''
    神经网络
    :param x_p: placeholder, 输入平均密度
    :param x_f: placeholder, 输入剩余特征
    :param is_training: placeholder, 指示是否正在进行训练
    :return: 网络最终输出节点
    '''
    with tf.name_scope('cnn'):
        input_cnn = tf.reshape(tensor=x_p, shape=[-1, 10, 10, 1], name='input_cnn')
        conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       name='conv1')(input_cnn) #(None, 10, 10, :)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool1')(conv1)
        bn1 = batch_normoalization(input=pool1, is_training=is_training)

        conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       name='conv2')(bn1) #(None, 5, 5, :)
        conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       name='conv3')(conv2) #(None, 5, 5, :)
        pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool2')(conv3)
        bn2 = batch_normoalization(input=pool2, is_training=is_training)
        conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=[2, 2], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       name='conv4')(bn2) #(None, 3, 3, :)
        flat = tf.keras.layers.Flatten(name='flat')(conv4) #(None, 3*3*:)
        # print(flat)

    with tf.name_scope('rnn'):
        input_rnn = tf.reshape(tensor=flat, shape=[-1, 48, 48], name='input_rnn')
        rnn1 = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=True, name='rnn1')(input_rnn)
        rnn2 = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=False, name='rnn2')(rnn1)
        output_rnn = tf.keras.layers.Flatten(name='flat')(rnn2)
        # print(output_rnn)

    with tf.name_scope('fc'):
        input_fc = tf.concat(values=[output_rnn, x_f], axis=1)
        layer1 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='layer1')(input_fc)
        layer1 = tf.nn.dropout(x=layer1, keep_prob=0.8, name='dropout1')
        layer2 = tf.keras.layers.Dense(units=400, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='layer1')(layer1)
        layer2 = tf.nn.dropout(x=layer2, keep_prob=0.8, name='dropout2')
        output = tf.keras.layers.Dense(units=4, use_bias=True, activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='output')(layer2)
        # print(output)

    return output

def sess(training_time, is_finishing):
    '''
    训练主函数
    :return: None
    '''
    LEARNING_RATE = 1e-2
    g = tf.Graph()
    with g.as_default():
        summary_visualization = Summary_Visualization()
        with tf.name_scope('placeholder'):
            x_p = tf.placeholder(shape=[None, 100], dtype=tf.float32, name='x_p')
            x_f = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='x_f')
            y = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='y')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        output = network(x_p, x_f, is_training)
        with tf.name_scope('prediction'):
            logits = tf.nn.softmax(output, name='logits')
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
            # 添加摘要loss
            summary_visualization.scalar_summaries(arg={'loss':loss})
        with tf.name_scope('optimal'):
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)
        with tf.name_scope('acc'):
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)), tf.float32), name='acc')
            # 添加摘要loss
            summary_visualization.scalar_summaries(arg={'acc':acc})
        with tf.name_scope('etc'):
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            # 摘要汇总
            merge = summary_visualization.summary_merge()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g) as sess:
        sess.run(init)
        # 建立checkpoint节点保存对象
        saverestore_model = SaveRestore_model(sess=sess, save_file_name='fnn', max_to_keep=1)
        saver = saverestore_model.saver_build()
        if training_time != 0:
            # 导入checkpoint节点，继续训练
            saverestore_model.restore_checkpoint(saver=saver)
        # 摘要文件
        summary_writer = summary_visualization.summary_file(p='logs/', graph=sess.graph)
        #导入数据
        p=r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_fft_cl.pickle'
        # 记录折数
        fold = 0
        # 总训练轮数
        epoch_all = 1000
        # 将数据集划分为训练集和测试集
        for train, test in data_stepone_1(p_dataset_ori=p, proportion=4, is_shuffle=True):
            # print(train.shape, test.shape)
            for epoch in range(epoch_all):
                # print(epoch)
                # 设定标志在100的倍数epoch时只输出一次结果
                flag = 1
                # 以一定批次读入某一折数据进行训练
                for batch_x, batch_y in data_steptwo(train_data=train, batch_size=5*190):
                    # print(batch_x.shape, batch_y.shape)
                    # 所有训练数据每折各个批次的模型参数摘要汇总
                    summary = sess.run(merge, feed_dict={x_p: batch_x[:, :100], x_f: batch_x[:, 100:],
                                                         y: batch_y, learning_rate: LEARNING_RATE,
                                                         is_training: True})
                    _ = sess.run(opt, feed_dict={x_p: batch_x[:, :100], x_f: batch_x[:, 100:],
                                                          y: batch_y, learning_rate: LEARNING_RATE,
                                                          is_training: True})
                    summary_visualization.add_summary(summary_writer=summary_writer, summary=summary,
                                                      summary_information=epoch)
                    if (epoch % 100) == 0 and flag == 1:
                        loss_, train_acc = sess.run([loss, acc], feed_dict={x_p: batch_x[:, :100], x_f: batch_x[:, 100:],
                                                                             y: batch_y, is_training:False,
                                                                             learning_rate:LEARNING_RATE})

                        test_acc = sess.run(acc, feed_dict={x_p:test[:, :100], x_f:test[:, 100:-4], y:test[:, -4:],
                                                             is_training:False})
                        print('loss: %s, train_acc: %s, test_acc: %s' % (loss_, train_acc, test_acc))
                        # test_pred = sess.run(logits, feed_dict={x_p:test[:, :100], x_f:test[:, 100:-4], y:test[:, -4:],
                        #                                         is_training:False})
                        # y_true, y_pred = np.argmax(test[:, -4:], axis=1), np.argmax(test_pred, axis=1)
                        # pred = classific_report(y_true=y_true, y_pred=y_pred)
                        # print(pred)
                        flag = 0
                    # 保存checkpoint节点
                    saverestore_model.save_checkpoint(saver=saver, epoch=epoch, is_recording_max_acc=False)

            fold += 1
        summary_visualization.summary_close(summary_writer=summary_writer)
        # if is_finishing:
        #     # 将最终训练好的模型保存为pb文件
        #     savemodel = SaveImport_model(sess_ori=sess, file_suffix='\\fft_cl_model', ops=(output, x_p, x_f),
        #                                  usefulplaceholder_count=2)
        #     savemodel.save_pb()

if __name__ == '__main__':
    # data_make()
    sess(training_time=0, is_finishing=False)