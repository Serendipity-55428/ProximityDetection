#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: PyCharm
@file: main.py
@time: 2018/11/10 10:20
@desc:
'''
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from dataoperation import FileArrangement, LoadFile, SaveFile, CostError
import time

def variable_summaries(var, name):
    '''监控指标可视化函数'''
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

def DataNorm(x):
    '''数据归一化normalized_value = (real _ value - min{values}) /(max{values} - min{values})'''
    x = (x - x.min(axis= 0)) / (x.max(axis= 0) - x.min(axis= 0))
    return x

def DataCreat(x, y, test_size): #test_size 用于设定训练检测比例
    '''用于将数据集按比例划分成训练集合验证集'''
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size= test_size, shuffle= True)
    return x_train, y_train, x_test, y_test

def fclayer(weights, x):
    '''构建包含四个隐藏层的全连接网络, 并用relu函数做神经元激活'''

    #三层全连接 net:激活前, a:relu激活后 输入整个x_train组成行特征有效矩阵
    with tf.name_scope('net'):
        net_1 = tf.matmul(x, weights['w_1']) + weights['b_1']
        a_1 = tf.nn.relu(net_1, name= 'a_1')
        net_2 = tf.matmul(a_1, weights['w_2']) + weights['b_2']
        a_2 = tf.nn.relu(net_2, name= 'a_2')
        net_3 = tf.matmul(a_2, weights['w_3']) + weights['b_3']
        a_3 = tf.nn.relu(net_3, name= 'a_3')
        net_4 = tf.matmul(a_3, weights['w_4']) + weights['b_4']
        a_4 = tf.nn.relu(net_4, name= 'a_4')
    return a_4

def main():
    #样例数据集,判断模型是否收敛
    # rng = np.random.RandomState(0)
    # x = np.linspace(0, 199, 600, dtype= np.float32).reshape(200, 3)
    # y = x[:, 0] + rng.normal(0, 0.05, 200)
    # y = y[:, np.newaxis]
    p = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\COST_PNY.pickle' #PNY和OLDBURG
    # p = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\COST_OLDBURG.pickle'
    data = LoadFile(p) #shape: [200, 4] [objects, friends, Threshold, radius]
    x, y = data[:, :3], data[:, 3]
    y = y[:, np.newaxis]
    # 设定全连接权重矩阵尺寸(三个隐藏层)
    #PNY隐藏层
    weight = {
        'w_1_insize': 3,
        'w_1_outsize': 100,
        'w_2_insize': 100,
        'w_2_outsize': 200,
        'w_3_insize': 200,
        'w_3_outsize': 300,
        'w_4_insize': 300,
        'w_4_outsize': 1
    }
    #OLDBURG隐藏层
    # weight = {
    #     'w_1_insize': 3,
    #     'w_1_outsize': 200,
    #     'w_2_insize': 200,
    #     'w_2_outsize': 300,
    #     'w_3_insize': 300,
    #     'w_3_outsize': 400,
    #     'w_4_insize': 400,
    #     'w_4_outsize': 1
    # }
    with tf.name_scope('data_s'):
        x_s = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        y_s = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

    # 权重矩阵
    with tf.name_scope('wb'):
        weights = {
            'w_1': tf.Variable(tf.truncated_normal([weight['w_1_insize'], weight['w_1_outsize']], stddev= 0.1), \
                               dtype= tf.float32, name= 'w_1'),
            'w_2': tf.Variable(tf.truncated_normal([weight['w_2_insize'], weight['w_2_outsize']], stddev= 0.1), \
                               dtype= tf.float32, name= 'w_2'),
            'w_3': tf.Variable(tf.truncated_normal([weight['w_3_insize'], weight['w_3_outsize']], stddev= 0.1), \
                               dtype=tf.float32, name= 'w_3'),
            'w_4': tf.Variable(tf.truncated_normal([weight['w_4_insize'], weight['w_4_outsize']], stddev= 0.1), \
                               dtype= tf.float32, name= 'w_4'),
            #设为常量
            # 'b_1': tf.constant(1.0, shape=[weight['w_1_outsize']], name='b_1'),
            # 'b_2': tf.constant(1.0, shape=[weight['w_2_outsize']], name='b_2'),
            # 'b_3': tf.constant(1.0, shape=[weight['w_3_outsize']], name='b_3')
            #设为变量
            'b_1': tf.Variable(tf.truncated_normal([weight['w_1_outsize']], dtype= tf.float32, stddev= 0.1), name= 'b_1'),
            'b_2': tf.Variable(tf.truncated_normal([weight['w_2_outsize']], dtype= tf.float32, stddev= 0.1), name= 'b_2'),
            'b_3': tf.Variable(tf.truncated_normal([weight['w_3_outsize']], dtype= tf.float32, stddev= 0.1), name= 'b_3'),
            'b_4': tf.Variable(tf.truncated_normal([weight['w_4_outsize']], dtype= tf.float32, stddev= 0.1), name= 'b_4'),
        }
        # variable_summaries(weights['w_1'], 'w_1')
        # variable_summaries(weights['w_2'], 'w_2')
        # variable_summaries(weights['w_3'], 'w_3')
        # variable_summaries(weights['w_4'], 'w_4')
        # variable_summaries(weights['b_1'], 'b_1')
        # variable_summaries(weights['b_2'], 'b_2')
        # variable_summaries(weights['b_3'], 'b_3')
        # variable_summaries(weights['b_4'], 'b_4')

    #归一化后数据
    x = DataNorm(x)
    # print(x)
    # 代价函数及梯度下降函数
    with tf.name_scope('train_error'):
        # 代价函数及梯度下降函数
        a_3 = fclayer(weights, x_s)
        cost = tf.reduce_mean(tf.square(a_3 - y_s))
        #cost摘要
        # tf.summary.scalar('cost', cost)
        # optimize = tf.train.GradientDescentOptimizer(8e-4).minimize(cost) #PNY学习率为8e-4 OLDBURG学习率8e-4
        optimize = tf.train.AdamOptimizer(1e-3).minimize(cost) #PNY学习率为1e-3 OLDBURG学习率1e-3
    with tf.name_scope('test_error'):
        cost_t = tf.reduce_mean(tf.square(a_3 - y_s))
        predict = a_3

    init = tf.global_variables_initializer()
    #摘要汇总
    # merged = tf.summary.merge_all()

    #检验模型是否收敛
    # x_train, y_train = x, y

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options= gpu_options)) as sess:
        sess.run(init)
        #摘要文件
        # summary_writer = tf.summary.FileWriter('logs/', sess.graph)
        for i in range(300000): #PNY2000000 #OLDBURG3000000
            #自助法划分train和test集合每个数据未被划分入test的概率是1/e
            x_train, x_test, y_train, y_test = DataCreat(x, y, test_size= 0.37)
            #每个epoch的摘要记录
            # summary = sess.run(merged, feed_dict={x_s: x_train, y_s: y_train})
            #记录optimize起始时间start,只记录一次
            if i == 0:
                start = time.time()
            _, cost_fc = sess.run([optimize, cost], feed_dict= {x_s: x_train, y_s: y_train})
            #记录每轮优化后的时间stop
            stop = time.time()
            time_statistic = str(stop - start)
            if i % 1000 == 0:
                print('第%s轮优化结束训练误差是: %s, 时间是: %s秒\n' % (i, cost_fc, time_statistic))
                cost_test, predict_test = sess.run([cost_t, predict], feed_dict={x_s: x_test, y_s: y_test})
                print('测试误差为: %s \n' % cost_test)
                #统计epoch、epoch_time、train_cost、test_cost
                statistic = np.array([i, time_statistic, cost_fc, cost_test]) if i == 0 else \
                    np.vstack((statistic, np.array([i, time_statistic, cost_fc, cost_test])))
        CostError(statistic)

            #添加摘要
    #         summary_writer.add_summary(summary, i)
    # summary_writer.close()
if __name__ == '__main__':
    main()





