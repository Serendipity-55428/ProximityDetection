#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: Routine_operation
@time: 2019/2/18 13:08
@desc:
'''
import tensorflow as tf
import numpy as np
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt

def LoadFile(p):
    '''
    读取文件
    :param p: 数据集绝对路径
    :return: 数据集
    '''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data

def SaveFile(data, savepickle_p):
        '''
        存储整理好的数据
        :param data: 待存储数据
        :param savepickle_p: pickle后缀文件存储绝对路径
        :return: None
        '''
        if not os.path.exists(savepickle_p):
            with open(savepickle_p, 'wb') as file:
                pickle.dump(data, file)

class Summary_Visualization:
    '''
    生成摘要文本，并将摘要信息写入摘要文本中
    '''
    def variable_summaries(self, var, name):
        '''监控指标可视化函数'''
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)

    def summary_merge(self):
        '''
        摘要汇总
        :return: 汇总所有摘要的函数
        '''
        return tf.summary.merge_all()

    def summary_file(self, p, graph):
        '''
        生成摘要文件对象summary_writer
        :param p: 摘要文件保存路径
        :param graph: 写入文件中的计算图
        :return: 文件对象
        '''
        return tf.summary.FileWriter(p, graph)

    def add_summary(self, summary_writer, summary, summary_information):
        '''
        在摘要文件中添加摘要
        :param summary_writer: 摘要文件对象
        :param summary: 摘要汇总变量
        :param summary_information: 摘要信息（至少含有经过merge后那些节点摘要信息）
        :return: None
        '''
        summary_writer.add_summary(summary, summary_information)

    def summary_close(self, summary_writer):
        '''
        关闭摘要文件对象
        :param summary_writer: 摘要文件对象
        :return: None
        '''
        summary_writer.close()

    def scalar_summaries(self, **arg):
        '''
        生成节点摘要
        :param arg: 生成节点名和节点变量名键值对的关键字参数
        :return: None
        '''
        for key, value in arg.items():
            tf.summary.scalar(key, value)

class SaveImport_model:
    '''
    将模型保存为节点和pb格式
    '''


