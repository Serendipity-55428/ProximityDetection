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
from tensorflow.python.framework import graph_util

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
        :return: 摘要汇总节点
        '''
        merge = tf.summary.merge_all()
        return merge

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

    def scalar_summaries(self, arg):
        '''
        生成节点摘要
        :param arg: 生成节点名和节点变量名键值对的字典
        :return: None
        '''
        for key, value in arg.items():
            tf.summary.scalar(key, value)

class SaveImport_model:
    '''
    将模型写入序列化pb文件
    '''
    def __init__(self, sess_ori, file_suffix, ops):
        '''
        构造函数
        :param sess_ori: 原始会话实例对象(sess)
        :param file_suffix: type= str, 存储模型的文件名后缀
        :param ops: iterable, 节点序列（含初始输入节点x）
        '''
        self.__sess_ori = sess_ori
        self.__pb_file_path = os.getcwd() #获取pb文件保存路径前缀
        self.__file_suffix = file_suffix
        self.__ops = ops

    def save_pb(self):
        '''
        保存计算图至指定文件夹目录下
        :return: None
        '''
        # 存储计算图为pb格式
        #设置output_node_names列表(含初始输入x节点)
        output_node_names = ['{op_name}'.format(op_name = per_op.op.name) for per_op in self.__ops]
        # Replaces all the variables in a graph with constants of the same values
        constant_graph = graph_util.convert_variables_to_constants(self.__sess_ori,
                                                                   self.__sess_ori.graph_def,
                                                                   output_node_names= output_node_names)
        # 写入序列化的pb文件
        with tf.gfile.FastGFile(self.__pb_file_path + '\\' + 'model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        # Builds the SavedModel protocol buffer and saves variables and assets
        # 在和project相同层级目录下产生带有savemodel名称的文件夹
        builder = tf.saved_model.builder.SavedModelBuilder(self.__pb_file_path + self.__file_suffix)
        # Adds the current meta graph to the SavedModel and saves variables
        # 第二个参数为字符列表形式的tags – The set of tags with which to save the meta graph
        builder.add_meta_graph_and_variables(self.__sess_ori, ['cpu_server_1'])
        # Writes a SavedModel protocol buffer to disk
        # 此处p值为生成的文件夹路径
        p = builder.save()
        print('计算图保存路径为: ', p)
        for i in output_node_names:
            print('节点名称为:' + i)

    def use_pb(self, sess_new, ops_name):
        '''
        将计算图从指定文件夹导入至工程
        :param sess_new: 待导入节点的新会话对象
        :param ops_name: iterable, 待导入节点名
        :return: 模型节点序列
        '''
        # Loads the model from a SavedModel as specified by tags
        tf.saved_model.loader.load(sess_new, ['cpu_server_1'], self.__pb_file_path + self.__file_suffix)

        # Returns the Tensor with the given name
        # 名称都为'{output_name}: output_index'格式
        ops = [sess_new.graph.get_tensor_by_name('{op_name}:0'.format(op_name = per_op))
               for per_op in ops_name] #ops序列第一个值为初始化节点x
        return ops



