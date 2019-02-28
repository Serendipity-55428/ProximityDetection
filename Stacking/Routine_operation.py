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
    def __init__(self, sess_ori, file_suffix, ops, usefulplaceholder_count):
        '''
        构造函数
        :param sess_ori: 原始会话实例对象(sess)
        :param file_suffix: type= str, 存储模型的文件名后缀
        :param ops: iterable, 节点序列（含初始输入节点x）
        :param usefulplaceholder_count: int, 待输入有用节点(placeholder)数量
        '''
        self.__sess_ori = sess_ori
        self.__pb_file_path = os.getcwd() #获取pb文件保存路径前缀
        self.__file_suffix = file_suffix
        self.__ops = ops
        self.__usefulplaceholder_count = usefulplaceholder_count

    def save_pb(self):
        '''
        保存计算图至指定文件夹目录下
        :return: None
        '''
        # 存储计算图为pb格式,将所有保存后的结点名打印供导入模型使用
        #设置output_node_names列表(含初始输入x节点)
        output_node_names = ['{op_name}'.format(op_name = per_op.op.name) for per_op in self.__ops]
        # Replaces all the variables in a graph with constants of the same values
        constant_graph = graph_util.convert_variables_to_constants(self.__sess_ori,
                                                                   self.__sess_ori.graph_def,
                                                                   output_node_names= output_node_names[:-self.__usefulplaceholder_count])
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

    def use_pb(self, sess_new):
        '''
        将计算图从指定文件夹导入至工程
        :param sess_new: 待导入节点的新会话对象
        :return: None
        '''
        # Loads the model from a SavedModel as specified by tags
        tf.saved_model.loader.load(sess_new, ['cpu_server_1'], self.__pb_file_path + self.__file_suffix)

    def import_ops(self, sess_new, op_name):
        '''
        获取图中的某一个计算节点
        :param sess_new: 待带入节点的新会话对象
        :param op_name: 计算节点名
        :return: 计算节点
        '''
        op = sess_new.graph.get_tensor_by_name('%s:0' % op_name)
        return op

class SaveRestore_model:
    '''
    在训练过程中及时保存中间训练结果
    '''
    def __init__(self, sess, save_file_name, max_to_keep):
        '''
        checkpoint保存类构造函数
        :param sess: 当前会话
        :param save_file_name: 模型临时保存路径
        :param max_to_keep: 保存模型的个数（可自动根据max_to_keep的数值保存模型）
        '''
        self.__sess = sess
        self.__save_file_name = save_file_name
        self.__max_to_keep = max_to_keep

    def saver_build(self):
        '''
        创建Saver对象
        :return: Saver对象
        '''
        saver = tf.train.Saver(max_to_keep= self.__max_to_keep)
        return saver

    def save_checkpoint(self, saver, epoch, is_recording_max_acc, max_acc= 0, curr_acc= 0):
        '''
        生成checkpoint节点
        :param saver: Saver实例对象
        :param epoch: epoch值
        :param is_recording_max_acc: bool, 选择是否需要在出现最大精确率时候记录max_to_keep次精确度的值和epoch值
        :param max_acc: 最大精确度,默认为0（只有在is_recording_max_acc为True时才指定）
        :param curr_acc: 当前精确度,默认为0（只有在is_recording_max_acc为True时才指定）
        :return: None
        '''
        def save_ckpt(sess, saver, ckpt_name, global_step):
            '''
            保存ckpt节点
            :param sess: 会话实例对象
            :param saver: Saver实例对象
            :param ckpt_name: ckpt节点名
            :param global_step: epoch值
            :return: None
            '''
            ckpt_path = r'ckpt/%s.ckpt' % ckpt_name
            saver.save(sess, ckpt_path, global_step= global_step)

        if is_recording_max_acc:
            if curr_acc > max_acc:
                save_ckpt(self.__sess, saver, self.__save_file_name, global_step= epoch)
        else:
            save_ckpt(self.__sess, saver, self.__save_file_name, global_step= epoch)

    def restore_checkpoint(self, saver, checkpoint_file= 'ckpt/'):
        '''
        将checkpoint文件导入当前图中
        :param saver: Saver实例对象
        :param checkpoint_file: checkpoint文件路径, 默认为'ckpt/'
        :return: None
        '''
        model_file = tf.train.latest_checkpoint('ckpt/')
        saver.restore(self.__sess, model_file)





