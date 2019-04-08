#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: TestEvaluation
@time: 2019/2/16 10:04
@desc:
'''
import tensorflow as tf
import numpy as np
from collections import Counter

class Evaluation:

    def __init__(self, one_hot, logit, label, regression_pred, regression_label):
        '''
        评估类构造函数(输入参数均为节点op，如果未使用tensorflow框架，需要先把数据转换为张量节点后方可使用)
        :param logit: dtype= tf.Tensor, shape= (test_label_batch, classifical_num(one_hot)) 预测结果(softmax或sigmoid后的结果)
        :param label: dtype= tf.Tensor, shape= (test_label_batch, classifical_num(one_hot)) 实际标签
        :param regression_pred: dtype= tf.Tensor, shape= (test_label_batch, regression) 预测结果
        :param regression_label: dtype= tf.Tensor, shape= (test_label_batch, regression) 实际标签值
        :param one_hot: dtype= np.bool, 是否使用one-hot编码
        '''
        self.__one_hot = one_hot
        self.__logit = logit
        self.__label = label
        self.__regression_pred = regression_pred
        self.__regression_label = regression_label

    @staticmethod
    def acc_regression_divided(Threshold, divided_point, ndarray_label, ndarray_logits):
        '''
        分段回归精确度
        :param Threshold: 预测值与实际值之间的绝对值之差阈值
        :param divided_point: 划分标签数据的分段点
        :param ndarray_label: ndarray型标签向量
        :param ndarray_logits: ndarray型预测向量
        :return: 分段精确度散列表
        '''
        #预测准确值向量
        bool_ = np.abs(ndarray_label-ndarray_logits) <= Threshold
        is_true = np.where(bool_, 1, 0)
        # print(np.abs(ndarray_label-ndarray_logits))
        # print(is_true.shape)
        #分段准确率散列初始化
        acc_divided = {}
        divided = ['<1', '1~10', '10~100', '>100']
        for l in range(len(divided_point)-1):
            per_group = np.where((ndarray_label>=divided_point[l]) & (ndarray_label<divided_point[l+1]), is_true, 0)
            # print(per_group.shape)
            total = np.where((ndarray_label>=divided_point[l]) & (ndarray_label<divided_point[l+1]), 1, 0)
            acc_per = np.sum(per_group) / np.sum(total)
            # print(np.sum(per_group), np.sum(total))
            acc_divided[divided.pop(0)] = acc_per

        return acc_divided


    def acc_classification(self):
        '''
        模式识别精确度（预测正确数/样本总数）
        :return: 精确率, 返回计算图节点op，结果需要放在计算图中运行转为ndarray
        '''
        #输入非one-hot编码形式
        if not self.__one_hot:
            is_equal = tf.equal(self.__logit, self.__label)
        #输入one-hot编码形式
        else:
            is_equal = tf.equal(tf.argmax(self.__logit, axis= 1), tf.argmax(self.__label, axis= 1))
        is_equal_cast = tf.cast(is_equal, tf.float32)
        acc_rate = tf.reduce_mean(is_equal_cast)
        return acc_rate

    def acc_regression(self, Threshold):
        '''
        回归精确度（预测值与实际值残差在阈值范围内的数量/样本总数）
        :param Threshold: 预测值与实际值之间的绝对值之差阈值
        :return: 精确率，返回计算图节点op，结果需要放在计算图中运行转为ndarray
        '''
        #残差布尔向量
        is_true = tf.abs(self.__regression_pred - self.__regression_label) <= Threshold
        is_true_cast = tf.cast(is_true, tf.float32)
        acc_rate_regression = tf.reduce_mean(is_true_cast)
        return acc_rate_regression

    def PRF_tables(self, mode_num):
        '''
        构造多（二）精确率、召回率、F1参数表格
        :param mode_num: 模式类别数
        :return: PRF表格字典, 字典值为计算图节点元组, shape= (pre_op, recall_op, F1_op)
        '''

        # 无论之前是否已经建立过计算图，此时都要建立一个专门用于计算性能指标的计算图
        g_eval = tf.Graph()
        with g_eval.as_default():
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            # 初始化PRF字典
            PRF_dict = {}
            # 非one-hot编码类型
            if not self.__one_hot:
                logit_convert, label_convert = self.__logit, self.__label
            else:
                logit_convert = tf.argmax(self.__logit, 1)
                label_convert = tf.argmax(self.__label, 1)
            for i in range(mode_num):
                # 模式i的预测向量bool，预测为模式i及为正例，未预测为模式i及为反例
                each_mode_logit = tf.equal(logit_convert, i)
                # 模式i的标签向量bool，实际是模式i及为正例，实际不是模式i及为反例
                each_mode_label = tf.equal(label_convert, i)
                # 二阶混淆矩阵中的真正例数
                TP = tf.reduce_sum(tf.cast(each_mode_label & each_mode_logit, tf.float32))
                #bool转化为float32
                each_mode_logit = tf.cast(each_mode_logit, tf.float32)
                each_mode_label = tf.cast(each_mode_label, tf.float32)
                # 预测为正例数（模式i）
                P = tf.reduce_sum(each_mode_logit)
                # 实际为正例数（模式i）
                T = tf.reduce_sum(each_mode_label)

                # 精确率、召回率、F1参数元组
                pre, recall = TP / P, TP / T
                f1 = (2 * pre * recall) / (pre + recall)
                # 评价指标元组shape = (pre, recall, F1_score)
                eval_tuple = pre, recall, f1
                #调试代码
                # eval_tuple = TP, P, T
                PRF_dict[str(i)] = eval_tuple

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g_eval) as sess:
            sess.run(init)

            # 如果是分类问题则需要把PRF字典中所有op都转换成具体值
            for i in range(len(PRF_dict.keys())):
                pre, recall, f1 = sess.run(list(PRF_dict[str(i)]))
                PRF_dict[str(i)] = (pre, recall, f1)

        return PRF_dict

    def session_acc(self, acc):
        '''
        精确率在计算图中执行相关计算
        :param acc: 分类回回归问题的精确率计算节点
        :return: ndarray型数组
        '''
        #创建专门计算精确率的计算图
        g_acc = tf.Graph()
        with g_acc.as_default():
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            acc = acc
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_acc) as sess:
            sess.run(init)
            acc_ = sess.run(acc)

        return acc_


if __name__ == '__main__':
    pass


