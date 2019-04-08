#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: DataGenerate
@time: 2019/2/20 21:51
@desc:
'''
from Stacking.Routine_operation import SaveFile, LoadFile
from sklearn import model_selection
import numpy as np
import os
import pickle

#数据生成器制作
def data_stepone(p_dataset_ori, proportion):
    '''
    数据集生成步骤1:划分训练/测试集数据
    :param p_dataset_ori: string, 原始数据提取绝对路径
    :param proportion: int, 选择10/5折交叉验证
    :return: 训练集， 测试集 ，shape=((-1, 25/20+1), (-1, 25/20+1))
    '''
    dataset_ori = LoadFile(p= p_dataset_ori)
    batch_size = dataset_ori.shape[0] // proportion
    for i in range(0, dataset_ori.shape[0], batch_size): #取一折为测试集，剩下组合为训练集
        train = np.vstack((dataset_ori[:i, :], dataset_ori[i+batch_size:, :])) #只用后20个密度特征
        test = dataset_ori[i:i+batch_size, :]
        yield train, test

def data_stepone_1(p_dataset_ori, proportion, is_shuffle):
    '''
    交叉验证,按比例划分训练/测试集
    :param p_dataset_ori: string, 原始数据提取绝对路径
    :param proportion: int, 选择10/5折交叉验证
    :param is_shuffle: Ture/False, 选择是否随机划分
    :return: 划分后的训练集和测试集
    '''
    dataset_ori = LoadFile(p=p_dataset_ori)
    # k-fold对象,用于生成训练集和交叉验证集数据
    kf = model_selection.KFold(n_splits=proportion, shuffle=is_shuffle, random_state=32)
    for train_data_index, cv_data_index in kf.split(dataset_ori):
        # 找到对应索引数据
        train_data, cv_data = dataset_ori[train_data_index], dataset_ori[cv_data_index]
        # print(np.isnan(train_data).any(), np.isnan(cv_data).any())
        yield train_data, cv_data


def data_steptwo(train_data, batch_size):
    '''
    对训练数据按照给定批次大小输出
    :param train_data: 训练数据
    :param batch_size: 输出批次大小
    :return: 训练数据批次特征、标签
    '''
    for i in range(0, train_data.shape[0], batch_size):
        feature = train_data[i:i+batch_size, :-1]
        # feature = (feature - np.min(feature, axis=0)) / (np.max(feature, axis=0) - np.min(feature, axis=0))
        # print(feature_norm.max())
        yield feature, train_data[i:i+batch_size, -1]

def second_dataset(dataset_ori, pred):
    '''
    生成次级学习器数据集
    :param dataset_ori: (-1, 24+1)原始数据集(feature, label)
    :param pred: shape= (-1, 2)初级学习器预测后的结果
    :return: 次级学习器数据集
    '''
    sub_dataset = np.hstack((dataset_ori[:, :4], pred, dataset_ori[:, -1][:, np.newaxis]))
    return sub_dataset

# 类别划分通用函数
#     def transform(label):
#         '''
#         将回归标签转换为类别标签
#         :param label: 待转换回归标签
#         :return: 类别标签
#         '''
#         def divide(x):
#             divide_point = [1, 10, 30, 100, 200]
#             divide_label = [i for i in range(6)]
#             position = bisect(a=divide_point, x=x)
#             return divide_label[position]
#         divide_ufunc = np.frompyfunc(divide, 1, 1)
#         return divide_ufunc(label)
#
#     #提取特征归一化后的fft数据（带标签）
#     PNY_fft_norm = LoadFile(p=r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_fft_norm.pickle')
#     #提取标签
#     fft_label = PNY_fft_norm[:, -1]
#     #生成类别标签
#     fft_class = transform(fft_label)
#     #合成新数据
#     PNY_fft_norm_c = np.hstack((PNY_fft_norm[:, :-1], fft_class[:, np.newaxis]))
#     # SaveFile(data=PNY_fft_norm_c, savepickle_p=r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_fft_norm_c.pickle')
#     print(Counter(fft_class))

if __name__ == '__main__':
    #制作数据总数10000,5折
    rng = np.random.RandomState(0)
    # dataset = rng.randint(0, 10, size= (10000, 21))
    # SaveFile(data= dataset, savepickle_p= r'F:\ProximityDetection\Stacking\test_data.pickle')
    dataset_2 = rng.randint(0, 10, size= (10000, 7))
    SaveFile(data= dataset_2, savepickle_p= r'F:\ProximityDetection\Stacking\test_data_2.pickle')
    # 导入数据
    # p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data.pickle'
    p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data_2.pickle'
    dataset_ori = LoadFile(p=p_dataset_ori)
    #step1
    for train, test in data_stepone(p_dataset_ori= p_dataset_ori, proportion= 5):
        print(train.shape)
        print(test.shape)
        # for feature, label in data_steptwo(train_data= train, batch_size= 500):
        #     print(feature, label)
        # break


