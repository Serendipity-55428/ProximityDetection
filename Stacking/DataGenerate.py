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
from Routine_operation import SaveFile, LoadFile
import numpy as np
import os
import pickle

#数据生成器制作
def data_stepone(p_dataset_ori, padding, proportion):
    '''
    数据集生成步骤1:划分训练/测试集数据
    :param p_dataset_ori: string, 原始数据提取绝对路径
    :param padding: bool, 是否需要补一行0(cnn需要，rnn不需要)
    :param proportion: int, 选择10/5折交叉验证
    :return: 训练集， 测试集 ，shape=((-1, 25/20+1), (-1, 25/20+1))
    '''
    dataset_ori = LoadFile(p= p_dataset_ori)
    if padding:
        zeros = np.zeros(dtype= np.float32, shape= ([dataset_ori.shape[0], 5]))
        dataset_ori = np.hstack((dataset_ori[:, :-1], zeros, dataset_ori[:, -1][:, np.newaxis]))
    batch_size = dataset_ori.shape[0] // proportion
    for i in range(0, dataset_ori.shape[0]-batch_size+1, batch_size):
        train = np.vstack((dataset_ori[:i, :], dataset_ori[i+batch_size:, :]))
        test = dataset_ori[i: i+batch_size]
        yield train, test

def data_steptwo(train_data, batch_size):
    '''
    对训练数据按照给定批次大小输出
    :param train_data: 训练数据
    :param batch_size: 输出批次大小
    :return: 训练数据批次特征、标签
    '''
    for i in range(0, train_data.shape[0]-batch_size+1, batch_size):
        yield train_data[i:i+batch_size, :-1], train_data[i:i+batch_size, -1]

def second_dataset(dataset_ori, pred):
    '''
    生成次级学习器数据集
    :param dataset_ori: (-1, 24+1)原始数据集(feature, label)
    :param pred: shape= (-1, 2)初级学习器预测后的结果
    :return: 次级学习器数据集
    '''
    sub_dataset = np.hstack((dataset_ori[:, :4], pred, dataset_ori[:, -1][:, np.newaxis]))
    return sub_dataset

if __name__ == '__main__':
    #制作数据总数100,5折
    dataset = np.arange(25*5).reshape(25, 5)
    SaveFile(data= dataset, savepickle_p= r'F:\ProximityDetection\Stacking\test_data.pickle')
    # 导入数据
    p_dataset_ori = r'F:\ProximityDetection\Stacking\test_data.pickle'
    dataset_ori = LoadFile(p=p_dataset_ori)
    #step1
    for train, test in data_stepone(p_dataset_ori= p_dataset_ori, padding= True, proportion= 5):
        print(train)
        print(test)
        for feature, label in data_steptwo(train_data= train, batch_size= 5):
            print(feature, label)
        break


