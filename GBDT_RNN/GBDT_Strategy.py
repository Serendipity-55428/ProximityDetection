#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: GBDT_Strategy
@time: 2018/12/21 23:26
@desc:
'''
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot as plt

#数据集分为：训练集：测试集= 5:2，并且训练集分为5份做交叉验证
#数据从文件中读取
train_data = []
test_data = []

def data_generator():
    '''
    数据生成器函数，自动将训练集中四份划分为训练集将剩下一份划分为验证集
    :param :
    :return: 划分好的数据-标签组合
    '''

def xgboost_main():
    '''GBDT主函数'''

    #训练集均分为5组做5折交叉验证
    for fold in range(5):
        train_feature, train_target = data_generator()

        #设置GBDT所需参数散列
        GBDT_params = {
            'booster': 'gbtree', #所选择的若学习器算法框架
            'objective': 'reg:linear', #所选择的回归损失函数
            'gamma': 0.1, #
            'max_depth': 5,
            'lambda': 3,
            'subsample': 0.7,
            'colsample_bytree': 0.7, #
            'min_child_weight': 3., #决定最小叶子节点样本权重和
            'silent': 0, #默认为0，当参数值为1时，静默模式开启，不会输出任何信息
            'eta': 0.1, #控制每一步权重减小的参数0.01-0.2
            'seed': 1000, #随机数种子
            'nthread': 4 #进行多线程控制
        }





