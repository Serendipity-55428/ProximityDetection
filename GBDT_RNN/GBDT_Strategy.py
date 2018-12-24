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

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from xgboost import plot_importance
import xgboost as xgb
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

class DataGenerator:
    pass

test_data = []

def GBDT_main():
    '''GBDT主函数'''

    #初始化MSE
    MSE = 0
    #交叉验证折数值
    fold = 5
    #创建数据生成器类对象
    datagenerator = DataGenerator()
    #子学习器个数
    n_estimators = 10
    #设置误差阈值：三个误差评估设置#
    Threshold = 8
    for i in range(fold):
        #从训练集中生成子训练集和验证集
        train_dataset, validation_dataset = datagenerator.function # 加

        #如果不是第一次交叉验证则需要导入前一次交叉检验得到的模型
        if not i:

        else:
            # 加载模型
            model = xgb.Booster(model_file='GBDT.model')

        while MSE > 10:
            # 每次验证都重新生成GBDT模型
            model = XGBRegressor(
                max_depth=7,  # 树的最大深度(可调)
                learning_rate=0.1,  # 学习率(可调)
                n_estimators=n_estimators,  # 树的个数
                objective='reg:linear',  # 损失函数类型
                nthread=4,  # 线程数
                gamma=0.1,  # 节点分裂时损失函数所需最小下降值（可调）
                min_child_weight=1,  # 叶子结点最小权重
                subsample=1.,  # 随机选择样本比例建立决策树
                colsample_bytree=1.,  # 随机选择样本比例建立决策树
                reg_lambda=2,  # 二阶范数正则化项权衡值（可调）
                scale_pos_weight=1.,  # 解决样本个数不平衡问题
                random_state=1000,  # 随机种子设定值
            )

            # 训练数据
            model.fit(X=train_dataset[:, :-1],
                      y=train_dataset[:, -1],
                      eval_set= [(validation_dataset[:, :-1], validation_dataset[:, -1])], #交叉验证集特征和标签
                      early_stopping_rounds= 10, #模型多少次得分基本不变就停止训练
                      verbose= True, #是否使用验证集
                      )

            #对测试集进行MSE计算
            pred_test = model.predict(test_data[:, :-1])
            MSE = mean_squared_error(test_data[:, -1], pred_test)
            print('CART树个数: %s, 测试集MSE: %s' % (n_estimators, MSE))

            n_estimators += 1

        #保存模型
        model.get_booster().save_model('GBDT.model')





























