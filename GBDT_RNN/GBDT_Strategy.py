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
from sklearn import model_selection
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

def GBDT_main():
    '''GBDT主函数'''

    #初始化MSE和交叉验证折数fold
    MSE, fold = 0, 1
    #生成训练集和交叉验证集
    # 载入波士顿房价预测数据集做测试
    boston = datasets.load_boston()
    # k-fold对象
    kf = model_selection.KFold(n_splits=5, shuffle=False, random_state=32)

    boston_1 = np.hstack((boston.data, boston.target[:, np.newaxis]))
    np.random.shuffle(boston_1)

    #子学习器个数
    n_estimators = 2
    #设置误差阈值：三个误差评估设置#
    Threshold = 8

    #生成GBDT模型
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

    while (MSE > Threshold) or (fold == 1):
        #如果验证集MSE值大于阈值则将GBDT中弱学习器数量自增1
        model.n_estimators += 1
        for train_data_index, cv_data_index in kf.split(boston_1):
            #找到对应索引数据
            train_data, cv_data = boston_1[train_data_index], boston_1[cv_data_index]
            # 训练数据
            model.fit(X=train_data[:, :-1], y=train_data[:, -1])

            # 对测试集进行MSE计算
            pred_cv = model.predict(cv_data[:, :-1])
            MSE = ((fold - 1) * MSE + mean_squared_error(cv_data[:, -1], pred_cv)) / fold
            fold += 1

        print('CART树个数: %s, 验证集MSE: %s' % (model.n_estimators, MSE))

    #保存模型
    # model.get_booster().save_model('GBDT.model')

    # 显示重要参数
    plot_importance(model)
    plt.show()

    # 模型可视化
    digraph = xgb.to_graphviz(model, num_trees=4)
    digraph.format = 'png'
    digraph.view('./boston_xgb')

if __name__ == '__main__':
    GBDT_main()





























