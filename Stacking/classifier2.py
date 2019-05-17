#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: classifier2
@time: 2019/4/11 16:26
@desc:
'''
import xgboost as xgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Stacking.Routine_operation import LoadFile, SaveFile
from collections import Counter
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer
from xgboost import plot_importance
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

def classific_report(y_true, y_pred):
    '''
    生成precision, recall, f1
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: DataFrame
    '''
    return classification_report(y_true=y_true, y_pred=y_pred)

def data_operation(p):
    '''
    数据制作
    :param p: 导入数据路径
    :return: None
    '''
    data_fft = LoadFile(p)
    #将特征从整体数据中分离出来并做归一化后和标签进行组合
    label = data_fft[:, -4:]
    label_one = np.argmax(label, axis=1)
    # print(Counter(label_one))
    data_fft = np.hstack((data_fft[:, :-4], label_one[:, np.newaxis]))
    print(data_fft.shape)
    SaveFile(data=data_fft, savepickle_p=r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_fft_cl_1.pickle')


def multi_XGBoost(max_depth, learning_rate, n_estimators, objective, nthread, gamma, min_child_weight,
                  subsample, reg_lambda, scale_pos_weight):
    '''
    XGBoost对象
    :param max_depth: 树的最大深度
    :param learning_rate: 学习率
    :param n_estimators: 树的个数
    :param objective: 损失函数类型
   'reg:logistic' –逻辑回归。
   'binary:logistic' –二分类的逻辑回归问题，输出为概率。
   'binary:logitraw' –二分类的逻辑回归问题，输出的结果为wTx。
   'count:poisson' –计数问题的poisson回归，输出结果为poisson分布。在poisson回归中，max_delta_step的缺省值为0.7。(used to safeguard optimization)
   'multi:softmax' –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
   'multi:softprob' –和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量reshape成ndata行nclass列的矩阵。没行数据表示样本所属于每个类别的概率。
   'rank:pairwise' –set XGBoost to do ranking task by minimizing the pairwise loss
    :param nthread: 线程数
    :param gamma: 节点分裂时损失函数所需最小下降值
    :param min_child_weight: 叶子结点最小权重
    :param subsample: 随机选择样本比例建立决策树
    :param reg_lambda: 二阶范数正则化项权衡值
    :param scale_pos_weight: 解决样本个数不平衡问题
    :return: XGBoost对象
    '''
    xgbc = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        objective=objective,
        nthread=nthread,
        gamma=gamma,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=subsample,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=32,
    )

    return xgbc


def training_main(model, dataset_sim):
    '''
    针对多个模型进行训练操作
    :param model: 需要训练的模型
    :param training_data: 需要载入的数据集
    :param data_simset: 待输入数据
    :return: None
    '''
    # k-fold对象,用于生成训练集和交叉验证集数据
    kf = model_selection.KFold(n_splits=4, shuffle=True, random_state=32)
    # 交叉验证次数序号
    fold = 1

    for train_data_index, cv_data_index in kf.split(dataset_sim):
        # 找到对应索引数据
        train_data, cv_data = dataset_sim[train_data_index], dataset_sim[cv_data_index]
        # 训练数据
        model.fit(X=train_data[:, :-1], y=train_data[:, -1])

        print('第%s折模型训练集精度为: %s' % (fold, model.score(train_data[:, :-1], train_data[:, -1])))

        # 对验证集进行预测
        print(cv_data[:, :-1].shape)
        pred_cv = model.predict(cv_data[:, :-1])
        # print(pred_cv)
        # 对验证数据进行指标评估
        eva = classific_report(y_true=cv_data[:, -1], y_pred=pred_cv)
        print(eva)
        fold += 1

def main():
    data = LoadFile(p=r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_fft_cl_1.pickle')
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
    dataset_sim = imp.fit_transform(data)
    XGBoost = multi_XGBoost(max_depth=2, learning_rate=1e-2, n_estimators=300,
                            objective='binary:logistic', nthread=4, gamma=0.1,
                            min_child_weight=1, subsample=1, reg_lambda=2, scale_pos_weight=1.)
    training_main(model=XGBoost, dataset_sim=dataset_sim)
    digraph = xgb.to_graphviz(XGBoost, num_trees=2)
    digraph.format = 'png'
    digraph.view('./ProximityDetection_xgb')
    xgb.plot_importance(XGBoost)
    plt.show()

if __name__ == '__main__':
    # p = r'F:\ProximityDetection\Stacking\dataset_PNY\PNY_fft_cl.pickle'
    # data_operation(p)
    main()



