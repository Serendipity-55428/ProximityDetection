#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: GBDT_test
@time: 2018/12/24 19:01
@desc:
'''

from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn import model_selection
import xgboost as xgb
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

#数据集分为：训练集：测试集= 5:2，并且训练集分为5份做交叉验证
#数据从文件中读取
# train_data = []
# test_data = []
#
# def data_generator():
#     '''
#     数据生成器函数，自动将训练集中四份划分为训练集将剩下一份划分为验证集
#     :param :
#     :return: 划分好的数据-标签组合
#     '''
#
# model = XGBRegressor(
#     max_depth= 7, #树的最大深度
#     learning_rate= 0.1, #学习率
#     n_estimators= 90, #树的个数
#     objective= 'reg:linear', #损失函数类型
#     nthread= 4, #线程数
#     gamma= 0.1, #节点分裂时损失函数所需最小下降值
#     min_child_weight= 1, #叶子结点最小权重
#     subsample= 1., #随机选择样本比例建立决策树
#     colsample_bytree= 1., #随机选择样本比例建立决策树
#     reg_lambda= 2, #二阶范数正则化项权衡值
#     scale_pos_weight= 1., #解决样本个数不平衡问题
#     random_state= 1000, #随机种子设定值
# )
#
# # train_feature, train_target = data_generator()
# # test_feature, test_target = data_generator()
#
# #载入波士顿房价预测数据集做测试
# boston = datasets.load_boston()
# train_feature, test_feature, train_target, test_target = train_test_split(boston.data, boston.target, test_size= 0.25,
#                                                                           random_state= 42)
#
# #训练数据
# model.fit(X= train_feature,
#           y= train_target,
#           # eval_set= [(test_feature, test_target)], #交叉验证集特征和标签
#           # early_stopping_rounds= 10, #模型多少次得分基本不变就停止训练
#           # verbose= True, #是否使用验证集
#           )
#
# #模型预测计算精确度
# pred_target = model.predict(test_feature)
# # print(pred_target)
# ex_var = explained_variance_score(test_target, pred_target) #解释方差：1-(var(y_pred-y)/var(y_pred))
# print('解释方差为: %s' % ex_var)
# me_ab = mean_absolute_error(test_target, pred_target) #街区距离
# print('街区距离为: %s' % me_ab)
# me_sq = mean_squared_error(test_target, pred_target)
# print('均方差为: %s' % me_sq)
#
#
# #保存模型
# model.get_booster().save_model('GBDT.model')

#显示重要参数
# plot_importance(model)
# plt.show()

#加载模型
# model = xgb.Booster(model_file= 'GBDT.model')
#
#载入波士顿房价预测数据集做测试
boston = datasets.load_boston()
print(boston.data.shape, boston.target.shape)
# train_feature, test_feature, train_target, test_target = train_test_split(boston.data, boston.target, test_size= 0.333,
#                                                                           random_state= 42)
#
# #训练数据
# model.fit(X= train_feature,
#           y= train_target,
#
#           # eval_set= [(test_feature, test_target)], #交叉验证集特征和标签
#           # early_stopping_rounds= 10, #模型多少次得分基本不变就停止训练
#           # verbose= True, #是否使用验证集
#           )
# #模型预测计算精确度
# pred_target = model.predict(xgb.DMatrix(test_feature))
# print(pred_target)
# ex_var = explained_variance_score(test_target, pred_target) #解释方差：1-(var(y_pred-y)/var(y_pred))
# print('解释方差为: %s' % ex_var)
# me_ab = mean_absolute_error(test_target, pred_target) #街区距离
# print('街区距离为: %s' % me_ab)
# me_sq = mean_squared_error(test_target, pred_target)
# print('均方差为: %s' % me_sq)
#
# #模型可视化
# digraph = xgb.to_graphviz(model, num_trees= 4)
# digraph.format = 'png'
# digraph.view('./boston_xgb')

#k-fold
kf = model_selection.KFold(n_splits= 5, shuffle= False, random_state= 32)
boston_1 = np.hstack((boston.data, boston.target[:, np.newaxis]))
np.random.shuffle(boston_1)
for train, test in kf.split(boston_1):
    # train_data, test_data = boston_1[train], boston_1[test]
    print(test)

# for train, test in kf.split(boston_1):
#     train_data, test_data = boston_1[train], boston_1[test]
#     print(test)

# print(boston.target.shape)
# x = np.hstack((boston.data, boston.target[:, np.newaxis]))
# print(x.shape)
# print(type(kf.split(boston_1)))
