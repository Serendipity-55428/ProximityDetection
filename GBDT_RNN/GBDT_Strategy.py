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
            'gamma': 0.1, #指定节点分裂所需的最小损失函数下降值
            'max_depth': 5, #书的最大深度
            'lambda': 3, #正则化项参数
            'subsample': 0.7, #控制对于每一棵树随机采样的比例
            'colsample_bytree': 0.7, #和subsample作用相同
            'min_child_weight': 3., #决定最小叶子节点样本权重和
            'silent': 0, #默认为0，当参数值为1时，静默模式开启，不会输出任何信息
            'eta': 0.1, #控制每一步权重减小的参数0.01-0.2
            'seed': 1000, #随机数种子
            'nthread': 4 #进行多线程控制
        }

        #将数据特征和标签存为DMatrix类型数据
        dtrain = xgb.DMatrix(train_feature, train_target)
        #设置集成弱学习器数量(需根据自行设定误差限设定)
        num_boost_round = 100
        #构建模型并训练
        model = xgb.train(params= GBDT_params, dtrain= dtrain,
                          num_boost_round= num_boost_round)

        #保存模型
        model.save_model('GBDT.model')
        #导出模型和特征映射到txt文件
        model.dump_model('dump.raw.txt', 'featmap.txt')

        #加载模型
        #init model
        model = xgb.Booster({'nthread': 4})
        #load data
        model.load_model('model.bin')

        #对测试集预测
        test_feature, test_target = data_generator()
        predict_test = model.predict(test_feature)

        #显示重要特征
        plot_importance(model)
        plt.show()










