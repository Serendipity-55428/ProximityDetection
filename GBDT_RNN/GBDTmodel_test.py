#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: GBDTmodel_test
@time: 2018/12/26 22:26
@desc:
'''
import numpy as np
import tensorflow as tf
import os
import xgboost as xgb

#首先加载训练好的GBDT结构
model = xgb.Booster(model_file= 'GBDT.model')
#加载数据
input = np.arange(720*24, dtype= np.float32).reshape(720, 24)
predict_GBDT = model.predict(xgb.DMatrix(input[:, :4]))


#pb文件路径（在当前目录下生成）
pb_file_path = os.getcwd()

#建立计算图，用于载入最后一级学习器的所有计算节点
g1 = tf.Graph()
with g1.as_default():
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config= tf.ConfigProto(gpu_options= gpu_options), graph= g1) as sess:
    sess.run(init)
    # Loads the model from a SavedModel as specified by tags
    tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path+'savemodel')

    # Returns the Tensor with the given name
    # 名称都为'{output_name}: output_index'格式
    x = sess.graph.get_tensor_by_name('x-y/x:0')
    # y = sess.graph.get_tensor_by_name('y:0')
    gru_op = sess.graph.get_tensor_by_name('gru_ops/fc/Relu_2:0')

    #读入的数据需要用户自行输入(单独测试gru数据)
    # input = np.arange(720*20, dtype= np.float32).reshape(720, 20)
    # predict_gru = sess.run(gru_op, feed_dict={x: input[:, :]})

    #同时测试GBDT+GRU数据
    predict_gru = sess.run(gru_op, feed_dict={x: input[:, 4:]})
    #初级学习器输出特征
    # print('初级学习器预测特征向量为:', predict_gru)

    #输出最终预测
    fin_predict = predict_GBDT + predict_gru
    print(fin_predict)