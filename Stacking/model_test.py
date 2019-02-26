#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: model_test
@time: 2019/2/26 22:54
@desc:
'''
import tensorflow as tf
import numpy as np
import os
from Routine_operation import SaveImport_model

cnn_graph = tf.Graph()
with cnn_graph.as_default():
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config= tf.ConfigProto(gpu_options= gpu_options), graph= cnn_graph) as sess:
    sess.run(init)
    savemodel = SaveImport_model(sess_ori= None, file_suffix= '\\cnn_model', ops= )