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
import numpy as np
import os
import pickle

'''
数据集格式：分为train和test,按照9:1或4:1比例分，样本特征和标签同时被一个生成器生成，样本特征总数为24，分为20和4单独存放,20的数据长度变为25（补0）
'''