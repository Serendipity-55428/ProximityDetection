#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: subtest
@time: 2018/12/13 15:48
@desc:
'''

import tensorflow as tf
import numpy as np

x = tf.Variable(np.arange(20).reshape(4, 5), dtype= tf.float32)
y = tf.Variable(np.arange(5)[:, np.newaxis], dtype= tf.float32)
z = tf.matmul(x, y)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    z_1 = sess.run(z)
    z_2 = np.hstack((z_1, z_1))
    print(z_2)
