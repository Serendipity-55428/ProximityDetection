#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: Tenfold
@time: 2018/11/23 20:38
@desc:
'''

import tensorflow as tf
import numpy as np
from cnn_rnn.Fmake2read import FileoOperation, Excel2Numpy

# 只用于print对象输出非科学计数法数值
np.set_printoptions(suppress=True)

# 类中需要输入参数p_in, filename, read_in_fun, num_shards, instance_per_shard, ftype, ttype, fshape, tshape,
# batch_size, capacity, batch_fun, batch_step, min_after_dequeue(choice)

p_in = r'C:\Users\xiaosong\Desktop\TeamProject\all.xls'
filename = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\output.tfrecords-%.5d-of-%.5d'
num_shards = 5
instance_per_shard = 80
read_in_fun = Excel2Numpy
ftype, ttype = tf.float64, tf.float64
fshape, tshape = [4], [1]
batch_size = 40
capacity = 400 + 40 * batch_size
batch_fun = tf.train.batch
batch_step = 10

fileop = FileoOperation(p_in, filename, read_in_fun, num_shards, instance_per_shard, ftype, ttype, fshape, tshape,
                        batch_size, capacity, batch_fun, batch_step)
feature_batch, target_batch = fileop.ParseDequeue(r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\output.tfrecords-*')

######################################
#结合TFRecord文件解析编组的十折交叉验证
######################################
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    # sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 十折交叉验证进行10轮, test_batch为测试集
    threshold = 100
    train_steps = 0
    i = 0
    try:
        while not coord.should_stop():  # 如果线程应该停止则返回True
            cur_feature_batch, cur_target_batch = sess.run([feature_batch, target_batch])

            if train_steps // 11 != i:
                print(cur_feature_batch, cur_target_batch)
                # 此处可以添加训练数据操作ndarray类型节点sess.run([], feed_dict= {})
            else:
                test_batch = (cur_feature_batch, cur_target_batch)
                print('test_set NO.%s:' % i)
                print(test_batch)
                i += 1
                # 此处可以添加测试数据操作ndarray类型节点ress.run([], feed_dict= {})

            train_steps += 1
            if train_steps >= threshold:
                coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        # When done, ask the threads to stop. 请求该线程停止
        coord.request_stop()
        # And wait for them to actually do it. 等待被指定的线程终止
        coord.join(threads)



