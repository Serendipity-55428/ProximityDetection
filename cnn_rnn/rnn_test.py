#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: rnn_test
@time: 2018/11/27 21:43
@desc:
'''
#
import tensorflow as tf
import numpy as np

x = tf.placeholder(dtype= tf.float32, shape= [4, 20])
y = tf.placeholder(dtype= tf.float32, shape= [4])
#定义LSTMCell实例对象
# cell = tf.nn.rnn_cell.LSTMCell(num_units= 128) #设置隐藏层h大小
# #print <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x00000150CB5DAF60>
#
# #因LSTM网络需要前一时刻的h向量，故需要定义h0，第一个参数为batch_size
# h0 = cell.zero_state(4, tf.float32)
#
# #新版本tensorflow中LSTMCell实例对象中的__call__函数改为和普通类中的
# # __call__魔法方法一致，使得类对象可被作为工厂函数直接调用
# #直接调用只能前进一步，后面的tf.nn.dynamic_rnn可以设置一次前进的步数
#
# output, h1 = cell(x, h0)
#
# #输出隐层shape和长记忆单元shape
# # print(h1.h.shape) # (4, 128)
# # print(h1.c.shape) # (4, 128)
#
inputs = tf.reshape(x, [-1, 5, 4])
#因为tf.nn.dynamic_rnn函数中的inputs参数接收的数据维度格式为: [batch_size, max_time, depth],
#所以需要在中间加入max_time维度，该维度意思为循环次数。
# initial_state = cell.zero_state(4, tf.float32)
# outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state= initial_state)

#多层LSTM神经网络
def get_a_cell():
    return tf.nn.rnn_cell.GRUCell(num_units= 128)
    # return tf.nn.rnn_cell.LSTMCell(num_units=128)

cell = get_a_cell()
cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(get_a_cell(), input_keep_prob= 1.0, output_keep_prob= 0.8) for _ in range(3)])
output_m, state_m = tf.nn.dynamic_rnn(cells, inputs, initial_state= cells.zero_state(4, tf.float32))
# output, state = tf.nn.dynamic_rnn(cell, inputs, initial_state= cell.zero_state(4, tf.float32))
# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
#state_m = (h, c) h is a tensor of shape [batch_size, cell_state_size], c is a tensor of shape h

#双向神经网络
cell_fw = get_a_cell()
cell_bw = get_a_cell()
outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, \
                                                         inputs, initial_state_fw= cell_fw.zero_state(4, tf.float32),\
                                                         initial_state_bw= cell_bw.zero_state(4, tf.float32))
# output_1 = tf.concat([outputs[0], outputs[-1]], -1)
# output_1 = tf.reshape(tf.concat(output_1, 2), [4, -1]) #4*50
# output_2 = tf.matmul(state_m, tf.Variable(tf.truncated_normal([128, 1], dtype= tf.float32))) + tf.Variable([1], dtype= tf.float32)
# loss = tf.reduce_mean(tf.square(output_2 - y))
# opt = tf.train.AdamOptimizer(learning_rate= 1e-4).minimize(loss)
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out, state_h = sess.run([output_m, state_m], feed_dict= {x: np.arange(80, dtype= np.float32).reshape(4, 20)})
        print(out.shape, state_h[-1].shape)
        # outputs_put, outstate_put = sess.run([output, state], feed_dict= {x: np.arange(80, dtype= np.float32).reshape(4, 20)})
        # print(outputs_put.shape, outstate_put.shape)
        # print(outstate_put[0].h.shape, outstate_put[0].h.shape)
        # for _ in range(10000):
        #     _, loss_s = sess.run([opt, loss], feed_dict= {x: np.arange(80, dtype= np.float32).reshape(4, 20),
        #                                                   y: np.arange(4, dtype= np.float32)})
        #     print(loss_s)



