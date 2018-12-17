#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: HyMultiNN
@time: 2018/12/4 20:05
@desc:
'''
import tensorflow as tf
import numpy as np

#RNN以及所有RNN变体模型类
class RecurrentNeuralNetwork:
    '''
    所有循环神经网络以及多种组合
    '''

    __slots__ = ('__x', '__keep_prob')

    @classmethod
    def get_a_cell(cls, net_name, num_units):
        '''
        生成一个LSTM节点
        :param net_name: 选择‘LSTM’/‘GRU’
        :param num_units: 隐藏层向量维度
        :return: 一个LSTM的cell
        '''
        return tf.nn.rnn_cell.LSTMCell(num_units= num_units) if net_name == 'LSTM' else tf.nn.rnn_cell.GRUCell(num_units= num_units)

    @classmethod
    def dynamic_rnn(cls, cells, inputs, max_time):
        '''
        :param cells: 读入结点
        :param inputs: 读入'ndarray' / 'Tensor', shape= [batch_size, max_time, depth]
        :param max_time: 循环网络中循环次数
        :return: max_time个隐藏层h,c,最终的输出h,c
        '''
        inputs_Tensor, inputs_shape = RecurrentNeuralNetwork.reshapex(x= inputs, max_time= max_time)
        outputs, final_state_h = tf.nn.dynamic_rnn(cells, inputs_Tensor, initial_state= cells.zero_state(inputs_shape[0], tf.float32))
        return outputs, final_state_h

    @classmethod
    def bidirectional_dynamic_rnn(cls, cell_fw, cell_bw, inputs, batch_size):
        '''
        :param cell_fw: 读入正向循环LSTM网络
        :param cell_bw：读入反向循环LSTM网络
        :param inputs: 读入Tensor, shape= [batch_size, max_time, depth]
        :param batch_size: Tensor中包含的样本数
        :return: max_time * 2个隐藏层的h,c, 最终的输出h,c
        '''
        outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw= cell_fw, cell_bw= cell_bw,
                                                         inputs= inputs, initial_state_fw= cell_fw.zero_state(batch_size, tf.float32),
                                                         initial_state_bw= cell_bw.zero_state(batch_size, tf.float32))
        return outputs, final_state

    @staticmethod
    def reshapex(x, max_time):
        '''
        :param x: Variable
        :param max_time: 最大循环次数
        :return: Variable, x.shape= [x.get_shape().as_list()[0], max_time, x.get_shape().as_list()[-1]/max_time]
        '''
        shape = x.get_shape().as_list()
        batch_size = shape[0]
        depth = int(shape[-1] / max_time)
        x_Tensor = tf.reshape(x, shape= [batch_size, max_time, depth])

        return x_Tensor, shape

    def __init__(self, x, keep_prob):
        '''
        构造函数
        :param x: 一个批次样本特征向量集，shape= [batch_size, depth]，type= 'ndarray' / 'Tensor'
        :param keep_prob: dropout技术中所需参数
        '''
        self.__x = x
        self.__keep_prob = keep_prob

    def output_x(self, max_time):
        '''
        :param max_time: 最大循环次数
        :return: x_Tensor, y_Tensor, type= Variable
        '''
        x_Tensor, x_shape = RecurrentNeuralNetwork.reshapex(x= self.__x, max_time= max_time)
        return x_Tensor, x_shape

    def dropout_rnn(self, cell):
        '''
        对循环神经网络采用dropout技术正则化
        :param cell: 循环神经网络结点
        :return: 经过dropout后的结点cell
        '''
        cell = tf.nn.rnn_cell.DropoutWrapper(cell= cell, input_keep_prob= 1.0, output_keep_prob= self.__keep_prob)
        return cell

    def h_combination(self, style, outputs, final_state):
        '''
        对循环神经网络各时刻输出进行组合，该函数不包括情况是：outputs的type= tuple时且输出为最后一层的最后一时刻的h_state情况（多层单向LSTM）
        :param style: 'per' / 'final'
        :param outputs: LSTM网络输出的outputs
        :param final_state: LSTM网络输出的final_state_h
        :return: 组合后的矩阵，type= Variable
        '''
        if isinstance(outputs, tuple): #双向LSTM情况下outputs中第一个是fw_cell的ouput，第二个是bw_cell的output
            outputs = tf.concat([outputs[0], outputs[-1]], -1) #outputs的第二个维度为max_time
            result = tf.reshape(tf.concat(outputs, 2), shape= [self.__x.shape[0], -1]) if style == 'per' else \
                tf.concat([final_state[0].h, final_state[-1].h], -1) #双向LSTM情况下final_state_h也是两个（h，c）
        else:
            result = tf.reshape(tf.concat(outputs, 2), shape= [self.__x.shape[0], -1]) if style == 'per' else \
                final_state.h
        return result

    def multiLSTM(self, net_name, num_unit, layer_num):
        '''
        :param net_name: 所使用的网络名字 'LSTM'/ 'GRU'
        :param num_unit: LSTM节点的单元数量(所有结点的单元数量设置相同)
        :param layer_num:  LSTM结点层数
        :return: 按线性图方式连接的multiLSTM网络
        '''
        # cell = RecurrentNeuralNetwork.get_a_cell(num_units= num_unit)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell= cell, input_keep_prob= 1.0, output_keep_prob= self.__keep_prob)
        cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
            cell= RecurrentNeuralNetwork.get_a_cell(net_name= net_name, num_units= num_unit), input_keep_prob= 1.0,
            output_keep_prob= self.__keep_prob) for _ in range(layer_num)])
        return cells

class FCNN:
    '''
    全连接神经网络及其所需参数
    '''
    __slots__ = ('__x', '__y', '__keep_prob')

    def __init__(self, x, keep_prob= 1.0):
        '''
        :param x: Tensor, 批次数据特征
        :param keep_prob: dropout屏蔽阻断神经元概率
        '''
        self.__x = x
        self.__keep_prob = keep_prob

    def per_layer(self, w, b, param= 'None', name= 'fc_ops'):
        '''
        构建单层全连接神经网络，并加入dropout技术减轻过拟合
        :param w: 权重矩阵， type= 'ndarray'
        :param b: 激活值， type= 'ndarray'
        :param param: 每次per_layer函数返回的函数值作为per_layer函数的参数循环调用
        :param name: 模型名称， 默认为‘fc_ops’
        :return: 单层神经元输出
        '''
        layer = tf.matmul(self.__x, w) + b if param == 'None' else tf.matmul(param, w) + b
        layer = tf.nn.relu(layer)
        return tf.nn.dropout(layer, keep_prob= self.__keep_prob, name= name)

class CNN:
    '''
    卷积网络及其所需参数和操作
    '''
    @staticmethod
    def conv2d(x, w, b, strides= 1, use_bn= 'no'):
        '''
        二维卷积运算后经过relu激活
        :param x: Tensor shape= [batch_size, h_x, v_x, d_x]
        :param w: Variable shape= [h_w, v_w, d_w, d_new_w]
        :param b: Variable shape= [d_new_w]
        :param strides: 卷积核移动步伐
        :param use_bn: 'yes' / 'no' 控制是否使用batch_normalization进行标准化
        :return: 卷积、激活后所得结果
        '''
        #定义在使用bn技术时生成training结点来区分正在训练还是在测试中使用bn
        training = None
        #如果使用bn技术时需要将更新bn技术中参数gama和bata的操作加入到tf.GraphKeys.UPDATE_OPS中
        extra_update_ops = None

        conv = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        if use_bn == 'no':
            conv = tf.nn.bias_add(conv, b)
        else:
            training = tf.placeholder(dtype= tf.bool)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            conv = CNN.batch_normalization(conv, training= training)

        conv = tf.nn.relu(conv)
        return conv, training, extra_update_ops

    @staticmethod
    def pooling(style, x, k= 2):
        '''
        池化操作
        :param style: type= 'function', 'tf.nn.max_pool' / 'tf.nn.avg_pool'
        :param x: Variable shape= [batch_size, h_x, v_x, d_x]
        :param k: 池化核大小，池化核移动步伐默认和k值一致
        :return: Variable, 池化后结果
        '''
        pooling = style(x, ksize= [1, k, k, 1], strides= [1, k, k, 1], padding= 'SAME')
        return pooling

    @staticmethod
    def batch_normalization(x, training):
        '''
        batch_normalization技术用于使各批次数据以及训练集和测试集数据满足相同分布，这个分布先经过归一化然后反归一化
        :param x: Variable, 在进行relu之前的x
        :param training: tf.placeholder, 在sess.run()函数中需要feed进'True'：在训练 / 'False'：在测试
        :return: bn标准化之后的数据
        '''
        return tf.layers.batch_normalization(x, training)

    @staticmethod
    def d_one2d_two(x, den_2, den_3):
        '''
        如果数据非二维或高维，则需要将数据按照一定规则转化形状为至少二维
        :param x: Tensor , shape= (batch_size, depth)
        :param den_2: 需要将depth转化为 den_2 * den_3, 其中的den_2
        :param den_3: 需要将depth转化为den_2 * den_3, 其中的den_3
        :return: 转换后的数据， shape= (batch_size, den_2, den_3)
        '''
        #Tensor无法被做任何改变，需要先转换为Variable类型
        # x = tf.Variable(x)
        x = tf.reshape(x, [x.get_shape().as_list()[0], den_2, den_3, 1])
        return x






























