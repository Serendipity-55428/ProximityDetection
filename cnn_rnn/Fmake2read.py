#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: Fmake2read
@time: 2018/11/22 21:03
@desc:
'''

###################################
#用于将带训练以及检验、测试的样本及标签先按照TFRecord协议格式存储并解析存储好的TFRecord文件
#若样本少可以只存为一个TFRecord文件，一次读取该文件
#若样本量大可以存储为多个TFRecord文件，采用多线程批次读取这些TFRecord文件
###################################

import tensorflow as tf
import numpy as np
import xlrd


def Excel2Numpy(p):
    '''
    将表格文件改成ndarray类型矩阵
    :param p: 文件路径
    :return: ndarray类型矩阵
    '''
    np.set_printoptions(suppress=True)
    data = xlrd.open_workbook(p)
    table = data.sheets()[0]
    row = 1
    while 1:
        try:
            data = np.array(table.row_values(row, start_colx=0, end_colx=None)) if row == 1 else \
                np.vstack((data, table.row_values(row, start_colx=0, end_colx=None)))
            row += 1
        except IndexError:
            break
    features = data[::, :-1]
    targets = data[::, -1]
    return features, targets

class FileoOperation:
    '''存取TFRecord文件及相关操作封装,输入属性为p_in, filename, read_in_fun, num_shards, instance_per_shard, ftype, ttype, fshape, tshape,
                 batch_size, capacity, batch_fun, batch_step, min_after_dequeue(choice)'''

    #p_in为
    #read_in_fun为处理使得p_in路径文件为python可操作文件的转换函数,返回处理好的数据
    ##将海量数据写入不同的TFRecord文件,num_shards定义了总共写入多少个TFRecord文件，
    #instances_per_shard定义了每个TFRecord文件中有多少个数据。
    #ftype, ttype分别为特征和标签转换为TFRecord文件前的原始数据类型
    #fshape, tshape分别为转换为TFRecord文件前的原始单个特征和的单个标签形状
    #batch_size为处理后的样本特征向量和标签数据整理成神经网络训练时需要的batch
    #capacity为定义一个batch中样例的个数
    #min_after_dequeue为出队时队列中元素的最少个数,当出队函数被调用但是队列中元素不够时，出队操作将等待更多的元素入队才会完成且
    #Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
    #batch_fun为待选择的出队数据组合函数，选择在数据出队前是否需要打乱
    #batch_step为tf.train.shuffle_batch或tf.train.batch函数循环输出batch组合的次数
    __slots__ = ('__p_in', '__filename', '__read_in_fun', '__num_shards',
                 '__instances_per_shard', '__features', '__targets', '__ftype', '__ttype',
                 '__fshape', '__tshape', '__batch_size', '__capacity', '__min_after_dequeue',
                 '__batch_fun', '__batch_step')

    @staticmethod
    # 用于将数据存储为TFRecord文件时的数据类型格式转换，生成字符串型的属性
    def bytes_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    def __init__(self, p_in, filename, read_in_fun, num_shards, instance_per_shard, ftype, ttype, fshape, tshape,
                 batch_size, capacity, batch_fun, batch_step, min_after_dequeue = 0):
        '''
        :param p_in: 读入文件名
        :param filename: 转化为TFRecord文件名或格式化文件名（若存储多个文件）
        :param read_in_fun: 处理使得p_in路径文件为python可操作文件的转换函数,返回处理好的数据
        :param num_shards: 总共写入多少个TFRecord文件
        :param instance_per_shard: 每个TFRecord文件中有多少个数据
        :param ftype: 特征转换为TFRecord文件前的原始数据类型
        :param ttype: 标签转换为TFRecord文件前的原始数据类型
        :param fshape: 转换为TFRecord文件前的原始的单个特征形状
        :param tshape: 转换为TFRecord文件前的原始的单个标签形状
        :param batch_size: 处理后的样本特征向量和标签数据整理成神经网络训练时需要的batch
        :param capacity: tf.train.shuffle_batch函数所需参数
        :param batch_fun: 待选择的出队数据组合函数，选择在数据出队前是否需要打乱
        :param batch_step: tf.train.shuffle_batch或tf.train.batch函数循环输出batch组合的次数
        :param min_after_dequeue: 出队时队列中元素的最少个数,当出队函数被调用但是队列中元素不够时，出队操作将等待更多的元素入队才会完成且
        Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
        '''
        self.__p_in = p_in
        #转换为TFRecord文件所需属性
        self.__filename = filename
        self.__read_in_fun = read_in_fun
        self.__num_shards = num_shards
        self.__instances_per_shard = instance_per_shard
        #读入数据及标签
        self.__features, self.__targets = self.__read_in_fun(self.__p_in)

        #解析读取TFRecord文件数据所需属性
        self.__ftype = ftype
        self.__ttype = ttype
        self.__fshape = fshape
        self.__tshape = tshape
        self.__batch_size = batch_size
        self.__capacity = capacity
        self.__min_after_dequeue = min_after_dequeue
        self.__batch_fun = batch_fun
        self.__batch_step = batch_step

    def file2TFRecord(self):
        '''将数据写入多个TFRecord文件中'''
        for i in range(self.__num_shards):
            # 将数据趣味多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分。其中m
            # 表示了数据总共被存在了多少个文件中，n表示当前文件的编号。
            filename = self.__filename % (i, self.__num_shards)
            writer = tf.python_io.TFRecordWriter(filename)
            for index in range(i * self.__instances_per_shard, (i + 1) * self.__instances_per_shard):
                # 将特征向量转化成一个字符串
                features_raw = self.__features[index].tostring()
                targets_raw = self.__targets[index].tostring()

                # 将一个样例转化为Example Protocol Buffer, 并将所有的信息写入这个数据结构
                example = tf.train.Example(features=tf.train.Features(feature={
                    'target_raw': FileoOperation.bytes_feature(targets_raw),
                    'feature_raw': FileoOperation.bytes_feature(features_raw),

                }))
                # 将一个Example写入TFRecord文件
                writer.write(example.SerializeToString())
            writer.close()

    def ParseDequeue(self, files):
        '''
        解析所有TFRecord文件并按照自行选择的方式处理并出队
        参数files为tf.train.match_filenames_once函数中参数pattern：匹配各个文件前部分的正则表达式
        :param files: 正则表达式，所有分文件的文件名前面相同部分
        :return: 一个batch的特征矩阵和标签向量
        '''
        files = tf.train.match_filenames_once(files)

        # Note: if num_epochs is not None, this function creates local counter epochs.
        # Use local_variables_initializer() to initialize local variables.
        filename_queue = tf.train.string_input_producer(files, shuffle=False)

        # 创建一个reader来读取TFRecord文件中的样例
        reader = tf.TFRecordReader()

        # Returns the next record (key, value) pair produced by a reader.
        _, serialized_example = reader.read(filename_queue)

        # 解析读入的一个样例。如果需要解析多个样例，可以用parse_example函数
        features = tf.parse_single_example(
            serialized_example,
            features={
                # 这里解析数据格式要和读入TFRecord时数据转化的格式一致Shape of input data. dtype: Data type of input.
                'target_raw': tf.FixedLenFeature([], tf.string),
                'feature_raw': tf.FixedLenFeature([], tf.string)
            }
        )

        # tf.decode_raw可以将字符串解析成feature_raw所对应的数组，此处一定要按照features字典中键值对的顺序来解析否则报错
        target = tf.decode_raw(features['target_raw'], self.__ttype)
        feature = tf.decode_raw(features['feature_raw'], self.__ftype)

        # pre-defined shape
        target.set_shape(self.__tshape)
        feature.set_shape(self.__fshape)

        # 使用train.shuffle_batch函数来组合样例This function adds the following to the current Graph
        if self.__batch_fun == tf.train.shuffle_batch:
            feature_batch, target_batch = tf.train.shuffle_batch([feature, target], batch_size= self.__batch_size,
                                                             capacity= self.__capacity, min_after_dequeue= self.__min_after_dequeue)
        else:
            # 使用train.shuffle_batch函数读出未打乱顺序的样本
            feature_batch, target_batch = tf.train.batch([feature, target], batch_size=self.__batch_size, capacity=self.__capacity)


        # If enqueue_many is False, tensors is assumed to represent a single example.
        # An input tensor with shape [x, y, z] will be output as a tensor with shape [batch_size, x, y, z].
        # 结果为Tensor("shuffle_batch:0", shape= (4, 4), dtype= float64) Tensor("shuffle_batch: 1", shape= (4, 1), dtype= float64)

        return feature_batch, target_batch

    def testfun(self, files):
        '''
        用于对解析后的数据进行测试
        :param files: ParseDequeue函数所需参数
        :return: 多线程生成特征矩阵和标签向量
        '''
        feature_batch, target_batch = self.ParseDequeue(files)
        with tf.Session() as sess:
            # 在使用tf.train。match_filenames_once函数时需要初始化一些变量
            sess.run(tf.local_variables_initializer())
            # sess.run(tf.global_variables_initializer())

            # 线程调配管理器
            coord = tf.train.Coordinator()
            # Starts all queue runners collected in the graph.
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # 获取并打印组合之后的样例
            # 由于tf.train。match_filenames_once函数机制:
            # The returned operation is a dequeue operation and will throw
            # tf.errors.OutOfRangeError if the input queue is exhausted.If
            # this operation is feeding another input queue, its queue runner
            # will catch this exception, however, if this operation is used
            # in your main thread you are responsible for catching this yourself.
            # 故需要在循环读取时及时捕捉异常
            train_steps = self.__batch_step
            try:
                while not coord.should_stop():  # 如果线程应该停止则返回True
                    cur_feature_batch, cur_target_batch = sess.run([feature_batch, target_batch])
                    print(cur_feature_batch, cur_target_batch)

                    train_steps -= 1
                    if train_steps <= 0:
                        coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

            except tf.errors.OutOfRangeError:
                print('Done training epoch limit reached')
            finally:
                # When done, ask the threads to stop. 请求该线程停止
                coord.request_stop()
                # And wait for them to actually do it. 等待被指定的线程终止
                coord.join(threads)



if __name__ == '__main__':


    # 只用于print对象输出非科学计数法数值
    np.set_printoptions(suppress=True)

    #类中需要输入参数p_in, filename, read_in_fun, num_shards, instance_per_shard, ftype, ttype, fshape, tshape,
    #batch_size, capacity, batch_fun, batch_step, min_after_dequeue(choice)

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
    fileop.file2TFRecord()
    # feature_batch, target_batch = fileop.ParseDequeue(r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\output.tfrecords-*')
    # print(feature_batch)
    # fileop.testfun(r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\output.tfrecords-*')