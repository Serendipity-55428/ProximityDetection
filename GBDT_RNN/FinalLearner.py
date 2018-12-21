#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: FinalLearner
@time: 2018/12/21 22:43
@desc:
'''
import tensorflow as tf
import numpy as np
from cnn_rnn.HyMultiNN import RecurrentNeuralNetwork, FCNN
from cnn_rnn.Fmake2read import FileoOperation
import pickle
import os
from tensorflow.python.framework import graph_util

def variable_summaries(var, name):
    '''
    监控指标可视化函数
    :param var: Variable 类型变量
    :param name: 变量名
    :return: None
    '''
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

def sub_LossOptimize(net, target, optimize_function, learning_rate):
    '''
    对子学习器做损失函数的优化过程
    :param net: 网络最终的ops
    :param target: 批次数据标签
    :param optimize_function: 自选优化函数
    :param learning_rate: 学习率
    :return: 损失函数和优化损失函数的操作结点ops
    '''
    with tf.name_scope('loss_optimize'):
        loss = tf.reduce_mean(tf.square(net - target))
        # tf.summary.scalar('loss', loss)
        optimize = optimize_function(learning_rate= learning_rate).minimize(loss)
    return optimize, loss

def Ensemble_GRU(x, num_units, arg_dict, name):
    '''
    梯度提升树中最后一级GRU/LSTM子学习器
    :param x: type= 'ndarray' / 'Tensor'
    :param num_units: lstm/gru隐层神经元数量
    :param arg_dict: 全连接层权重以及偏置量矩阵散列
    :param name: 计算节点命名
    :return: MULSTM模型最终输出
    '''
    with tf.name_scope('multi_LSTMorGRU'):
        # 生成RecurrentNeuralNetwork对象

        #一层一对一输出隐层状态的GRU/LSTM,一层多对一输出隐层状态的GRU/LSTM,
        # 衔接一层神经元结点为上一层一半的fc层，再衔接一层神经元数量为上一层一半的fc层
        recurrentnn = RecurrentNeuralNetwork(x, keep_prob=0.8)
        # 添加layer_num层LSTM结点组合
        # LSTM
        # cells = recurrentnn.multiLSTM(net_name='LSTM', num_unit=num_units, layer_num=2)
        # GRU
        cells = recurrentnn.multiLSTM(net_name='GRU', num_unit= num_units, layer_num= 2)
        # outputs.shape= [batch_size, max_time, hide_size]
        # (lstm)multi_state= ((h, c), (h, c)), (gru)multi_state= (h, h) h.shape= [batch_size, hide_size]
        outputs, multi_state = recurrentnn.dynamic_rnn(cells, x, max_time= 5) #若特征数24则分成6份，若特征数20则分成5份
        # LSTM
        # result = multi_state[-1].h
        # GRU
        result = multi_state[-1]
        # 生成FCNN对象

    with tf.name_scope('fc'):
        fcnn = FCNN(result, keep_prob=1.0)
        net_1 = fcnn.per_layer(arg_dict['w_1'], arg_dict['b_1'])
        net_2 = fcnn.per_layer(arg_dict['w_2'], arg_dict['b_2'], param= net_1)
        out = fcnn.per_layer(arg_dict['w_3'], arg_dict['b_3'], param= net_2, name= name)
    return out

def LoadFile(p):
    '''读取文件'''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data[:, :-1], data[:, -1]

def GBDT_main():
    '''
    GBDT策略主函数
    :return: None
    '''

    # 训练集数据所需参数
    tr_p_in = 0
    tr_filename = 0
    tr_read_in_fun = 0
    tr_num_shards = 0
    tr_instance_per_shard = 0
    tr_ftype = 0
    tr_ttype = 0
    tr_fshape = 0
    tr_tshape = 0
    tr_batch_size = 0
    tr_capacity = 0
    tr_batch_fun = 0
    tr_batch_step = 0
    tr_files = 0

    # 测试集数据所需参数
    te_p_in = 0
    te_filename = 0
    te_read_in_fun = 0
    te_num_shards = 0
    te_instance_per_shard = 0
    te_ftype = 0
    te_ttype = 0
    te_fshape = 0
    te_tshape = 0
    te_batch_size = 0
    te_capacity = 0
    te_batch_fun = 0
    te_batch_step = 0
    te_files = 0

    with tf.name_scope('data_batch'):
        # 定义读取训练集数据对象
        train_fileoperation = FileoOperation(tr_p_in, tr_filename, tr_read_in_fun, tr_num_shards, tr_instance_per_shard,
                                             tr_ftype, tr_ttype, tr_fshape, tr_tshape, tr_batch_size, tr_capacity,
                                             tr_batch_fun, tr_batch_step)
        train_fileoperation.file2TFRecord()

        train_feature_batch, train_target_batch = train_fileoperation.ParseDequeue(tr_files)

        # 定义读取测试集数据对象
        test_fileoperation = FileoOperation(te_p_in, te_filename, te_read_in_fun, te_num_shards, te_instance_per_shard,
                                            te_ftype, te_ttype, te_fshape, te_tshape, te_batch_size, te_capacity,
                                            te_batch_fun, te_batch_step)
        test_fileoperation.file2TFRecord()

        test_feature_batch, test_target_batch = test_fileoperation.ParseDequeue(te_files)

    with tf.name_scope('x-y'):
        # 训练数据批次占位符,占位符读入数据形状和一个批次的数据特征矩阵形状相同
        x = tf.placeholder(dtype=tf.float32, shape=[tr_batch_size, tr_fshape], name= 'x')
        y = tf.placeholder(dtype=tf.float32, shape=[tr_batch_size, tr_tshape], name= 'y')

    #############################GRU/LSTM###############################
    # 定义GRU子学习器中全连接层参数矩阵以及偏置量尺寸
    with tf.name_scope('gru_weights'):
        gru_weights = {
            'w_1': tf.Variable(tf.truncated_normal([256, 128], mean=0, stddev=1.0), dtype=tf.float32),
            # 256为GRU网络最终输出的隐藏层结点数量
            'w_2': tf.Variable(tf.truncated_normal([128, 64], mean=0, stddev=1.0), dtype=tf.float32),
            'b_1': tf.Variable(tf.truncated_normal([128], mean=0, stddev=1.0), dtype=tf.float32),
            'b_2': tf.Variable(tf.truncated_normal([64], mean=0, stddev=1.0), dtype=tf.float32),
            'w_3': tf.Variable(tf.truncated_normal([64, 1], mean=0, stddev=1.0), dtype=tf.float32),
            'b_3': tf.Variable(tf.truncated_normal([1], mean=0, stddev=1.0), dtype=tf.float32)
        }

        variable_summaries(gru_weights['w_1'], 'w_1')
        variable_summaries(gru_weights['w_2'], 'w_2')
        variable_summaries(gru_weights['w_3'], 'w_3')
        variable_summaries(gru_weights['b_1'], 'b_1')
        variable_summaries(gru_weights['b_2'], 'b_2')
        variable_summaries(gru_weights['b_3'], 'b_3')

    with tf.name_scope('gru_ops'):
        # 定义多层GRU的最终输出ops(待保存计算节点)
        gru_ops = Ensemble_GRU(x=x, num_units=256, arg_dict=gru_weights, name='gru_op')

    with tf.name_scope('gru_optimize-loss'):
        # 定义GRU自学习器的损失函数和优化器
        gru_optimize, gru_loss = sub_LossOptimize(gru_ops, y, optimize_function=tf.train.RMSPropOptimizer,
                                                      learning_rate=1e-4)
        tf.summary.scalar('gru_loss', gru_loss)

        train_steps = tr_batch_step  # 将所有最后一级学习器所需要的训练集数据分5批读入时的循环计数变量

        # 循环训练次数和总轮数
        epoch, loop = 1, 1000

        # 摘要汇总
        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #############################Session###########################
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # 摘要文件
        summary_writer = tf.summary.FileWriter('logs/', sess.graph)

        # 线程调配管理器
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                # 批量读取次级学习器训练集特征和标签数据
                tr_feature_batch, tr_target_batch = sess.run([train_feature_batch, train_target_batch])

                summary = sess.run(merged, feed_dict={x: tr_feature_batch, y: tr_target_batch})

                _ = sess.run(gru_optimize, feed_dict={x: tr_feature_batch, y: tr_target_batch})
                if not (train_steps % 5):
                    loss_fc = sess.run(gru_loss, feed_dict={x: tr_feature_batch, y: tr_target_batch})
                    print('FC次级学习器损失函数在第 %s 个epoch的数值为: %s' % (epoch, loss_fc))

                    # 对测试集进行acc（loss）计算
                    test_steps = 2
                    for i in range(test_steps):
                        # 批量读取次级学习器测试集特征和标签数据
                        te_feature_batch, te_target_batch = sess.run([test_feature_batch, test_target_batch])
                        loss_fc = sess.run(gru_loss, feed_dict={x: te_feature_batch, y: te_target_batch})
                        # 对每个批次的测绘集数据进行输出
                        print('FC次级学习器预测损失在第 %s 个epoch的数值为: %s' % (epoch, loss_fc))

                    # 在train_steps为5的倍数时(5个批次的测试集已经全部读入预测后)更新
                    summary_writer.add_summary(summary, epoch)
                    epoch += 1

                train_steps += 1
                if train_steps > 5 * loop:
                        coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

        except tf.errors.OutOfRangeError:
            print('次级训练器训练结束!')
        finally:
            # When done, ask the threads to stop. 请求该线程停止
            coord.request_stop()
            # And wait for them to actually do it. 等待被指定的线程终止
            coord.join(threads)

            # 关闭摘要
            summary_writer.close()

            # 获取pb文件保存路径前缀
            pb_file_path = os.getcwd()

            # 存储计算图为pb格式
            # Replaces all the variables in a graph with constants of the same values
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                           ['{gru_name}'.format(gru_name=gru_ops.op.name)])
            # 写入序列化的pb文件
            with tf.gfile.FastGFile(pb_file_path + 'model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            # Builds the SavedModel protocol buffer and saves variables and assets
            # 在和project相同层级目录下产生带有savemodel名称的文件夹
            builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path + 'sec_savemodel')
            # Adds the current meta graph to the SavedModel and saves variables
            # 第二个参数为字符列表形式的tags – The set of tags with which to save the meta graph
            builder.add_meta_graph_and_variables(sess, ['cpu_server_1'])
            # Writes a SavedModel protocol buffer to disk
            #此处p值为生成的文件夹路径
            p = builder.save()
            print('fc次级子学习器模型节点保存路径为: ', p)
            print('节点名称为: ' + '{gru_name}'.format(gru_name=gru_ops.op.name))




