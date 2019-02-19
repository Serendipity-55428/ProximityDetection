#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: garner
@file: DataRead
@time: 2018/12/28 13:06
@desc:
'''
import os.path
import pickle
import numpy as np
import xlwt

def GeneratorData(type, filename_prefix):
    '''
    生成numpy矩阵数据
    :param type: 1/2(1:'PNY'/2:'OLDENBURG')
    :param filename_prefix: 所有待处理文件绝对路径前缀
    :return: numpy数据矩阵
    '''
    movingObjects = [1000, 1500, 2000, 4000, 6000, 8000, 10000, 20000, 40000, 60000, 80000, 100000]
    maxSpeed = ['v' + '%s' % i for i in range(1, 7)]
    maxSpeed = list(zip(maxSpeed, [28, 28, 90, 90, 149, 149]))  # 改
    friends = [5, 10, 20, 30, 40]
    Threshold = [i for i in range(1, 11)]
    #初始化待输出numpy数据矩阵
    dataset = np.zeros(shape=(1, 25))
    for file_movingObjects in [movingObjects[2]]: #改
        for file_maxSpeed in [maxSpeed[0]]:  # file_maxSpeed类型为元组 #改
            for file_friends in friends:
                for file_Threshold in Threshold:
                    # 前4个特征列表，需要和文件中20个特征、1个最优半径结合后转换为ndarray类型向量
                    sub_feature = [file_movingObjects, file_maxSpeed[-1], file_friends, file_Threshold]
                    # 数据文件命名格式：type_movingObjects_maxSpeed_friends_Threshold
                    p = filename_prefix + '\\' + r'%d_%d_%s_%d_%d.dat' % \
                        (type, file_movingObjects, file_maxSpeed[0], file_friends, file_Threshold)
                    with open(p, 'r') as f:
                        # 将文件转为字符串类型，切片可以自动消除‘\n'字符
                        file = f.readlines()
                        # print(file[2][-2])
                        for ts in range(5):
                            # 设置在各个r组内指向每一个cost的指针
                            inside_point = 5 + ts * 3
                            # 定义全局指针
                            point = inside_point
                            # 初始化存储每20倍数时刻最小代价值索引,初始化指向point所指向的位置
                            min_cost = point
                            while point <= 255:
                                if float(file[point][7:-1]) <= float(file[min_cost][7:-1]):
                                    min_cost = point
                                # 指向下一个r中同一时间的cost
                                point += 17

                            # min_cost位置回退2便指向对应20个密度特征，回退3便指向当前最优半径
                            twenty_feature = file[min_cost - 2][:-2]
                            twenty_feature = twenty_feature.split(' ')
                            twenty_feature = [float(i) for i in twenty_feature] if len(
                                [float(i) for i in twenty_feature]) == 20 else \
                                [float(i) for i in twenty_feature[1:]]
                            # 拼接特征向量
                            sub_feature.extend(twenty_feature)
                            sub_feature.append(float(file[min_cost - (ts + 1) * 3][4:-1]))
                            # print(len(sub_feature))
                            dataset = np.array(sub_feature) if dataset.any() == 0 else \
                                np.vstack((dataset, np.array(sub_feature)[np.newaxis, :]))

                            # 重置sub_feature列表为前4个特征
                            sub_feature = [file_movingObjects, file_maxSpeed[-1], file_friends, file_Threshold]
    return dataset

def Numpy2Excel(data, save_p, name= 'PNY'):
    '''
    将numpy数组转化为xls/xlsx
    :param data: 待转化数据
    :param save_p: 输出xls/xlsx绝对路径
    :return: xls/xlsx
    '''
    file = xlwt.Workbook()
    table = file.add_sheet(sheetname= name, cell_overwrite_ok=True)
    # 建立列名:
    table.write(0, 0, 'movingObjects')
    table.write(0, 1, 'maxSpeed')
    table.write(0, 2, 'friends')
    table.write(0, 3, 'Threshold')
    for i in range(1, 21):
        table.write(0, i+3, 'Density %s' % i)
    table.write(0, 24, 'optimal_r')

    for h in range(data.shape[0]):
        for v in range(data.shape[-1]):
            table.write(h+1, v, str(data[h, v]))

    file.save(save_p)

def SaveFile(data, savepickle_p):
    '''
    存储整理好的数据'
    :param data: 待存储数据
    :param savepickle_p: pickle后缀文件存储绝对路径
    :return: None
    '''''
    if not os.path.exists(savepickle_p):
        with open(savepickle_p, 'wb') as file:
            pickle.dump(data, file)


if __name__ == '__main__':
    type= 1
    filename_prefix = r'D:\proximityOnTimeAwareRN_generateFriends'
    dataset = GeneratorData(type= type, filename_prefix= filename_prefix)
    # zero_r, nonzero_r = 0, 0
    # r = dataset[:, -1]
    # for i in r:
    #     if i == 0.01:
    #         zero_r += 1
    # nonzero_r = len(r) - zero_r
    # print('最优半径为0.01个数: %s \n 最优半径不为0.01的个数: %s' % (zero_r, nonzero_r))
    # print(dataset)
    # print(dataset.shape)
    # print(dataset.dtype)
    #保存文件到Excel
    save_p = r'F:\anaconda\envs\machinelearning\Scripts\ProximityDetection\datasets.xlsx'
    Numpy2Excel(data= dataset, save_p= save_p)

    #对单个文件进行测试
    # dataframe = np.zeros(shape=(1, 25))
    # # 前4个特征列表，需要和文件中20个特征、1个最优半径结合后转换为ndarray类型向量
    # sub_feature = [1000, 28.593955, 5, 1]
    # # 数据文件命名格式：type_movingObjects_maxSpeed_friends_Threshold
    # p = r'D:\proximityOnTimeAwareRN_generateFriends\%d_%d_%s_%d_%d.dat' % \
    #     (1, 1000, 'v3', 5, 1)
    # with open(p, 'r') as f:
    #     # 将文件转为字符串类型，切片可以自动消除‘\n'字符
    #     file = f.readlines()
    #     # print(file[2][-2])
    #     for ts in range(5):
    #         # 设置在各个r组内指向每一个cost的指针
    #         inside_point = 5 + ts * 3
    #         # 定义全局指针
    #         point = inside_point
    #         # 初始化存储每20倍数时刻最小代价值索引,初始化指向point所指向的位置
    #         min_cost = point
    #         while point <= 255:
    #             if float(file[point][7:-1]) <= float(file[min_cost][7:-1]):
    #                 min_cost = point
    #             # 指向下一个r中同一时间的cost
    #             point += 17
    #
    #         # min_cost位置回退2便指向对应20个密度特征，回退3便指向当前最优半径
    #         twenty_feature = file[min_cost - 2][:-2]
    #         twenty_feature = twenty_feature.split(' ')
    #         twenty_feature = [float(i) for i in twenty_feature] if len([float(i) for i in twenty_feature]) == 20 else \
    #             [float(i) for i in twenty_feature[1:]]
    #         # 拼接特征向量
    #         sub_feature.extend(twenty_feature)
    #         sub_feature.append(float(file[min_cost - (ts + 1) * 3][4:-1]))
    #         # print(len(sub_feature))
    #         dataframe = np.array(sub_feature) if dataframe.any() == 0 else \
    #             np.vstack((dataframe, np.array(sub_feature)[np.newaxis, :]))
    #
    #         # 临时存储列表删除后面拼接部分
    #         sub_feature = [1000, 28.593955, 5, 1]
    #
    # print(dataframe.dtype)



















