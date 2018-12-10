#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: PyCharm
@file: dataoperation.py
@time: 2018/11/10 10:20
@desc:
'''
import xlrd
import xlwt
import numpy as np
import os.path
import pickle
import pandas as pd

def FileArrangement(p):
    '''将excel数据整理成向量格式'''
    np.set_printoptions(suppress=True)
    workbook = xlrd.open_workbook(p)
    table = workbook.sheets()[0]
    Tthreshold = table.row_values(1, start_colx=2, end_colx=None)
    # 获得Tthreshold列向量shape: [10, 1]
    Tthreshold = np.array(Tthreshold)[:, np.newaxis]
    # 数据行数
    row_datagroup = table.nrows
    for i in range(2, row_datagroup):
        row = table.row_values(i, start_colx=0, end_colx=2)
        radius = table.row_values(i, start_colx=2, end_colx=None)
        # 获得半径列向量shape: [10, 1]
        radius = np.array(radius)[:, np.newaxis]
        # 获得对应objects, friends
        obfr = np.array(row)
        for _ in range(radius.shape[0] - 1):
            obfr = np.vstack((obfr, np.array(row)))  # obfr张量shape: [10, 2]
        # 拼接属性和target
        subdata = np.hstack((obfr, Tthreshold))
        subdata = np.hstack((subdata, radius))  # database张量shape: [10, 4] [objects, friends, Threshold, radius]
        data = subdata if i == 2 else np.vstack((data, subdata))
    return data #shape: [200, 4] [objects, friends, Threshold, radius]

def Excel2Numpy(p): #改后数据处理
    '''OLDENBURG表格数据转换为numpy'''
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
    return data

def SaveFile(data):
    '''存储整理好的数据'''

    p = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\COST_PNY.pickle' #PNY和OLDBURG
    # p = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\COST_OLDBURG.pickle'

    if not os.path.exists(p):
        with open(p, 'wb') as file:
            pickle.dump(data, file)

def LoadFile(p):
    '''读取文件'''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data

def MakeExcel(data):
    dataframe = pd.DataFrame(data, index= list(range(1, 201)), columns=['objects', 'friends', 'Tthreshold', 'radius'])
    # dataframe.to_excel(r'C:\Users\xiaosong\Desktop\TeamProject\COST_PNY_1.xls') #PNY和OLDBURG
    # dataframe.to_excel(r'C:\Users\xiaosong\Desktop\TeamProject\COST_OLDBURG_1.xls')

def CostError(statistic):
    '''将训练周期数、训练集损失函数误差以及时间统计量汇总为excel表格'''
    file = xlwt.Workbook()
    table = file.add_sheet('PNY_AdamOptimizer', cell_overwrite_ok=True) #sheet名需要根据优化器的不同而修改
    # table = file.add_sheet('PNY_GradientDescentOptimizer', cell_overwrite_ok=True) #PNY和OLDBURG
    # table = file.add_sheet('OLDBURG_GradientDescent', cell_overwrite_ok=True)
    # table = file.add_sheet('OLDBURG_AdamOptimizer', cell_overwrite_ok=True)

    # 建立列名:
    table.write(0, 0, 'epoch')
    table.write(0, 1, 'epoch_time')
    table.write(0, 2, 'train_loss')
    table.write(0, 3, 'test_loss')
    v, h = statistic.shape
    for i in range(v):
        for j in range(h):
            table.write(i + 1, j, str(statistic[i, j]))
    # PNY和OLDBURG根据不同优化器保存
    # file.save(r'C:\Users\xiaosong\Desktop\TeamProject\Error_static.xlsx')     #SGD
    file.save(r'C:\Users\xiaosong\Desktop\TeamProject\Error_static_1.xlsx')   #Ada
    # file.save(r'C:\Users\xiaosong\Desktop\TeamProject\Error_static_2.xlsx')     #SGD
    # file.save(r'C:\Users\xiaosong\Desktop\TeamProject\Error_static_3.xlsx')   #Ada



if __name__ == '__main__':
    # p = r'C:\Users\xiaosong\Desktop\TeamProject\COST_PNY.xlsx'  #PNY和OLDBURG
    # p = r'C:\Users\xiaosong\Desktop\TeamProject\COST_OLDBURG.xlsx' #数据有误
    # p = r'C:\Users\xiaosong\Desktop\TeamProject\Oldeburg_revise.xls' #修正后数据
    p = r'C:\Users\xiaosong\Desktop\TeamProject\Pny_revise.xls' #修正后数据
    # data = FileArrangement(p)
    # data = Excel2Numpy(p)
    # MakeExcel(data)
    # print(data)
    # print(data.shape)
    # SaveFile(data)
    # p1 = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\COST_PNY.pickle'
    # p1 = r'C:\Users\xiaosong\Anaconda3\envs\ml\Scripts\ProximityDetection\COST_OLDBURG.pickle'
    # data = LoadFile(p1)
    # print(data)
    # CostError(data)
