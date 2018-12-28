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
import re
# type = [i for i in range(1, 3)]
# movingObjects = [1000, 1500, 2000, 4000, 6000, 8000, 10000, 20000, 40000, 60000, 80000, 100000]
# maxSpeed = ['v'+'%s' % i for i in range(1, 7)]
# friends = [5, 10, 20, 30, 40]
# Threshold = [i for i in range(1, 11)]

# for file_type in type:
#     for file_movingObjects in movingObjects:
#         for file_maxSpeed in maxSpeed:
#             for file_friends in friends:
#                 for file_Threshold in Threshold:
#                     # 数据文件命名格式：type_movingObjects_maxSpeed_friends_Threshold
#                     p = r'D:\proximityOnTimeAwareRN_generateFriends\%d_%d_%s_%d_%d.dat' % \
#                         (file_type, file_movingObjects, file_maxSpeed, file_friends, file_Threshold)
#                     with open(p, 'r') as f:
#                         for i in f:
#                             print(i)

# 数据文件命名格式：type_movingObjects_maxSpeed_friends_Threshold
p = r'D:\proximityOnTimeAwareRN_generateFriends\%d_%d_%s_%d_%d.dat' % \
    (1, 1000, 'v3', 5, 1)
with open(p, 'r') as f:
    #转为字符串类型，切片可以自动消除‘\n'字符
    file = f.readlines()
    for i in range(0, 20):
        if i % 17 == 2:
            print(file[i])
    # num = re.sub("\D", "", file[4][11:21])
    # print(float(file[4][11:20]))



