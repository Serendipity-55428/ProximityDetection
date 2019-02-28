#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: ceshi
@time: 2019/2/26 22:45
@desc:
'''
import tensorflow as tf
import numpy as np
import os.path
import os
def fun():
    a = 1

    def fun1(fun_2):
        fun_2()
        return fun_2

    def fun_2():
        print(3)

    fun1(fun_2)


if __name__ == '__main__':
    fun()

