#!/usr/bin/env python
# coding: utf-8

# In[ ]:


######################主程序######################
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pylab

import time
from PIL import Image
import random
from pathlib2 import Path  # python3环境下
# from pathlib import Path  #python2环境下
import os
from tvtk.api import tvtk, write_data
import threading
from time import sleep
from tqdm import tqdm
import difflib
import itertools as it
import multiprocessing
#################################本程序自带子程序########################
from NEM import *
from TI import *
from probcombine_entropy2 import *
import heapq


def extractlist(Ti, listvalue):  # 提取单个训练图像中所有值的类型
    # 输入：
    # Ti--输入剖面
    # listvalue--空列表
    # 输出：
    # listvalue--训练图像中包含的值
    for x in range(Ti.shape[0]):
        for y in range(Ti.shape[1]):
            if Ti[x, y] not in listvalue:
                listvalue.append(Ti[x, y])
    return listvalue


def Tilistvalueextract():  # 自动提取训练图像中所有值类型
    file1 = open('./Ti/Tiparameter.txt')
    content = file1.readline()
    string1 = [i for i in content if str.isdigit(i)]
    num = int(''.join(string1))
    print('剖面数目：')
    print (num)
    valuelist = []
    for n in range(num):
        path = './Ti/' + str(n + 1) + '.bmp'
        section = cv2.imread(path, 0)
        valuelist = extractlist(section, valuelist)
    print(valuelist)
    return valuelist


def simgridex(m, beilv):  # 放大或缩小模型
    # 输入：
    # m--网格模型
    # beilv--放大或者缩小的倍率
    # 输出：
    # dst-缩放后的模型
    H0 = m.shape[0]
    W0 = m.shape[1]
    L0 = m.shape[2]

    dstHeight = int(H0 * beilv)
    dstWidth = int(W0 * beilv)
    dstLength = int(L0 * beilv)

    dst = np.zeros((dstHeight, dstWidth, dstLength), int)
    for i in range(0, dstHeight):
        for j in range(0, dstWidth):
            for k in range(0, dstLength):
                iNew = int(i * (H0 * 1.0 / dstHeight))
                jNew = int(j * (W0 * 1.0 / dstWidth))
                kNew = int(k * (L0 * 1.0 / dstLength))
                print(dst.shape, iNew, jNew, kNew)
                dst[i, j, k] = m[iNew, jNew, kNew]
    return dst


def Pythia(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U, hardlist, code, valuels, scale, epoch,
           flaglist, jvalue):
    # 皮提亚主程序
    # code 需要事先定义好TI中不包含的
    # 输入：m--空网格  template_h, template_x, template_y--模板大小  lag, lag_h, lag_x, lag_y--重叠区大小
    # N--备选模式个数 U--分类阈值
    # 输出：m--最终模型

    m, Tilist, Tizuobiaolist, codelist = sectionloadandextendG(m, template_x, template_y, 0, 1, jvalue)#剖面导入
    # 待插入二维高程建模区域约束
    codelist.append(code)

    print('当前合理地层层序列表：', codelist)
    print('当前模拟地层顺序：', valuels)
    m = np.load('./output1/outputinitial3.npy')
    m = initialAIforPythia(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U, hardlist, codelist,
                           valuels, flaglist)
    # 初始化部分结束

    # em迭代部分

    sancheck = 1  # sectionloadandextend倍率机制
    np.save('./output1/initial.npy', m)
    # EM迭代阶段
    for ni in range(len(scale)):
        sancheck = sancheck * scale[ni]
        # 构建新初始网格mm
        mm = -np.ones((int(m.shape[0] * scale[ni]), int(m.shape[1] * scale[ni]), int(m.shape[2] * scale[ni])), int)

        Tilist = []  # 储存Ti范围内矩阵组成的列表
        Tizuobiaolist = []  # 储存坐标矩阵的列表
        mm, Tilist, Tizuobiaolist, codelist = sectionloadandextendG(mm, patternSizex, patternSizey, 1, sancheck, jvalue)

        mm = extendTimodel(mm, patternSizeh, patternSizex, patternSizey)
        # 上一个尺度升采样
        m = simgridex(m, scale[ni])
        # 重新导入
        for hi in range(m.shape[0]):
            for xi in range(m.shape[1]):
                for yi in range(m.shape[2]):
                    if mm[hi, xi, yi] != -1:
                        m[hi, xi, yi] = mm[hi, xi, yi]
        print("该尺度扩展Ti完成")
        time_start = time.time()  # 计时开始

        CTI = []  # 检测使用率剖面

        m, CTI = Recodepatchmatch(m, mm, Tilist, Tizuobiaolist, size, itr, 4, 0)  # 并行进程的数目

        path = "./output1/reconstruction.npy"
        np.save(path, m)
        time_end = time.time()

        print("该尺度优化完成")
        print('timecost:')
        print(time_end - time_start)

        data = m.transpose(-1, -2, 0)  # 转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                              dimensions=data.shape)
        grid.point_data.scalars = np.ravel(data, order='F')
        grid.point_data.scalars.name = 'lithology'
        write_data(grid, './output1/output.vtk')


#################################主程序########################
################################计时程序########################################################
time_start1 = time.time()  # 计时开始

######################################参数读取阶段################################################################


file1 = open('./parameter.txt')
content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
Mh = int(''.join(string1))
# print Mh

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
Mx = int(''.join(string1))
# print Mx

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
My = int(''.join(string1))
# print My

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
lag = int(''.join(string1))
# print lag

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
lag_h = int(''.join(string1))
# print lag_h

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
lag_x = int(''.join(string1))
# print lag_x

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
lag_y = int(''.join(string1))
# print lag_y

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
patternSizeh = int(''.join(string1))
# print patternSizeh

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
patternSizex = int(''.join(string1))
# print patternSizex

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
patternSizey = int(''.join(string1))
# print patternSizey

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
U = int(''.join(string1))
# print U

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
N = int(''.join(string1))
# print N

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
size = int(''.join(string1))
# print size

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
itr = int(''.join(string1))
# print itr


content = file1.readline()
scale = []
for i in content:
    if str.isdigit(i):
        scale.append(int(i))
# print scale

# 一次模拟的个数
content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
Modelcount = int(''.join(string1))
# print Modelcount
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
#####################预处理阶段################################################################
epoch = 10000

valuels = []
valuels = Tilistvalueextract()
valuels.sort(reverse=True)
flaglist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # 用来判断地层为局部或者全局
print (valuels)
valuels = [3,5,7,161,120,110,100,90,80,70,60,50,40,31,20]  # Ti中包含的地层标签值
print('请自行进行排序:', valuels)
code = [3,5,7,161,120,110,100,90,80,70,60,50,40,31,20]  # 模拟顺序
hardlist = [5,3,7]  # 强约束
jvalue = 29  # 占比极小值
m = -np.ones((Mh, Mx, My), int)  # 默认空值为-1

Pythia(m, patternSizeh, patternSizex, patternSizey, lag, lag_h, lag_x, lag_y, N, U, hardlist, code, valuels, scale,
       epoch, flaglist, jvalue)

time_end1 = time.time()  # 计时结束
print('总耗时：', time_end1 - time_start1)
