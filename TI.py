# coding=utf-8
######################输入并扩展TI
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

import heapq


def max_list(lt):  # 计算列表中出现次数最多的值
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str


def extend2dAI(m, h1, x1, y1):  # 9格子内选取分布最广的值
    listcs = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            c = m[h1, x1 + i, y1 + j]
            if c != -1:  # 默认空值为-1
                listcs.append(c)

    if len(listcs) >= 2:
        value = max_list(listcs)
    else:
        value = -1
    return value


def extendTimodelsave(m, template_h, template_x, template_y, order):  # 全自动拓展插入硬数据的待模拟网格并储存为npy格式
    # 输入
    # m--导入Ti后的模型  template_h, template_x, template_y--模板大小 order--序号
    # 输出
    # m--拓展Ti后的模型
    lag = max(template_h, template_x, template_y) // 2  # 计算重叠区
    m2 = np.pad(m, lag, 'edge') # m2扩展是为了方便边缘值估计，并不是返回值
    d = []
    for h in range(lag, m2.shape[0] - lag):
        for x in range(lag, m2.shape[1] - lag):
            for y in range(lag, m2.shape[2] - lag):
                d.append((h, x, y))  # 储存需要遍历的矩阵坐标

    for i in range(lag):  # 扩展TI到lag个长度
        flag = 0
        for n in range(len(d)):
            h = d[n][0]
            x = d[n][1]
            y = d[n][2]
            if m2[h, x, y] == -1:
                value = extend2dAI(m2, h, x, y)  # 扩展取值，Ti附近才能取到非空值
                flag = 1
                if value != -1:

                    m[h - lag, x - lag, y - lag] = value  # Ti附近扩展才有效

                else:
                    if i == lag - 1:
                        m[h - lag, x - lag, y - lag] = value
        if flag == 0: # 已经扩展完成就不需要再扩展
            break
        m2 = np.pad(m, lag, 'edge')  # 扩展已经Ti扩展完成的m的边缘
        # 填充为1的
    path = './output/ext' + str(order) + '.npy'
    np.save(path, m)
    return m


def sectionex(section, height, length):  # 剖面的缩放
    # 输入：
    # section--剖面数组  height, length--缩放后的高和长
    # 输出：
    # section--缩放后的剖面
    ns = section.shape[1]
    ns2 = section.shape[0]
    lv = length
    lv2 = height
    if ns2 != lv2:
        if ns2 > lv2:
            # 缩小至lv2高度
            beilv = float(lv2) / ns2
            section_new = np.zeros((height, section.shape[1]), int)
            for n in range(ns2):
                section_new[int(n * beilv), :] = section[n, :]

            section = section_new
        elif ns2 < lv2:
            # 扩大至lv长度
            beilv = ns2 / float(lv2)
            section_new = np.zeros((height, section.shape[1]), int)
            for n in range(lv2):
                section_new[n, :] = section[int(n * beilv), :]
            section = section_new
    if ns != lv:
        if ns > lv:
            # 缩小至lv长度
            beilv = float(lv) / ns
            section_new2 = np.zeros((section.shape[0], lv), int)
            for n in range(ns):
                section_new2[:, int(n * beilv)] = section[:, n]
            section = section_new2
        elif ns < lv:
            # 扩大至lv长度
            beilv = ns / float(lv)
            section_new2 = np.zeros((section.shape[0], lv), int)
            for n in range(lv):
                section_new2[:, n] = section[:, int(n * beilv)]
            section = section_new2

    return section


def sectionloadG(m, section, hz, hz2, xz, yz, xz2, yz2):  # 相对坐标的剖面导入，斜剖面导入后需要扩充才正确
    # 输入：
    # m--模型  section--剖面  xz,yz,xz2,yz2--剖面两端点的相对坐标
    # 输出：
    # m--导入剖面后的模型
    ns = section.shape[1]
    hc = float(hz2 - hz) + 1
    xc = float(xz2 - xz)
    yc = float(yz2 - yz)
    if xc < 0:
        xc1 = xc - 1
    else:
        xc1 = xc + 1
    if yc < 0:
        yc1 = yc - 1
    else:
        yc1 = yc + 1
    # 计量后加一为长度
    lv = int(max(abs(xc1), abs(yc1)))  # 比较长度绝对值大小，得到即为斜剖面需要填网格总数，所以需要加一
    xlv = xc / (lv - 1)
    ylv = yc / (lv - 1)
    x1 = xz
    y1 = yz
    # 对section的处理
    section = sectionex(section, int(hc), lv)

    for n in range(lv):
        m[hz:hz2 + 1, x1, y1] = section[:, n]
        # print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
        x1 = int(xz + (n + 1) * xlv + 0.5)  # 四舍五入
        y1 = int(yz + (n + 1) * ylv + 0.5)
    return m


def sectionex2(section, height, length, jivalue):  # 剖面的缩放
    # 功能与sectionex基本一样，多了保留jivalue值的功能
    ns = section.shape[1]  # 剖面原始长
    ns2 = section.shape[0]  # 剖面原始高
    lv = length  # 缩放后的长
    lv2 = height  # 缩放后的高
    if ns2 != lv2:
        if ns2 > lv2:
            # 缩小至lv2高度
            beilv = float(lv2) / ns2
            section_new = np.zeros((height, section.shape[1]), int)
            for n in range(ns2):
                for i in range(section_new.shape[1]):
                    if section_new[int(n * beilv), i] != jivalue:
                        section_new[int(n * beilv), i] = section[n, i]

            section = section_new
        elif ns2 < lv2:
            # 扩大至lv2长度
            beilv = ns2 / float(lv2)
            section_new = np.zeros((height, section.shape[1]), int)
            for n in range(lv2):
                section_new[n, :] = section[int(n * beilv), :]
            section = section_new
    if ns != lv:
        if ns > lv:
            # 缩小至lv长度
            beilv = float(lv) / ns
            section_new2 = np.zeros((section.shape[0], lv), int)
            for n in range(ns):
                for i in range(section_new2.shape[0]):
                    if section_new2[i, int(n * beilv)] != jivalue:
                        section_new2[i, int(n * beilv)] = section[i, n]
            section = section_new2
        elif ns < lv:
            # 扩大至lv长度
            beilv = ns / float(lv)
            section_new2 = np.zeros((section.shape[0], lv), int)
            for n in range(lv):
                section_new2[:, n] = section[:, int(n * beilv)]
            section = section_new2

    return section


def RecodeTIextendforEMG(section, m, template_x, template_y, h1, h2, x1, y1, x2, y2):  # EM迭代用剖面提取
    # 输入：
    # section--剖面 m--已经导入Ti的模拟网格 template_x, template_y--模板大小  x,y为固定剖面的坐标
    # 输出：
    # Ti--剖面范围内的矩阵 Tizuobiao--坐标矩阵
    dx = []  # 储存X方向的空列表
    dy = []  # 储存Y方向的空列表
    dh = []  # 储存H方向的空列表
    lag = max(template_x, template_y) // 2
    ms = -np.ones((m.shape[0], m.shape[1], m.shape[2]), int)
    sectionloadG(ms, section, h1, h2, x1, y1, x2, y2)  # 独立载入防止造成多剖面混乱
    Tizuobiaox = -np.ones((m.shape[0], m.shape[1], m.shape[2]), int)  # 储存X方向的空矩阵
    Tizuobiaoy = -np.ones((m.shape[0], m.shape[1], m.shape[2]), int)  # 储存Y方向的空矩阵
    Tizuobiaoh = -np.ones((m.shape[0], m.shape[1], m.shape[2]), int)  # 储存H方向的空矩阵
    for h in range(Tizuobiaoh.shape[0]):
        for x in range(Tizuobiaoh.shape[1]):
            for y in range(Tizuobiaoh.shape[2]):
                Tizuobiaoh[h, x, y] = h
    if abs(h1 - h2) >= lag:
        for i in range(min(h1, h2), max(h1, h2) + 1):
            dh.append(i)
    else:
        for i in range(max(0, min(h1, h2) - lag), min(max(h1, h2) + lag, m.shape[1] - 1) + 1):
            dh.append(i)

    if abs(x1 - x2) >= lag:
        for i in range(min(x1, x2), max(x1, x2) + 1):
            dx.append(i)
    else:
        for i in range(max(0, min(x1, x2) - lag), min(max(x1, x2) + lag, m.shape[1] - 1) + 1):
            dx.append(i)

    if abs(y1 - y2) >= lag:
        for i in range(min(y1, y2), max(y1, y2) + 1):
            dy.append(i)
    else:
        for i in range(max(0, min(y1, y2) - lag), min(max(y1, y2) + lag, m.shape[2] - 1) + 1):
            dy.append(i)

    for h in range(ms.shape[0]):
        for x in range(ms.shape[1]):
            for y in range(ms.shape[2]):
                if ms[h, x, y] != -1:
                    Tizuobiaox[h, x, y] = x
                    Tizuobiaoy[h, x, y] = y
    temp = ms[:, dx, :]
    fowt = temp[:, :, dy]
    fow = fowt[dh, :, :]  # 储存包含特征值范围的矩阵
    Tizuobiaoxt = Tizuobiaox[:, dx, :]
    Tizuobiaoxx = Tizuobiaoxt[:, :, dy]
    Tizuobiaox = Tizuobiaoxx[dh, :, :]  # 储存包含特征值范围的X方向坐标矩阵
    Tizuobiaoyt = Tizuobiaoy[:, dx, :]
    Tizuobiaoyy = Tizuobiaoyt[:, :, dy]
    Tizuobiaoy = Tizuobiaoyy[dh, :, :]
    Tizuobiaoht = Tizuobiaoh[:, dx, :]
    Tizuobiaohh = Tizuobiaoht[:, :, dy]
    Tizuobiaoh = Tizuobiaohh[dh, :, :]
    c = max(fow.shape[1], fow.shape[2]) # 找出Ti矩阵最长边

    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=extendTimodelsave, args=(fow, c, c, c, 1))  # 扩展导入Ti的矩阵

    p2 = multiprocessing.Process(target=extendTimodelsave, args=(Tizuobiaox, c, c, c, 2))
    p3 = multiprocessing.Process(target=extendTimodelsave, args=(Tizuobiaoy, c, c, c, 3))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

    Tizuobiao = []  # 坐标矩阵列表
    Tizuobiaox = np.load('./output/ext2.npy')
    Tizuobiaoy = np.load('./output/ext3.npy')
    for h in range(Tizuobiaox.shape[0]):
        for x in range(Tizuobiaox.shape[1]):
            for y in range(Tizuobiaox.shape[2]):
                zb = np.array([Tizuobiaoh[h, x, y], Tizuobiaox[h, x, y], Tizuobiaoy[h, x, y]])
                Tizuobiao.append(zb)

    Ti = np.load('./output/ext1.npy')
    return Ti, Tizuobiao


def sectionloadG2(m, section, hz, hz2, xz, yz, xz2, yz2, jivalue):  # 相对坐标的剖面导入,比sectionloadG多了保留jivalue值的功能
    # 输入：
    # m--模型网格 section--剖面数组 hz, hz2--剖面顶底埋深 xz,yz--剖面两端点的相对坐标 jivalue--小个体
    # 斜剖面导入后需要扩充才正确 jivalue为基质值
    # 输出：
    # m--已经完成Ti导入的模拟网格

    ns = section.shape[1]
    hc = float(hz2 - hz) + 1
    xc = float(xz2 - xz)
    yc = float(yz2 - yz)
    if xc < 0:
        xc1 = xc - 1
    else:
        xc1 = xc + 1
    if yc < 0:
        yc1 = yc - 1
    else:
        yc1 = yc + 1
    # 计量后加一为长度
    lv = int(max(abs(xc1), abs(yc1)))  # 比较长度绝对值大小，得到即为斜剖面需要填网格总数，所以需要加一
    xlv = xc / (lv - 1)  # 将长度按lv均分，方便后续遍历导入
    ylv = yc / (lv - 1)
    x1 = xz
    y1 = yz
    # 对section的处理

    section = sectionex2(section, int(hc), lv, jivalue)  # 缩放剖面至固定坐标
    for n in range(lv):  # 按最长长度导入（以柱子为单位）
        m[hz:hz2 + 1, x1, y1] = section[:, n]
        x1 = int(xz + (n + 1) * xlv + 0.5)  # 四舍五入
        y1 = int(yz + (n + 1) * ylv + 0.5)

    # 剖面上方有空值时
    x1 = xz
    y1 = yz
    for n in range(lv):
        if section[0, n] == 0:
            m[0:hz + 1, x1, y1] = 0
        x1 = int(xz + (n + 1) * xlv + 0.5)  # 四舍五入
        y1 = int(yz + (n + 1) * ylv + 0.5)

    return m


def Extractcodelist(Ti, codelist):  # 提取TI地层层序
    # 输入：
    # Ti--剖面  codelist--空
    # 输出：
    # codelist--地层层序
    for x in range(Ti.shape[1]):
        Tempcodelist = []
        Tempcodelist.append(Ti[0, x])
        for h in range(1, Ti.shape[0]):
            if Ti[h, x] != Ti[h - 1, x]:
                if Ti[h, x] != -1:
                    Tempcodelist.append(Ti[h, x])
        if Tempcodelist not in codelist:
            codelist.append(Tempcodelist)
    return codelist


def sectionloadandextendG(m, template_x, template_y, flag, scale, jvalue):  # 导入和扩展全部剖面，flag==1为patchmatch步骤，0为initial步骤
    # 输出的坐标列表为RecodePatchmatch需要的格式
    # 插入地层层序判断的版本 加入了h方向的坐标
    # 输入：
    # m--初始网格 template_x, template_y--模板大小 flag--标志  scale--倍率  jvalue--小个体
    # 输出：
    # m--已经完成Ti导入并扩展的模拟网格  Tilist--Ti范围内地层值矩阵组成的列表  Tizuobiaolist--坐标矩阵列表   codelist--所有Ti的地层层序库
    Tilist = []
    Tizuobiaolist = []
    codelist = []  # 地层层序列表
    file1 = open('./Ti/Tiparameter.txt')
    content = file1.readline()
    string1 = [i for i in content if str.isdigit(i)]
    num = int(''.join(string1))
    print('剖面数目：')
    print(num)
    for n in range(num):  # 逐个导入Ti
        guding = []
        for j in range(6):
            content = file1.readline()
            string1 = [i for i in content if str.isdigit(i)]
            xx = int(''.join(string1))
            guding.append(xx)
        path = './Ti/' + str(n + 1) + '.bmp'
        section = cv2.imread(path, 0)  # 转剖面为数组

        codelist = Extractcodelist(section, codelist)  # 提取地层层序

        m = sectionloadG2(m, section, guding[0] * scale, guding[1] * scale, guding[2] * scale, guding[3] * scale,
                          guding[4] * scale, guding[5] * scale, jvalue)  # 载入单个剖面
        if flag == 1:
            Ti, Tizuobiao = RecodeTIextendforEMG(section, m, template_x, template_y, guding[0] * scale,
                                                 guding[1] * scale, guding[2] * scale, guding[3] * scale,
                                                 guding[4] * scale, guding[5] * scale)
            # 提取Ti矩阵和位置矩阵
            Tilist.append(Ti)
            Tizuobiaolist.append(Tizuobiao)

    return m, Tilist, Tizuobiaolist, codelist
