# coding=utf-8
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

def temextract(Ti, template_h, template_x, template_y, h0, x0, y0):  # 提取坐标h0,x0,y0处模式
    # 输入： Ti--Ti拓展后的模拟网格 template_h, template_x, template_y--模板大小
    # 输出： tem--模板
    ph = template_h // 2
    px = template_x // 2
    py = template_y // 2
    tem = Ti[h0 - ph:h0 + ph + 1, x0 - px:x0 + px + 1, y0 - py:y0 + py + 1]
    return tem

def Ou_distance(drill1, drill2):  # 模型欧式距离计算
    # 输入：drill1, drill2--两个三维模型
    # 输出：smstr--两个模型之间的欧式距离
    smstr = 0

    m=drill1-drill2
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            for k in range(m.shape[2]):
                smstr=smstr+m[i,j,k]*m[i,j,k]
    Ou= math.sqrt(smstr)
    return Ou


def density(template_h, template_x, template_y,h1, x1, y1,databaseVp,patchlist):
    m1=np.load('./Vp/studyVp.npy')
    dis=[]
    for i in range(len(patchlist)):
        o1=temextract(m1, template_h, template_x, template_y, h1, x1, y1)
        o2=databaseVp[patchlist[i]]
        dis.append(Ou_distance(o1,o2))
    return dis

def temdetect(tem):  # 检测模型是否包含待模拟点，若包含则返回值为False
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h, x, y] == -1:
                    return False
    return True

def Simplecluster(cdatabase, U, order):  # 简单聚类方法，单个方向
    # 输入：cdatabase--分类后的模式库 U--阈值 order--序号
    # 输出：Cdatabase--单个方向聚类后的模式序号库
    Cdatabase = []
    c = []
    for n in range(len(cdatabase)):
        if n not in c:  # 以没被聚类过的模式为起点
            d = []
            for m in range(n, len(cdatabase)):
                if  Ou_distance(cdatabase[n], cdatabase[m]) <= U:  # 计算两模式的距离，若距离小于分类阈值则聚为一类
                    d.append(m)
                    c.append(m)
            Cdatabase.append(d)
    np.save('./database/clusters' + str(order) + '.npy', Cdatabase)
    return Cdatabase


def databaseclusterAI(cdatabase, U):  # 不同方向模式库聚类
    # 输入：cdatabase--分类后的模式数据库  U--分类阈值
    # 输出：Cdatabase--聚类后的模式序号库
    print('start')

    p1 = multiprocessing.Process(target=Simplecluster, args=(cdatabase[0], U, 0))
    print('process start')
    p1.start()

    p2 = multiprocessing.Process(target=Simplecluster, args=(cdatabase[1], U, 1))
    print('process start')
    p2.start()

    p3 = multiprocessing.Process(target=Simplecluster, args=(cdatabase[2], U, 2))
    print('process start')
    p3.start()

    p4 = multiprocessing.Process(target=Simplecluster, args=(cdatabase[3], U, 3))
    print('process start')
    p4.start()

    p5 = multiprocessing.Process(target=Simplecluster, args=(cdatabase[4], U, 4))
    print('process start')
    p5.start()

    p6 = multiprocessing.Process(target=Simplecluster, args=(cdatabase[5], U, 5))
    print('process start')
    p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()

    time.sleep(5)
    print('process end')

    Cdatabase = []
    cc1 = np.load('./database/clusters0.npy')
    cc2 = np.load('./database/clusters1.npy')
    cc3 = np.load('./database/clusters2.npy')
    cc4 = np.load('./database/clusters3.npy')
    cc5 = np.load('./database/clusters4.npy')
    cc6 = np.load('./database/clusters5.npy')

    Cdatabase.append(cc1)
    Cdatabase.append(cc2)
    Cdatabase.append(cc3)
    Cdatabase.append(cc4)
    Cdatabase.append(cc5)
    Cdatabase.append(cc6)
    os.remove('./database/clusters0.npy')
    os.remove('./database/clusters1.npy')
    os.remove('./database/clusters2.npy')
    os.remove('./database/clusters3.npy')
    os.remove('./database/clusters4.npy')
    os.remove('./database/clusters5.npy')

    np.save('./database/Cdatabase.npy', Cdatabase)
    return Cdatabase


def databasecataAI(database, lag):  # 按照重叠区分类，六向面提取
    # 输入：database--模式库  lag--重叠区
    # 输出：cdatabase--分类后的模式数据库,大列表里有7个小列表，前6个分别储存不同方向的模式，最后一个为database
    template_h = database[0].shape[0]
    template_x = database[0].shape[1]
    template_y = database[0].shape[2]
    le = len(database)
    dis = []  # 后左上
    disx = []  # 前
    disy = []  # 右
    dish = []  # 下

    for n in range(lag):
        dis.append(n)
    for n in range(template_x - lag, template_x):
        disx.append(n)
    for n in range(template_y - lag, template_y):
        disy.append(n)
    for n in range(template_h - lag, template_h):
        dish.append(n)
    cdatabase = []
    # 下
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库0
        b = np.zeros((template_h, template_x, template_y))
        b[dish, :, :] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 左
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库1
        b = np.zeros((template_h, template_x, template_y))
        b[:, :, dis] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 右
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库2
        b = np.zeros((template_h, template_x, template_y))
        b[:, :, disy] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 后
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库3
        b = np.zeros((template_h, template_x, template_y))
        b[:, dis, :] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 前
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库4
        b = np.zeros((template_h, template_x, template_y))
        b[:, disx, :] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 上
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库5
        b = np.zeros((template_h, template_x, template_y))
        b[dis, :, :] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    cdatabase.append(database)  # 数据库6为本体
    print('done')
    return cdatabase


def databasebuildAI(Exm, template_h, template_x, template_y):  # 智能构建模式库
    # 输入：Exm--Ti拓展后的模拟网格  template_h,template_x,template_y--模板大小
    # 输出：database--模式库列表，每个元素为三维数组  zuobiaolist--模式坐标库列表
    lag = max(template_h, template_x, template_y)
    Exm2 = np.pad(Exm, lag, 'edge')  # 边缘拓展
    database = []
    zuobiaolist = []
    for h in range(Exm.shape[0]):
        for x in range(Exm.shape[1]):
            for y in range(Exm.shape[2]):
                if Exm[h, x, y] != -1:
                    h0 = h + lag
                    x0 = x + lag
                    y0 = y + lag
                    tem = temextract(Exm2, template_h, template_x, template_y, h0, x0, y0) # 提取模式
                    if temdetect(tem):  # 如果不包含待模拟点则为模式
                        database.append(tem)
                        zuobiaolist.append((h, x, y))
    return database, zuobiaolist


def initialAIforPythia(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U):
    # 全自动初始化流程整合
    # 输入：
    # m为已经导入了Ti的模拟网格 template_h, template_x, template_y--模板大小
    # lag, lag_h, lag_x, lag_y--重叠区大小 N--备选模式个数 U--分类阈值 valuels--Ti包含的值
    # 输出：
    # m--无空值的初始模型R0，保存为npy并输出vtk
    time_start1 = time.time()

    my_file = Path("./database/CdatabaseVp.npy")
    if my_file.exists():
        CdatabaseVp = np.load('./database/CdatabaseVp.npy')
        cdatabaseVp = np.load('./database/cdatabaseVp.npy')
        databaseVp = np.load('./database/databaseVp.npy')
        zuobiaolistVp = np.load('./database/zuobiaolistVp.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        databaseVp, zuobiaolistVp = databasebuildAI(m, template_h, template_x, template_y)  # 数据库构建
        np.save('./database/databaseVp.npy', databaseVp)
        np.save('./database/zuobiaolistVp.npy', zuobiaolistVp)
        cdatabaseVp = databasecataAI(databaseVp, lag) # 按方位分类数据库
        np.save('./database/cdatabaseVp.npy', cdatabaseVp)
        CdatabaseVp = databaseclusterAI(cdatabaseVp, U) # 聚类
        np.save('./database/CdatabaseVp.npy', CdatabaseVp)
        print('Patterndatabase has been builded!')
    time_end1 = time.time()
    print('数据库构建时间损耗:')
    print(time_end1 - time_start1)

    time_start = time.time()


file1 = open('./parameter.txt')
content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
Mh = int(''.join(string1))
# print Mh
# 初始模型高

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
Mx = int(''.join(string1))
# print Mx
# 初始模型长

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
My = int(''.join(string1))
# print My
# 初始模型宽

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
lag = int(''.join(string1))
# print lag
# 重叠区/扩展区

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
lag_h = int(''.join(string1))
# print lag_h
# 移动间隔

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
lag_x = int(''.join(string1))
# print lag_x
# 移动间隔


content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
lag_y = int(''.join(string1))
# print lag_y
# 移动间隔

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
patternSizeh = int(''.join(string1))
# print patternSizeh
# 模板大小

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
# 分类阈值

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
N = int(''.join(string1))
# print N
# 备选模式个数

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
size = int(''.join(string1))
# print size
# 迭代模板大小

content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
itr = int(''.join(string1))
# print itr
# 单个尺度迭代次数

content = file1.readline()
scale = []
for i in content:
    if str.isdigit(i):
        scale.append(int(i))
# print scale
# 尺度倍率

# 一次模拟的个数
content = file1.readline()
string1 = [i for i in content if str.isdigit(i)]
Modelcount = int(''.join(string1))
# print Modelcount
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

U=20000
N=5
m=np.load('./output/initial_Vp.npy')
initialAIforPythia(m, patternSizeh, patternSizex, patternSizey, lag, lag_h, lag_x, lag_y, N, U)


