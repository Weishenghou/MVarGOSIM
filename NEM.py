# coding=utf-8
# EM多尺度优化迭代
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pylab
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

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
from probcombine_entropy1 import *
from TI import *


def cal_distance(a, b, A_padding, B, p_size):  # 计算两个模型间的某两个模式的汉明距离
    # 输入：a、b--分别为模型A、B的某个模式位置  A_padding, B--A、B模型  p_size--模板大小
    # 输出：dist--两个模式间的汉明距离

    p = p_size // 2
    patch_a = A_padding[a[0] - p:a[0] + p + 1, a[1] - p:a[1] + p + 1, a[2] - p:a[2] + p + 1]

    patch_b = B[b[0] - p:b[0] + p + 1, b[1] - p:b[1] + p + 1, b[2] - p:b[2] + p + 1]

    temp = patch_b - patch_a
    smstr = np.nonzero(temp)

    dist = np.shape(smstr[0])[0]

    return dist


def Recoderandom_searchsub(f, a, dist, A_padding, B, p_size, lag, alpha=0.5):  # 随机搜索，以传播后匹配到的模式坐标为中心来确定搜索范围
    # 输入：
    # f--初始化时的匹配位置  a--待搜索点 dist--距离矩阵  A_padding--边缘扩展的模型 B--边缘扩展的Ti矩阵
    # 输出：
    # f--完成随机搜索的匹配模式位置矩阵 dist--距离矩阵
    p = p_size // 2
    x = int((a[0] - p) / (lag + 1))
    y = int((a[1] - p) / (lag + 1))
    z = int((a[2] - p) / (lag + 1))

    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)

    i = 2
    search_h = B_h * alpha ** i  # 搜索范围
    search_l = B_l * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y, z][0]  # 传播后的匹配坐标
    b_y = f[x, y, z][1]
    b_z = f[x, y, z][2]
    while search_h > 1 and search_l > 1 and search_w > 1:  # 搜索范围小于1时停止

        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h - p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        if (B_l > (3 * p_size)) and (B_w > (3 * p_size)):
            print('and')
            print(B_l, B_h)
            search_min_c = max(b_y - search_l, p)
            search_max_c = min(b_y + search_l, B_l - p)

            random_b_y = np.random.randint(search_min_c, search_max_c)
            search_min_v = max(b_z - search_w, p)
            search_max_v = min(b_z + search_w, B_w - p)

            random_b_z = np.random.randint(search_min_v, search_max_v)
            search_l = B_l * alpha ** i
            search_w = B_w * alpha ** i

        else:
            if B_l >= B_w:  # Ti矩阵宽度较小时
                search_min_c = max(b_y - search_l, p)
                search_max_c = min(b_y + search_l, B_l - p)

                random_b_y = np.random.randint(search_min_c, search_max_c)
                random_b_z = np.random.randint(p, B_w - 1 - p)
                search_l = B_l * alpha ** i
                search_w = B_w
            else:  # Ti矩阵长度较小时
                random_b_y = np.random.randint(p, B_l - 1 - p)
                search_min_v = max(b_z - search_w, p)
                search_max_v = min(b_z + search_w, B_w - p)

                random_b_z = np.random.randint(search_min_v, search_max_v)
                search_l = B_l
                search_w = B_w * alpha ** i

        search_h = B_h * alpha ** i

        b = np.array([random_b_x, random_b_y, random_b_z])
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < dist[x, y, z]:  # 替换更适合的最近邻
            dist[x, y, z] = d
            f[x, y, z] = b
        i += 1

    return f, dist


def Recodepropagationsub(f, a, dist, A_padding, B, p_size, is_odd, lag):  # 传播,a为待传播点的坐标
    # 输入：
    # f--初始化时的匹配位置  a--待传播点 dist--距离矩阵  A_padding--边缘扩展的模型 B--边缘扩展的Ti矩阵
    # p_size--模板大小 is_odd--标志 lag--重叠区
    # 输出：传播后的位置矩阵和距离矩阵
    A_h = f.shape[0]
    A_l = f.shape[1]
    A_w = f.shape[2]
    p = p_size // 2
    x = int((a[0] - p) / (lag + 1))  # 调整坐标为序号
    y = int((a[1] - p) / (lag + 1))
    z = int((a[2] - p) / (lag + 1))

    if is_odd:
        d_left = dist[max(x - 1, 0), y, z]
        d_up = dist[x, max(y - 1, 0), z]
        d_forward = dist[x, y, max(z - 1, 0)]
        d_current = dist[x, y, z]
        idx = np.argmin(np.array([d_current, d_left, d_up, d_forward]))  # 可以多加几个传播值
        if idx == 1:
            f[x, y, z] = f[max(x - 1, 0), y, z]
            dist[x, y, z] = cal_distance(a, f[x, y, z], A_padding, B, p_size)
        if idx == 2:
            f[x, y, z] = f[x, max(y - 1, 0), z]
            dist[x, y, z] = cal_distance(a, f[x, y, z], A_padding, B, p_size)
        if idx == 3:
            f[x, y, z] = f[x, y, max(z - 1, 0)]
            dist[x, y, z] = cal_distance(a, f[x, y, z], A_padding, B, p_size)
    else:
        d_right = dist[min(x + 1, A_h - 1), y, z]
        d_down = dist[x, min(y + 1, A_l - 1), z]
        d_back = dist[x, y, min(z + 1, A_w - 1)]
        d_current = dist[x, y, z]
        idx = np.argmin(np.array([d_current, d_right, d_down, d_back]))
        if idx == 1:
            f[x, y, z] = f[min(x + 1, A_h - 1), y, z]
            dist[x, y, z] = cal_distance(a, f[x, y, z], A_padding, B, p_size)
        if idx == 2:
            f[x, y, z] = f[x, min(y + 1, A_l - 1), z]
            dist[x, y, z] = cal_distance(a, f[x, y, z], A_padding, B, p_size)
        if idx == 3:
            f[x, y, z] = f[x, y, min(z + 1, A_w - 1)]
            dist[x, y, z] = cal_distance(a, f[x, y, z], A_padding, B, p_size)
    return f, dist


def RecodereNNSBsub(f, dist, items, name, core, p_size, lag):  # 重组，导入完成随机搜索后的匹配模式位置矩阵和距离矩阵
    # 输入：
    # f--初始化匹配到的模式位置  dist--距离矩阵  items--需要搜索模式的模型位置列表（以步长为单位）
    # name--Ti序号  p_size--模板大小 lag--坐标与实际坐标间隔
    # 输出：f--更新后的位置矩阵 dist--距离矩阵
    p = p_size // 2
    for n in range(len(items)):
        path1 = './database/patchmatchprocess(' + str(name) + ')(' + str(n) + ')f.npy'
        path2 = './database/patchmatchprocess(' + str(name) + ')(' + str(n) + ')dist.npy'
        fff = np.load(path1)
        distdistdist = np.load(path2)
        for i in range(len(items[n])):  # 重新导入
            f[(items[n][i][0] - p) // (lag + 1), (items[n][i][1] - p) // (lag + 1), (items[n][i][2] - p) // (lag + 1)] = \
            fff[(items[n][i][0] - p) // (lag + 1), (items[n][i][1] - p) // (lag + 1), (items[n][i][2] - p) // (lag + 1)]
            dist[(items[n][i][0] - p) // (lag + 1), (items[n][i][1] - p) // (lag + 1), (items[n][i][2] - p) // (
                        lag + 1)] = distdistdist[
                (items[n][i][0] - p) // (lag + 1), (items[n][i][1] - p) // (lag + 1), (items[n][i][2] - p) // (lag + 1)]
    return f, dist


def RecodeNNSBsub(f, dist, m_padding, Bref, p_size, item, name, core, lag):  # 开始传播和搜索近似最近邻
    # 输入：
    # f--初始化匹配到的模式位置  dist--距离  m_padding--边缘扩展半个模板的模型  Bref--边缘扩展半个模板的TI矩阵
    # p_size--模板大小 item--需要搜索模式的模型位置列表（以步长为单位）  name--Ti序号  core--第几组列表  lag--重叠区
    # 输出；
    # 返回并储存匹配到模式的位置列表和距离列表
    for n in range(len(item)):  # 遍历需要匹配模式的位置
        f, dist = Recodepropagationsub(f, item[n], dist, m_padding, Bref, p_size, bool(random.getrandbits(1)),
                                       lag)  # 传播
        f, dist = Recoderandom_searchsub(f, item[n], dist, m_padding, Bref, p_size, lag)  # 模式搜索
    path1 = './database/patchmatchprocess(' + str(name) + ')(' + str(core) + ')f.npy'
    path2 = './database/patchmatchprocess(' + str(name) + ')(' + str(core) + ')dist.npy'
    np.save(path1, f)
    np.save(path2, dist)


def apartlist(ls, size):  # 分割列表工具
    return [ls[i:i + size] for i in range(0, len(ls), size)]


def Recodeinitialization(A, B, Bzuobiao, p_size, lag):  # 初始化，构建映射，lag为超出计算规格时所用的空洞步长
    # 输入：
    # A--初始模型 B--Ti范围内地层值矩阵 Bzuobiao--坐标矩阵
    # 输出：
    # f--A匹配到B的随机位置坐标矩阵  dist--AB模式汉明距离 A_padding--边缘扩展的模型  B--边缘扩展的Ti矩阵
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)

    p = p_size // 2
    B = np.pad(B, p, 'edge')  # B往两边扩展模板的一半
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    random_B_r = np.random.randint(p, B_h - p, [A_h, A_l, A_w])  # 分别储存对应B的坐标矩阵

    random_B_c = np.random.randint(p, B_l - p, [A_h, A_l, A_w])
    random_B_v = np.random.randint(p, B_w - p, [A_h, A_l, A_w])

    A_padding = -np.ones([A_h + p * 2, A_l + p * 2, A_w + p * 2])
    A_padding[p:A_h + p, p:A_l + p, p:A_w + p] = A  # 扩展初始模型
    f = np.zeros([A_h // (lag + 1), A_l // (lag + 1), A_w // (lag + 1)], dtype=object)  # 按lag步长划分的矩阵
    dist = np.zeros([A_h // (lag + 1), A_l // (lag + 1), A_w // (lag + 1)])
    ###用两个矩阵分别储存随机匹配到模式的位置和距离###
    for i in range(A_h // (lag + 1)):
        for j in range(A_l // (lag + 1)):
            for k in range(A_w // (lag + 1)):
                a = np.array([i + p, j + p, k + p])
                b = np.array([random_B_r[i, j, k], random_B_c[i, j, k], random_B_v[i, j, k]], dtype=np.int32)

                f[i, j, k] = b
                dist[i, j, k] = cal_distance(a, b, A_padding, B, p_size)

    ###硬数据位置距离为0###
    number = 0
    for i in range(B_h - 2 * p):
        for j in range(B_l - 2 * p):
            for k in range(B_w - 2 * p):
                first = Bzuobiao[number] // (lag + 1)
                f[first[0], first[1], first[2]] = np.array([i + p, j + p, k + p])
                number = number + 1
                dist[first[0], first[1], first[2]] = 0

    return f, dist, A_padding, B


def RecodereconstructionTTT(A, mm, F, Fore, BTilist, p_size, lag):  # 更新模型
    # 输入：
    # A--初始模型  mm--对应尺度的硬数据模型  F--位置矩阵集合  Fore--最近邻所属TI的序号组成的矩阵
    # BTilist--Ti矩阵组成的列表  p_size--模板大小  lag--重叠区
    # 输出：Re--重构模型 CTilist--统计模式来源列表
    p = p_size // 2
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = -np.ones([A_h + p * 2, A_l + p * 2, A_w + p * 2])  # 构建临时空网格储存模型
    CTilist = []  # 构建模板使用统计TI
    for n in range(len(BTilist)):
        ti = np.zeros_like(BTilist[n])
        CTilist.append(ti)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                for n in range(len(F)):
                    if mm[i, j, k] == -1:
                        if Fore[i // (lag + 1), j // (lag + 1), k // (lag + 1)] == n:
                            Retem = temextract(BTilist[n], 2 * lag + 1, 2 * lag + 1, 2 * lag + 1,
                                               F[n][i // (lag + 1), j // (lag + 1), k // (lag + 1)][0],
                                               F[n][i // (lag + 1), j // (lag + 1), k // (lag + 1)][1],
                                               F[n][i // (lag + 1), j // (lag + 1), k // (lag + 1)][2])
                            temp = temextractRAI(temp, Retem, i + p, j + p, k + p)  # 更新搜索到的最近邻
                            CTilist[n][F[n][i // (lag + 1), j // (lag + 1), k // (lag + 1)][0],
                                       F[n][i // (lag + 1), j // (lag + 1), k // (lag + 1)][1],
                                       F[n][i // (lag + 1), j // (lag + 1), k // (lag + 1)][2]] = CTilist[n][F[n][i // (
                                        lag + 1), j // (lag + 1), k // (lag + 1)][0], F[n][i // (lag + 1), j // (
                                        lag + 1), k // (lag + 1)][1], F[n][i // (lag + 1), j // (lag + 1), k // (
                                        lag + 1)][2]] + 1
                    else:
                        temp[i + p, j + p, k + p] = mm[i, j, k]  # 硬数据重新导入
    Re = temp[p:A_h + p, p:A_l + p, p:A_w + p]
    return Re, CTilist


def RecodeprojectTTT(m, F, DIST, lag):  # 选择最近邻
    # 输入：m--初始模型 F--匹配模式的位置矩阵 DIST--距离矩阵 lag--重叠区
    # 输出：Fore--最近邻所属TI的序号组成的矩阵
    Fore = np.zeros((m.shape[0] // (lag + 1), m.shape[1] // (lag + 1), m.shape[2] // (lag + 1)), int)
    sw = 999999
    for i in range(Fore.shape[0]):
        for j in range(Fore.shape[1]):
            for k in range(Fore.shape[2]):
                for n in range(len(F)):
                    if DIST[n][i, j, k] <= sw:
                        Thor = n
                        sw = DIST[n][i, j, k]

                Fore[i, j, k] = Thor
                sw = 999999
    return Fore


def RecodeNNSB(m, ref, refzuobiao, p_size, itr, name, core, lag):
    # 寻找最近临并行版改进 name为ti编号，core为并行模拟核数,lag为坐标与实际坐标间隔
    # 输入：
    # m--初始模型 ref--单个Ti范围内地层值矩阵  refzuobiao--坐标矩阵
    # p_size--模板大小 itr--迭代次数 name--Ti序号
    # 输出:
    # f--m模型每个位置匹配到的模式相应坐标组成的矩阵 dist--距离矩阵 Bref--扩展后的ref
    A_h = np.size(m, 0)
    A_l = np.size(m, 1)
    A_w = np.size(m, 2)
    f, dist, m_padding, Bref = Recodeinitialization(m, ref, refzuobiao, p_size, lag)
    p = p_size // 2
    print("initialization done")
    zuobiaoarr = []
    for h in range(0, A_h, lag + 1):
        for x in range(0, A_l, lag + 1):
            for y in range(0, A_w, lag + 1):
                zuobiaoarr.append(np.array([h + p, x + p, y + p]))  # 储存每个模板的位置
    items = apartlist(zuobiaoarr, int(len(zuobiaoarr) / core))  # 按并行核数分割列表
    # 内嵌式多进程
    for itr in range(1, itr + 1):
        if itr % 2 == 0:
            processes = list()
            print(len(items))
            for n in range(len(items)):
                s = multiprocessing.Process(target=RecodeNNSBsub,
                                            args=(f, dist, m_padding, Bref, p_size, items[n], name, n, lag))
                print('process:', n)
                s.start()
                processes.append(s)
            for s in processes:
                s.join()
            f, dist = RecodereNNSBsub(f, dist, items, name, core, p_size, lag)
        else:
            processes = list()
            for n in range(len(items)):
                newList = list(reversed(items[n]))  # 倒转列表顺序
                s = multiprocessing.Process(target=RecodeNNSBsub,
                                            args=(f, dist, m_padding, Bref, p_size, newList, name, n, lag))
                print('process:', n)
                s.start()
                processes.append(s)
            for s in processes:
                s.join()
            f, dist = RecodereNNSBsub(f, dist, items, name, core, p_size, lag)
        print("iteration: %d" % (itr))
    return f, dist, Bref


def Recodepatchmatch(m, mm, Tilist, Tizuobiaolist, size, itr, core, lag):  # patchmatch重新优化版
    # 增加新加速模式，首先提取列表，按照一定重叠区选择待模拟点，重构过程根据多源融合结果整块复制
    # 输入：
    # m--完成了序贯模拟无空值的模型 mm--对应尺度的导入Ti模型 Tilist--Ti范围内地层值矩阵 Tizuobiaolist--坐标矩阵
    # size--迭代模板大小 itr--迭代次数 core--并行进程数  lag--坐标与实际坐标间隔，默认为0
    # 输出：
    # Re--完成迭代优化的模型 CTilist--统计模式来源列表

    Fore = np.zeros([m.shape[0], m.shape[1], m.shape[2]])  # 计算模型体积
    print('本轮迭代步骤开始')
    start = time.time()  # 计时开始
    F = []
    DIST = []
    BTilist = []
    processes = list()

    for n in range(len(Tilist)):  # 遍历每个TI矩阵
        print(Tilist[n].shape)
        f, dist, Bref = RecodeNNSB(m, Tilist[n], Tizuobiaolist[n], size, itr, n, core, lag)  # 搜索步骤
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)
    print("Searching done!")
    Fore = RecodeprojectTTT(m, F, DIST, lag)
    print("Pick done!")
    Re, CTilist = RecodereconstructionTTT(m, mm, F, Fore, BTilist, size, lag)
    print('更新步骤完成')
    end = time.time()
    print(end - start)  # 计时结束
    return Re, CTilist
