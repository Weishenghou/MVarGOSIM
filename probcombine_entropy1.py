# coding=utf-8

########### 初始模型构建 ###########


import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pylab
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot
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

import heapq
from TI import *
from Cross_entropyweight import *
from jixiaoxiufu4 import *


def cut(m, lag):  # 裁剪模型
    return m[lag:m.shape[0] - lag, lag:m.shape[1] - lag, lag:m.shape[2] - lag]


def temcheck(tem):  # 检测该节点模版大小内是否有待模拟点
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h, x, y] == -1:
                    return True
    return False


def temextract(Ti, template_h, template_x, template_y, h0, x0, y0):  # 提取坐标h0,x0,y0处模式
    # 输入： Ti--Ti拓展后的模拟网格 template_h, template_x, template_y--模板大小
    # 输出： tem--模板
    ph = template_h // 2
    px = template_x // 2
    py = template_y // 2
    tem = Ti[h0 - ph:h0 + ph + 1, x0 - px:x0 + px + 1, y0 - py:y0 + py + 1]
    return tem


def temdetect(tem):  # 检测模型是否包含待模拟点，若包含则返回值为False
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h, x, y] == -1:
                    return False
    return True


class earlystopcc(keras.callbacks.Callback):  # 满足训练要求，提前停止训练的提示
    def __init__(self, monitor='loss', value=0.00001, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warning.warn("Early stop requires %s availabel!" % self.monitor, RuntimeWarning)
        if abs(current) < abs(self.value):
            print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def temextractRAI(Ti, tem, h0, x0, y0):  # 拼接模式
    # 输入：
    # Ti--模型 tem--匹配到的模式 h0, x0, y0--对应坐标点
    # 输出：
    # Ti--拼贴好模式后的模型
    template_h = tem.shape[0]
    template_x = tem.shape[1]
    template_y = tem.shape[2]
    nn1 = 0
    nn2 = 0
    nn3 = 0
    hh = int((template_h - 1) / 2)
    xx = int((template_x - 1) / 2)
    yy = int((template_y - 1) / 2)
    for n1 in range(h0 - hh, h0 + hh + 1):
        for n2 in range(x0 - xx, x0 + xx + 1):
            for n3 in range(y0 - yy, y0 + yy + 1):
                if Ti[n1, n2, n3] == -1:
                    Ti[n1, n2, n3] = tem[nn1, nn2, nn3]

                nn3 = nn3 + 1

            nn2 = nn2 + 1
            nn3 = 0
        nn1 = nn1 + 1
        nn2 = 0

    return Ti  # 提取坐标x0,y0处模板


def getListMinNumIndex(num_list, topk=3):  # 获取列表中最小的前n个数值的位置索引
    min_number = heapq.nsmallest(topk, num_list)
    min_index = []
    for t in min_number:
        index = num_list.index(t)
        min_index.append(index)
        num_list[index] = 0

    return min_index



def hamming_distance(drill1, drill2):  # 计算汉明距离
    # 输入：drill1, drill2--两个一维数组
    # 输出：np.shape(smstr[0])[0]--汉明距离
    vector1 = np.mat(drill1)
    vector2 = np.mat(drill2)

    smstr = np.nonzero(vector1 - vector2)
    return np.shape(smstr[0])[0]


def hamming_distance2(drill1, drill2):  # 模型汉明距离计算
    # 输入：drill1, drill2--两个三维模型
    # 输出：smstr--两个模型之间的汉明距离
    smstr = 0
    for n in range(drill1.shape[0]):
        vector1 = np.mat(drill1[n, :, :])
        vector2 = np.mat(drill2[n, :, :])

        vector3 = vector1 - vector2

        smstr = smstr + np.shape(np.nonzero(vector1 - vector2)[0])[0]
    return smstr


def Ou_distance(m1, m2):  # 模型欧式距离计算
    # 输入：drill1, drill2--两个三维模型
    # 输出：smstr--两个模型之间的欧式距离
    smstr = 0

    m = m1 - m2
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            for k in range(m.shape[2]):
                smstr = smstr + m[i, j, k] * m[i, j, k]
    Ou = math.sqrt(smstr)
    return Ou


def Ou_distance1(drill1, drill2):  # 一维模型欧式距离计算
    # 输入：drill1, drill2--两个三维模型
    # 输出：smstr--两个模型之间的欧式距离
    smstr = 0

    m = drill1 - drill2
    for i in range(m.shape[0]):
        smstr = smstr + m[i] * m[i]
    Ou = math.sqrt(smstr)
    return Ou


def temextractR(Ti, tem, h0, x0, y0):  # 模板返还值
    ph = tem.shape[0] // 2
    px = tem.shape[1] // 2
    py = tem.shape[2] // 2
    Ti[h0 - ph:h0 + ph + 1, x0 - px:x0 + px + 1, y0 - py:y0 + py + 1] = tem
    return Ti


def lujinglistAI(m, template_h, template_x, template_y, lag):  # 将所有待模拟网格中空值（-1）的待模拟点加入模拟路径,lag为重叠区大小
    roadlist = []

    for h in range(0, m.shape[0], lag):
        for x in range(0, m.shape[1], lag):
            for y in range(0, m.shape[2], lag):
                if temcheck(m[h - lag:h + lag + 1, x - lag:x + lag + 1, y - lag:y + lag + 1]):
                    roadlist.append((h, x, y))
    return roadlist


def faultmodelbuild(xy_data, xy_value, inputdata, epoch):  # 训练从Ti中提取的埋深数据得出模型并将inputdata数据作为输出得出地层整体顶底埋深
    model = Sequential()
    rate = 0
    model.add(Dense(50, input_dim=2, activation='relu'))  # 隐藏层
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(rate))

    model.add(Dense(300, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate))

    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adamax')

    model.summary()
    Mycallback = [
        earlystopcc(monitor='loss', value=0.00005, verbose=0)  # 0.00005
        # keras.callbacks.EarlyStopping(monitor='loss',patience=100,verbose=0,mode='auto',min_delta=100)
    ]  # 提前停止训练提示

    history = model.fit(xy_data, xy_value, epochs=epoch, verbose=1, callbacks=Mycallback)
    output = model.predict(inputdata)  # 预测硬数据外的地层埋深
    return output


def imgloaderforkeras2(depthimg, flag0):  # 将俯视图调整为可输入神经网络的训练数据
    # 输入：depthimg--埋深二维俯视图 flag0--标志
    # 输出：xy_data--（x，y）坐标  xy_value--（x，y）坐标处value值的埋深  inputdata--没有值的（x，y）坐标
    L = depthimg.shape[0]
    W = depthimg.shape[1]
    xy_data = []
    xlist = []
    ylist = []
    xy_value = []
    z_data = []
    inputdata = []
    count = 0
    xcount = 0
    for x in range(L):
        for y in range(W):
            if depthimg[x, y] != -1:  # 当非空值时
                xlist.append(float(x))
                ylist.append(float(y))

    if flag0 == 0:
        x1 = int(min(xlist))
        x2 = int(max(xlist))
        y1 = int(min(ylist))
        y2 = int(max(ylist))
    else:
        x1 = 0
        x2 = L - 1
        y1 = 0
        y2 = W - 1

    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            if depthimg[x, y] != -1:  # 当非空值时
                xy_data.append([float(x) / float(L), float(y) / float(W)])

                xy_value.append(float(depthimg[x, y]))
                count = count + 1
            else:
                inputdata.append([float(x) / float(L), float(y) / float(W)])
                xcount = xcount + 1
    xy_data = np.array(xy_data).astype(np.float32)
    xy_data = xy_data.reshape(count, 2)
    inputdata = np.array(inputdata).astype(np.float32)
    inputdata = inputdata.reshape(xcount, 2)
    xy_value = np.array(xy_value).astype(np.float32)
    xy_value = xy_value.reshape(count, 1)

    return xy_data, xy_value, inputdata


def twodfaultsurface(m, value):  # fault trans to 二维俯视面
    # 输入：value--模拟值
    # 输出：top、bottom--顶底面埋深二维俯视图
    L = m.shape[1]
    W = m.shape[2]
    mm = np.pad(m, ((1, 1), (0, 0), (0, 0)), 'constant', constant_values=-1)  # 为了让高程值对应上索引和最低层也能判断
    top = -np.ones((L, W), int)
    bottom = -np.ones((L, W), int)
    for x in range(L):
        for y in range(W):
            count = 0
            for h in range(1, m.shape[0] + 1):
                if mm[h, x, y] == value:
                    if mm[h - 1, x, y] != value and count == 0:
                        top[x, y] = h
                    if mm[h + 1, x, y] != value:
                        bottom[x, y] = h
                        count = 1  # 标志遍历完了（x，y）处的value地层，找出了顶底埋深
    return top, bottom


def cluster_distance(vecA, vecB):  # 调整向量/模式的形状来计算汉明距离
    # 输入：vecA, vecB--需调整成一维的三维向量/模式
    # 输出：d--两个模式间的汉明距离
    ss = vecA.shape[0] * vecA.shape[1] * vecA.shape[2]  # 计算网格总数
    drill1 = vecA.reshape(ss, 1)  # 调整为一维
    drill2 = vecB.reshape(ss, 1)
    d = hamming_distance(drill1, drill2)  # 计算汉明距离
    return d


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str


def subroadlistinitialfornew(m2, lujing, template_h, template_x, template_y, lag):  # 新一次序贯模拟的随机路径设置
    # 输入：m2--经过序贯模拟后因层序有错误需重新模拟的模型 lujing--需重新模拟的路径
    # 输出：Roadlist--最终随机路径

    Roadlist = []  # 最终路径名单

    random.shuffle(lujing)

    m3 = m2.copy()

    DevilTrigger = False

    Banlujing = []
    b = np.zeros((template_h, template_x, template_y), int)
    n = 0

    while n < len(lujing):
        if m2[lujing[n][0], lujing[n][1], lujing[n][2]] == -1:
            h1 = lujing[n][0]
            x1 = lujing[n][1]
            y1 = lujing[n][2]

            o1 = temextract(m3, template_h, template_x, template_y, h1, x1, y1)
            k = 0  # 重叠区计数器

            if temdetectD1(o1[template_h - lag:template_h, :, :]):
                # 下
                k = k + 1
            if temdetectD1(o1[:, 0:lag, :]):
                # 后
                k = k + 1
            if temdetectD1(o1[:, template_x - lag:template_x, :]):
                # 前
                k = k + 1
            if temdetectD1(o1[:, :, 0:lag]):
                # 左
                k = k + 1
            if temdetectD1(o1[:, :, template_y - lag:template_y]):
                # 右
                k = k + 1
            if (h1 > template_h - lag) and (k >= 2):
                m2 = temextractRAI(m3, b, h1, x1, y1)
                Roadlist.append((h1 - lag, x1 - lag, y1 - lag))
            elif (h1 <= template_h - lag) and (k != 0):
                m2 = temextractRAI(m3, b, h1, x1, y1)
                Roadlist.append((h1 - lag, x1 - lag, y1 - lag))
            else:
                lujing.append(lujing[n])
        n = n + 1
    return Roadlist


def checkunreal2(m, lag):  # 检测待模拟点然后返回待模拟点列表
    # 输入：m--初步完成序贯模拟的模型 lag--重叠区
    # 输出：m--将模拟到强约束的值恢复待模拟 d--仍需要模拟的坐标列表/路径
    d = []
    for h in range(lag, m.shape[0] - lag):
        for x in range(lag, m.shape[1] - lag):
            for y in range(lag, m.shape[2] - lag):
                if m[h, x, y] == -2:
                    m[h, x, y] = -1
                if m[h, x, y] == -1:
                    d.append((h, x, y))

    return m, d


def replacepi(re, hardlist):  # re为带替换的值，hardlist为不替换的坐标
    for n in range(re.shape[0]):
        if re[n] not in hardlist:
            re[n] = -1
    return re


def codecheckZ(list1, codelist):  # 地层检测机制，如果存在错误则返回真
    # 输入：list1--待检查地层层序柱  codelist--地层层序库
    # 输出：若list1不属于codelist或其子集则层序错误并返回真
    sub = []  # 子集
    a = len(list1)
    if a == 1:
        return True
    for neck in range(len(codelist)):
        sub1 = (sorted(set(it.combinations(codelist[neck], r=a))))
        sub2 = [list(m) for m in sub1]
        sub = sub + sub2

    if list1 not in sub:
        return True
    return False


def Extractcodelist2(drill):  # 提取虚拟钻孔地层层序

    codelistfromdrill = []

    codelistfromdrill.append(drill[0])

    for h in range(drill.shape[0]):
        if drill[h] != drill[h - 1]:
            codelistfromdrill.append(drill[h])
    for cik in range(len(codelistfromdrill)):
        if -1 in codelistfromdrill:
            codelistfromdrill.remove(-1)
        if -2 in codelistfromdrill:

            codelistfromdrill.remove(-2)
        else:
            break
    codelistfromdrill = [k for k, g in it.groupby(codelistfromdrill)]
    # 去掉-1，去掉相邻重复

    return codelistfromdrill


def TemplateHard(m, tem, h0, x0, y0, hardvaluelist):  # 填充匹配到的模式
    # 输入：m--顶底面拟合后空值待模拟的模型 tem--匹配搭配的模式  h0, x0, y0--坐标 hardvaluelist为硬约束的数据
    # 输出：m--填充好模式的模型

    template_h = tem.shape[0]
    template_x = tem.shape[1]
    template_y = tem.shape[2]
    nn1 = 0
    nn2 = 0
    nn3 = 0
    hh = int((template_h) // 2)
    xx = int((template_x) // 2)
    yy = int((template_y) // 2)
    for n1 in range(h0 - hh, h0 + hh + 1):
        for n2 in range(x0 - xx, x0 + xx + 1):
            for n3 in range(y0 - yy, y0 + yy + 1):
                if m[n1, n2, n3] == -1:
                    m[n1, n2, n3] = tem[nn1, nn2, nn3]
                    for ro in range(len(hardvaluelist)):
                        if (tem[nn1, nn2, nn3] == hardvaluelist[ro]):  # 模拟到强约束返回空值
                            m[n1, n2, n3] = -2
                            break

                nn3 = nn3 + 1
            nn2 = nn2 + 1
            nn3 = 0
        nn1 = nn1 + 1
        nn2 = 0
    return m


def hanmingdis_prob(temo, c, database, canpatternlist,si): #汉明距离转概率
    ss = temo.shape[0] * temo.shape[1] * temo.shape[2]
    drill1 = temo.reshape(ss, 1)
    d=[]
    for i in range(len(si)):
        ctem = database[canpatternlist[si[i]]] * c
        drill2 = ctem.reshape(ss, 1)
        d.append(hamming_distance(drill1, drill2))
    d_sum = sum(d)
    d = np.array(d).astype(float)
    s=np.where(d==0)[0]
    d[s]=0.001
    d = np.reciprocal(d) * d_sum
    d_sum2 = d.sum()
    prob_s = d / d_sum2
    return prob_s

def Oudis_prob(temo_d, c, database, canpatternlist,si):   #欧式距离转概率
    ss = temo_d.shape[0] * temo_d.shape[1] * temo_d.shape[2]
    drill1 = temo_d.reshape(ss, 1)
    d=[]
    for i in range(len(si)):
        ctem = database[canpatternlist[si[i]]] * c
        drill2 = ctem.reshape(ss, 1)
        d.append(Ou_distance1(drill1, drill2))
    d_sum = sum(d)
    d = np.array(d).astype(float)
    s=np.where(d==0)[0]
    d[s]=0.001
    d = np.reciprocal(d) * d_sum
    d_sum2 = d.sum()
    prob = d / d_sum2
    return prob

def p_distance(zuobiao_point,zuobiaolist,canpatternlist,si):
    d = []
    for i in range(len(si)):
      distance=math.sqrt((zuobiaolist[canpatternlist[si[i]]][0]-zuobiao_point[0])**2+(zuobiaolist[canpatternlist[si[i]]][1]-zuobiao_point[1])**2+(zuobiaolist[canpatternlist[si[i]]][2]-zuobiao_point[2])**2)
      d.append(distance)

    d_sum = sum(d)
    d = np.array(d).astype(float)
    d[d==0]=0.0001
    d = np.reciprocal(d) * d_sum
    d_sum2 = d.sum()
    prob = d / d_sum2
    return prob

def patternsearchAI2(o1,temo_d,temo_v,c, database,cdatabase_d,cdatabase_v, canpatternlist, N,zuobiao_point,zuobiaolist):  # 根据重叠区,在候选列表中选取备选模式 #直接返回候选模板
    # N为备选模板个数
    ss = o1.shape[0] * o1.shape[1] * o1.shape[2]
    template_h = database[0].shape[0]
    template_x = database[0].shape[1]
    template_y = database[0].shape[2]
    d = []  # 备选列表
    drill1 = o1.reshape(ss, 1)
    for n in range(len(canpatternlist)):
        ctem = database[canpatternlist[n]] * c
        drill2 = ctem.reshape(ss, 1)
        d.append(hamming_distance(drill1, drill2))
    si = getListMinNumIndex(d, N)
    prob_s=hanmingdis_prob(o1, c, database, canpatternlist,si)
    prob_d=Oudis_prob(temo_d, c, cdatabase_d, canpatternlist,si)
    prob_v=Oudis_prob(temo_v, c, cdatabase_v, canpatternlist,si)
    data=np.zeros((N,3))
    data[:,0]=prob_s
    data[:,1]=prob_d
    data[:,2]=prob_v
    p=Cross_entropyweight(data)
    #p_dis=p_distance(zuobiao_point,zuobiaolist,canpatternlist,si)
    p_total=p
    r = np.argmax(p_total)
    return database[canpatternlist[si[r]]],canpatternlist[si[r]]

def temdetect0d(tem):  # 检测剖面是否为0,单剖面版
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            if tem[h, x] != 0:
                return False
    return True


def patternsearchDi(Cdatabase, cdatabase, tem):  # 初始模型构建中依据重叠区用编辑距离搜索最适合模板类别的代号
    # Cdatabase和cdatabase都是重叠区定好的
    # 输入：Cdatabase--单个方向聚类好的模式序号库  cdatabase--单个方向的模式库 tem--待模拟的模式
    # 输出：Cdatabase[cc]--匹配到的模式类别
    s = tem.shape[0] * tem.shape[1] * tem.shape[2]
    drill1 = tem.reshape(s, 1)
    c1 = 99999
    cc = 0
    for n in range(Cdatabase.shape[0]):  # 遍历每个类别
        r = random.randint(0, len(Cdatabase[n]) - 1)  # 随机选一个序号
        tem2 = cdatabase[Cdatabase[n][r]]
        drill2 = tem2.reshape(s, 1)
        fun = hamming_distance(drill1, drill2)
        if fun <= c1:  # 选最小的序号
            c1 = fun
            cc = n
    return Cdatabase[cc]


def patternsearchDi_ou(Cdatabase, cdatabase, tem):  # 初始模型构建中依据重叠区用编辑距离搜索最适合模板类别的代号
    # Cdatabase和cdatabase都是重叠区定好的
    # 输入：Cdatabase--单个方向聚类好的模式序号库  cdatabase--单个方向的模式库 tem--待模拟的模式
    # 输出：Cdatabase[cc]--匹配到的模式类别
    s = tem.shape[0] * tem.shape[1] * tem.shape[2]
    drill1 = tem.reshape(s, 1)
    c1 = 99999
    cc = 0
    for n in range(Cdatabase.shape[0]):  # 遍历每个类别
        r = random.randint(0, len(Cdatabase[n]) - 1)  # 随机选一个序号
        tem2 = cdatabase[Cdatabase[n][r]]
        drill2 = tem2.reshape(s, 1)
        fun = Ou_distance1(drill1, drill2)
        if fun <= c1:  # 选最小的序号
            c1 = fun
            cc = n
    return Cdatabase[cc]


def temdetectD1(tem):  # 检测是否包含待模拟点是否大于阈值
    count = 0
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h, x, y] == -1:
                    count = count + 1
    if count >= 0.5 * (tem.shape[0]) * (tem.shape[1]) * (tem.shape[2]):
        return False
    return True


def initialroadlistAIR(m, template_h, template_x, template_y, lag):  # 设置随机路径
    # 输入：m--已经拟合地层顶底面的模型  template_h, template_x, template_y--模板大小  lag--重叠区
    # 输出：Roadlist--返回待模拟的随机路径
    lujing = []
    Roadlist = []  # 最终路径名单
    lujing = lujinglistAI(m, template_h, template_x, template_y, lag - 1)  # 顺序路径
    random.shuffle(lujing)  # 随机路径

    m2 = np.pad(m, lag, 'edge')  # 边缘拓展

    DevilTrigger = False
    H = m.shape[0]
    X = m.shape[1]
    Y = m.shape[2]
    Banlujing = []
    b = np.zeros((template_h, template_x, template_y), int)
    n = 0

    while n < len(lujing):
        if m2[lujing[n][0] + lag, lujing[n][1] + lag, lujing[n][2] + lag] == -1:

            h1 = lujing[n][0] + lag
            x1 = lujing[n][1] + lag
            y1 = lujing[n][2] + lag  # 储存边缘扩展后m2的坐标
            o1 = temextract(m2, template_h, template_x, template_y, h1, x1, y1)  # 提取对应位置的模式
            k = 0  # 重叠区计数器

            if temdetectD1(o1[0:lag, :, :]):
                # 上
                k = k + 1

            if temdetectD1(o1[template_h - lag:template_h, :, :]):
                # 下
                k = k + 1
            if temdetectD1(o1[:, 0:lag, :]):
                # 后
                k = k + 1
            if temdetectD1(o1[:, template_x - lag:template_x, :]):
                # 前
                k = k + 1
            if temdetectD1(o1[:, :, 0:lag]):
                # 左
                k = k + 1
            if temdetectD1(o1[:, :, template_y - lag:template_y]):
                # 右
                k = k + 1
            if (h1 > template_h - lag) and (k >= 2):
                m2 = temextractR(m2, b, h1, x1, y1)  # 返还0值，模式内有值也要归0
                Roadlist.append((h1 - lag, x1 - lag, y1 - lag))
            elif (h1 <= template_h - lag) and (k != 0):
                m2 = temextractR(m2, b, h1, x1, y1)
                Roadlist.append((h1 - lag, x1 - lag, y1 - lag))
            else:
                lujing.append(lujing[n])
        print(len(Roadlist), len(lujing) - n)
        n = n + 1
    print('roadlist initial done')
    return Roadlist


def buildfaultkerasS3(m, value, epoch, name, flag0):  # 重构地层体
    # 输入：value--当前需要训练的地层值 name--序号 flag0--标志为0
    # 输出：计算并保存拟合到的地层顶底面数据
    bottom, top = twodfaultsurface(m, value)  # 这里顶底返回值反了，后面弄巧成拙了

    xy_data, xy_value, inputdata = imgloaderforkeras2(top, flag0)  # 调整数据为可训练格式，xy_value为高程
    maxh = max(xy_value)
    xy_value = xy_value / m.shape[0]  # 数据归一化

    s = faultmodelbuild(xy_data, xy_value, inputdata, epoch)  # 计算地层整体顶面埋深
    s = s * m.shape[0]
    for h in range(len(inputdata)):
        top[int(inputdata[h][0] * float(m.shape[1]) + 0.5), int(inputdata[h][1] * float(m.shape[2]) + 0.5)] = int(
            s[h] + 0.5)  # 拟合地层顶面，得到地层整体顶面埋深俯视图

    xy_data, xy_value, inputdata = imgloaderforkeras2(bottom, flag0)
    minh = min(xy_value)
    xy_value = xy_value / m.shape[0]  # 数据归一化

    s = faultmodelbuild(xy_data, xy_value, inputdata, epoch)
    s = s * m.shape[0]
    for h in range(len(inputdata)):
        bottom[int(inputdata[h][0] * float(m.shape[1]) + 0.5), int(inputdata[h][1] * float(m.shape[2]) + 0.5)] = int(
            s[h] + 0.5)

    path = './database/init' + str(name) + '.npy'  # 地层顶面数据
    np.save(path, top)

    path = './database/inib' + str(name) + '.npy'  # 地层底面数据
    np.save(path, bottom)

    c = np.zeros((2), int)
    c[0] = maxh
    c[1] = minh
    path = './database/c' + str(name) + '.npy'  # 地层埋深最大、小值
    np.save(path, c)


def Simplecluster(cdatabase, U, order):  # 简单聚类方法，单个方向
    # 输入：cdatabase--分类后的模式库 U--阈值 order--序号
    # 输出：Cdatabase--单个方向聚类后的模式序号库
    Cdatabase = []
    d = []
    c = []
    for n in range(len(cdatabase)):
        if n not in c:  # 以没被聚类过的模式为起点
            d = []
            for m in range(n, len(cdatabase)):
                if cluster_distance(cdatabase[n], cdatabase[m]) <= U:  # 计算两模式的距离，若距离小于分类阈值则聚为一类
                    d.append(m)
                    c.append(m)
            Cdatabase.append(d)
    np.save('./database/clusters' + str(order) + '.npy', Cdatabase)
    return Cdatabase


def extend2dAI(m, h1, x1, y1):  # 9格子内选取一个分布最多的值
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


def initialPythiasub(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Cdatabase, cdatabase, zuobiaolist,
                     N, codelist, hardlist,order):
    # 自动初始化网格系统，分类加速ver2 直接载入路径版
    # 输入：m--顶底面拟合后仍有空值待模拟的模型
    # 输出：m--无空值的初始模型
    lujing = []
    Banlujing = []  # 已模拟黑名单
    lujing = initialroadlistAIR(m, template_h, template_x, template_y, lag)  # 设置随机路径
    print('initialize start')
    Cdatabase_density = np.load('./database/Cdatabase_density.npy')
    cdatabase_density = np.load('./database/cdatabase_density.npy')
    CdatabaseVp = np.load('./database/CdatabaseVp.npy')
    cdatabaseVp = np.load('./database/cdatabaseVp.npy')
    m_density = np.load('./density/study_density.npy')
    m_Vp = np.load('./Vp/study_area.npy')
    m2 = np.pad(m, lag, 'edge')  # 拓展
    m_density2 = np.pad(m_density, lag, 'edge')
    m_Vp2 = np.pad(m_Vp, lag, 'edge')
    Fin = m.shape[0] * m.shape[1] * m.shape[2] * 10  # 最大循环次数
    DevilTrigger = False
    H = m.shape[0]
    X = m.shape[1]
    Y = m.shape[2]
    ################重叠区选取器#####################
    ss = template_h * template_x * template_y
    dis = []  # 后左上
    disx = []  # 前
    disy = []  # 右
    dish = []  # 下
    b = np.zeros((template_h, template_x, template_y), int)
    reb = -np.ones((m2.shape[0]), int)
    d = []  # 待候选列表
    c1 = 99999  # 对比数
    cc = 0  # 对比序号

    for n in range(lag):
        dis.append(n)
    for n in range(template_x - lag, template_x):
        disx.append(n)
    for n in range(template_y - lag, template_y):
        disy.append(n)
    for n in range(template_h - lag, template_h):
        #############################################
        dish.append(n)

    num_Seq=0
    sqflag = 0
    while sqflag == 0:
        sqflag = 0
        for n in range(len(lujing)):
            h1 = lujing[n][0] + lag
            x1 = lujing[n][1] + lag
            y1 = lujing[n][2] + lag
            o1 = temextract(m2, template_h, template_x, template_y, h1, x1, y1)
            o1_d = temextract(m_density2, template_h, template_x, template_y, h1, x1, y1)
            o1_v = temextract(m_Vp2, template_h, template_x, template_y, h1, x1, y1)
            k = 0  # 重叠区计数器
            flag = 0
            canpatternlist0 = []  # 匹配到的模式库
            canpatternlist1 = []
            canpatternlist2 = []
            canpatternlist3 = []
            canpatternlist4 = []
            canpatternlist5 = []
            canpatternlist0_d = []  # 匹配到的模式库
            canpatternlist1_d = []
            canpatternlist2_d = []
            canpatternlist3_d = []
            canpatternlist4_d = []
            canpatternlist5_d = []

            canpatternlist0_v = []  # 匹配到的模式库
            canpatternlist1_v = []
            canpatternlist2_v = []
            canpatternlist3_v = []
            canpatternlist4_v = []
            canpatternlist5_v = []
            c = np.zeros((template_h, template_x, template_y), int)

            if temdetectD1(o1[0:lag, :, :]):
                # 上
                b = np.zeros((template_h, template_x, template_y), int)
                b[dis, :, :] = 1
                c[dis, :, :] = 1
                temo = o1 * b
                temo_d = o1_d * b
                temo_v = o1_v * b
                canpatternlist0 = patternsearchDi(Cdatabase[5], cdatabase[5], temo)  # 匹配模式
                canpatternlist0_d = patternsearchDi_ou(Cdatabase_density[5], cdatabase_density[5], temo_d)
                canpatternlist0_v = patternsearchDi_ou(CdatabaseVp[5], cdatabaseVp[5], temo_v)

            if temdetectD1(o1[template_h - lag:template_h, :, :]):
                # 下
                b = np.zeros((template_h, template_x, template_y), int)
                b[dish, :, :] = 1
                c[dish, :, :] = 1
                temo = o1 * b
                temo_d = o1_d * b
                temo_v = o1_v * b
                canpatternlist1 = patternsearchDi(Cdatabase[0], cdatabase[0], temo)
                canpatternlist1_d = patternsearchDi_ou(Cdatabase_density[0], cdatabase_density[0], temo_d)
                canpatternlist1_v = patternsearchDi_ou(CdatabaseVp[0], cdatabaseVp[0], temo_v)
                if temdetect0d(o1[template_h - 1, :, :]):
                    flag = 1

            if temdetectD1(o1[:, 0:lag, :]):
                # 后
                b = np.zeros((template_h, template_x, template_y), int)
                b[:, dis, :] = 1
                c[:, dis, :] = 1
                temo = o1 * b
                temo_d = o1_d * b
                temo_v = o1_v * b
                canpatternlist2 = patternsearchDi(Cdatabase[3], cdatabase[3], temo)
                canpatternlist2_d = patternsearchDi_ou(Cdatabase_density[3], cdatabase_density[3], temo_d)
                canpatternlist2_v = patternsearchDi_ou(CdatabaseVp[3], cdatabaseVp[3], temo_v)
                if temdetect0d(o1[:, 0, :]):    # 判断模板某个方向是否有零值，有则返回1，但基本不会有
                    flag = 1

            if temdetectD1(o1[:, template_x - lag:template_x, :]):
                # 前
                b = np.zeros((template_h, template_x, template_y), int)
                b[:, disx, :] = 1
                c[:, disx, :] = 1
                temo = o1 * b
                temo_d = o1_d * b
                temo_v = o1_v * b
                canpatternlist3 = patternsearchDi(Cdatabase[4], cdatabase[4], temo)
                canpatternlist3_d = patternsearchDi_ou(Cdatabase_density[4], cdatabase_density[4], temo_d)
                canpatternlist3_v = patternsearchDi_ou(CdatabaseVp[4], cdatabaseVp[4], temo_v)
                if temdetect0d(o1[:, template_x - 1, :]):
                    flag = 1

            if temdetectD1(o1[:, :, 0:lag]):
                # 左
                b = np.zeros((template_h, template_x, template_y), int)
                b[:, :, dis] = 1
                c[:, :, dis] = 1
                temo = o1 * b
                temo_d = o1_d * b
                temo_v = o1_v * b
                canpatternlist4 = patternsearchDi(Cdatabase[1], cdatabase[1], temo)
                canpatternlist4_d = patternsearchDi_ou(Cdatabase_density[1], cdatabase_density[1], temo_d)
                canpatternlist4_v = patternsearchDi_ou(CdatabaseVp[1], cdatabaseVp[1], temo_v)
                if temdetect0d(o1[:, :, 0]):
                    flag = 1

            if temdetectD1(o1[:, :, template_y - lag:template_y]):
                # 右
                b = np.zeros((template_h, template_x, template_y), int)
                b[:, :, disy] = 1
                c[:, :, disy] = 1
                temo = o1 * b
                temo_d = o1_d * b
                temo_v = o1_v * b
                canpatternlist5 = patternsearchDi(Cdatabase[2], cdatabase[2], temo)
                canpatternlist5_d = patternsearchDi_ou(Cdatabase_density[2], cdatabase_density[2], temo_d)
                canpatternlist5_v = patternsearchDi_ou(CdatabaseVp[2], cdatabaseVp[2], temo_v)
                if temdetect0d(o1[:, :, template_y - 1]):
                    flag = 1

            canpatternlist = []  # 整合6个方向都比较匹配的模式
            canpatternlist = list(set(canpatternlist0).union(set(canpatternlist1)))
            canpatternlist = list(set(canpatternlist).union(set(canpatternlist2)))
            canpatternlist = list(set(canpatternlist).union(set(canpatternlist3)))
            canpatternlist = list(set(canpatternlist).union(set(canpatternlist4)))
            canpatternlist = list(set(canpatternlist).union(set(canpatternlist5)))

            
            canpatternlistd = list(set(canpatternlist0_d).union(set(canpatternlist1_d)))
            canpatternlistd = list(set(canpatternlistd).union(set(canpatternlist2_d)))
            canpatternlistd = list(set(canpatternlistd).union(set(canpatternlist3_d)))
            canpatternlistd = list(set(canpatternlistd).union(set(canpatternlist4_d)))
            canpatternlistd = list(set(canpatternlistd).union(set(canpatternlist5_d)))

            
            canpatternlistv = list(set(canpatternlist0_v).union(set(canpatternlist1_v)))
            canpatternlistv = list(set(canpatternlistv).union(set(canpatternlist2_v)))
            canpatternlistv = list(set(canpatternlistv).union(set(canpatternlist3_v)))
            canpatternlistv = list(set(canpatternlistv).union(set(canpatternlist4_v)))
            canpatternlistv = list(set(canpatternlistv).union(set(canpatternlist5_v)))

            canpatternlist = list(set(canpatternlist).union(set(canpatternlistd)))
            canpatternlist = list(set(canpatternlist).union(set(canpatternlistv)))    
     
            canpatternlist = list(set(canpatternlist))
            print(n)
            if flag != 0:
                tem = np.zeros((template_h, template_x, template_y), int) #可能是为了填充表层空气
            else:
                temo = o1 * c
                temo_d=o1_d*c
                temo_v=o1_v*c
                tem,index= patternsearchAI2(temo, temo_d,temo_v,c, cdatabase[6], cdatabase_density[6],cdatabaseVp[6],canpatternlist, N,lujing[n],zuobiaolist)  # 随机挑选匹配到的模式
            m2 = TemplateHard(m2, tem, h1, x1, y1, hardlist)  # 填充模式

        path1 = './output/initial1' + str(order) + '.npy'
        np.save(path1, m2)

        F=[5,3,7]
        ls=[31,40,50,60,70,80,90,100,110,120,161]
        m2=clear(m2,ls[:len(ls)],order)
        print(m2.shape)
        for z in range(m2.shape[0] - 1):
            for y in range(m2.shape[1]):
                for x in range(m2.shape[2]):
                    for i in range(len(ls)-1):
                        if m2[z,y,x]==ls[i] and (m2[z+1,y,x] in ls[i:-1]):
                            if m2[z+1,y,x] not in ls[i:i+2]:
                                m2[z+1,y,x]=ls[i+1]
                        if m2[z,y,x]==ls[i] and m2[z+1,y,x] in ls[:i] and z>=5 and z<=155:
                           m2[z-3:z+3,y,x]=-1



        path2 = './output/initial2' + str(order) + '.npy'
        path3 = './output/outputinitial4' + str(order) + '.vtk'
        np.save(path2, m2)
        data = m2.transpose(-1, -2, 0)
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape)
        grid.point_data.scalars = np.ravel(data, order='F')
        grid.point_data.scalars.name = 'lithology'
        write_data(grid, path3)
        print('output')

        m2 = extendTimodel(m2, 5, 5, 5)
        lujing = []
        disss = []
        ms, disss = checkunreal2(m2, lag)
        if len(disss) <= 100:
            sqflag = 1
        else:
            num_Seq=num_Seq+1
            lujing = subroadlistinitialfornew(ms, disss, template_h, template_x, template_y, lag)  # 重新序贯模拟
        if num_Seq>10:
            sqflag = 1
    m = cut(m2, lag)
    path4 = './output/initial_zone' + str(order) + '.npy'
    np.save(path4,m)
    return m





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
        b = np.zeros((template_h, template_x, template_y), int)
        b[dish, :, :] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 左
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库1
        b = np.zeros((template_h, template_x, template_y), int)
        b[:, :, dis] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 右
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库2
        b = np.zeros((template_h, template_x, template_y), int)
        b[:, :, disy] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 后
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库3
        b = np.zeros((template_h, template_x, template_y), int)
        b[:, dis, :] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 前
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库4
        b = np.zeros((template_h, template_x, template_y), int)
        b[:, disx, :] = 1
        t = database[s] * b
        d1.append(t)
    cdatabase.append(d1)

    # 上
    d1 = []
    for s in range(le):  # 遍历模式库
        # 数据库5
        b = np.zeros((template_h, template_x, template_y), int)
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
                    tem = temextract(Exm2, template_h, template_x, template_y, h0, x0, y0)  # 提取模式
                    if temdetect(tem):  # 如果不包含待模拟点则为模式
                        database.append(tem)
                        zuobiaolist.append((h, x, y))
    return database, zuobiaolist


def extendTimodel(m, template_h, template_x, template_y):  # 全自动拓展插入硬数据的待模拟网格
    lag = max(template_h, template_x, template_y) // 2
    m2 = np.pad(m, lag, 'edge')
    d = []
    for h in range(lag, m2.shape[0] - lag):
        for x in range(lag, m2.shape[1] - lag):
            for y in range(lag, m2.shape[2] - lag):
                d.append((h, x, y))  # 储存需要遍历的矩阵坐标

    for i in range(lag):  # 往两边扩展lag个长度
        flag = 0
        for n in range(len(d)):
            h = d[n][0]
            x = d[n][1]
            y = d[n][2]
            if m2[h, x, y] == -1:
                value = extend2dAI(m2, h, x, y)  # 扩展取值
                flag = -1
                if value != -1:
                    m[h - lag, x - lag, y - lag] = value  # Ti附近扩展才有效

        if flag == 0:  # 已经扩展完成就不需要再扩展
            break
        m2 = np.pad(m, lag, 'edge')
        # 填充为1的
    return m


def initialAIforPythia(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U, hardlist, codelist,
                       valuels, flaglist):
    # 全自动初始化流程整合
    # 输入：
    # m为已经导入了Ti的模拟网格 template_h, template_x, template_y--模板大小
    # lag, lag_h, lag_x, lag_y--重叠区大小 N--备选模式个数 U--分类阈值 valuels--Ti包含的值
    # 输出：
    # m--无空值的初始模型R0，保存为npy并输出vtk


    database=np.load('./database/database.npy')
    Cdatabase=np.load('./database/Cdatabase.npy')
    cdatabase=np.load('./database/cdatabase.npy')
    zuobiaolist=np.load('./database/zuobiaolist.npy')

    #########################
    [a,b,c]=m.shape
    m1=m[:a,:b//2,:c//2]
    m2=m[:a,:b//2,c//2:401]
    m3=m[:a,b//2:401,:c//2]
    m4=m[:a,b//2:401,c//2:401]
    #m = initialPythiasub(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Cdatabase, cdatabase,
    #                     zuobiaolist, N, codelist, hardlist)  # 序贯模拟填补空值
    p1 = multiprocessing.Process(target=initialPythiasub, args=(m1, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Cdatabase, cdatabase,
                         zuobiaolist, N, codelist, hardlist,1))
    p2 = multiprocessing.Process(target=initialPythiasub, args=(m2, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Cdatabase, cdatabase,
                         zuobiaolist, N, codelist, hardlist,2))
    p3 = multiprocessing.Process(target=initialPythiasub, args=(m3, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Cdatabase, cdatabase,
                         zuobiaolist, N, codelist, hardlist,3))
    p4 = multiprocessing.Process(target=initialPythiasub, args=(m4, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Cdatabase, cdatabase,
                         zuobiaolist, N, codelist, hardlist,4))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    m[:a,:b//2,:c//2]=np.load('./output/initial_zone1.npy')
    m[:a,:b//2,c//2:401]=np.load('./output/initial_zone2.npy')
    m[:a,b//2:401,:c//2]=np.load('./output/initial_zone3.npy')
    m[:a,b//2:401,c//2:401]=np.load('./output/initial_zone4.npy')


    time_end = time.time()

    np.save('./output/outputinitial3.npy', m)

    print('initial done')
    print('初始化建模时间损耗:')
    print(time_end - time_start)
    # 初始化
    return m


time_start = time.time()
'''
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

codelist=[]
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

m = np.load('./output/initial.npy')
hardlist=[5,3,7]
valuels=[3,5,7,161,120,110,100,90,80,70,60,50,40,31,20]
flaglist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
initialAIforPythia(m, patternSizeh, patternSizex, patternSizey, lag, lag_h, lag_x, lag_y, N, U, hardlist, codelist,
                       values, flaglist)
'''
