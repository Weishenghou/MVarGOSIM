
import numpy as np


import time
from time import sleep

import multiprocessing
from tvtk.api import tvtk, write_data
#################################本程序自带子程序########################

import heapq
from TI import *

def templatecheck(m, z, x, y, code,lag):
    for n1 in range(z - lag, z + lag + 1):
        for n2 in range(x - lag, x + lag + 1):
            for n3 in range(y - lag, y + lag + 1):
                if m[n1, n2, n3] == code:
                    break
                    return True
    return True


def templatecheck1(m, z, x, y, c, lag):
    if c in m[z + lag, x - lag:x + lag + 1, y - lag:y + lag + 1] or c in m[z - lag:z + lag + 1, x + lag , y - lag:y + lag + 1] or c in m[z - lag:z + lag + 1, x - lag:x + lag + 1, y + lag] or c in m[z - lag, x - lag:x + lag + 1, y - lag:y + lag + 1] or c in m[z - lag:z + lag + 1, x - lag , y - lag:y + lag + 1] or c in m[z - lag:z + lag + 1, x - lag:x + lag + 1, y - lag]:
        flag=1
    else:
        flag=0
        print(flag)
    
    return flag


def templateclear(m, z, x, y, code,lag):
    for n1 in range(z - lag, z + lag + 1):
        for n2 in range(x - lag, x + lag + 1):
            for n3 in range(y - lag, y + lag + 1):
                if m[n1, n2, n3] == code:
                    print(z,x,y)
                    m[n1, n2, n3] = -1

    return m

def clear(m, code,order): #消除离散点
    # m--模型 code--地层编号 order--并行分区编号
    # 转存为区域模型npy
    for j in range(2, 25):
        lag = j
        for z in range(lag, m.shape[0] - lag,2):
            for x in range(lag, m.shape[1] - lag, 2):
                for y in range(lag, m.shape[2] - lag, 2):
                    for i in range(len(code)):
                        if code[i] in m[z - lag:z + lag + 1, x - lag:x + lag + 1,y - lag:y + lag + 1]:
                            if templatecheck1(m, z, x, y, code[i], lag) == 0:
                                m[z - lag:z + lag + 1, x - lag:x + lag + 1, y - lag:y + lag + 1][
                                    m[z - lag:z + lag + 1, x - lag:x + lag + 1, y - lag:y + lag + 1] == code[i]] = -1



    #np.save('./output/xiufu_zone'+str(order)+'.npy', m)
    return m

time_strat=time.time()

def xiufu(m):
    m = np.load('./output_random/initial1.npy')
    code = [20, 31, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    [a, b, c] = m.shape
    m1 = m[:a, :b // 2, :c // 2]
    m2 = m[:a, :b // 2, c // 2:c]
    m3 = m[:a, b // 2:b, :c // 2]
    m4 = m[:a, b // 2:b, c // 2:c]

    p1 = multiprocessing.Process(target=clear, args=(m1, code, 1))
    p2 = multiprocessing.Process(target=clear, args=(m2, code, 2))
    p3 = multiprocessing.Process(target=clear, args=(m3, code, 3))
    p4 = multiprocessing.Process(target=clear, args=(m4, code, 4))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    m[:a, :b // 2, :c // 2] = np.load('./output_random/xiufu_zone1.npy')
    m[:a, :b // 2, c // 2:c] = np.load('./output_random/xiufu_zone2.npy')
    m[:a, b // 2:b, :c // 2] = np.load('./output_random/xiufu_zone3.npy')
    m[:a, b // 2:b, c // 2:c] = np.load('./output_random/xiufu_zone4.npy')

    np.save('./output_random/xiufu_zone.npy', m)
    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output_random/xiufu_zone.vtk')
    print('output')
    time_end = time.time()
    print(time_end - time_strat)


'''
m=np.load('./output_random/initial1.npy')
code=[20,31,40,50,60,70,80,90,100,110,120]

[a, b, c] = m.shape
m1 = m[:a, :b // 2, :c // 2]
m2 = m[:a, :b // 2, c // 2:c]
m3 = m[:a, b // 2:b, :c // 2]
m4 = m[:a, b // 2:b, c // 2:c]

p1 = multiprocessing.Process(target=clear, args=(m1, code,1))
p2 = multiprocessing.Process(target=clear, args=(m2, code,2))
p3 = multiprocessing.Process(target=clear, args=(m3, code,3))
p4 = multiprocessing.Process(target=clear, args=(m4, code,4))

p1.start()
p2.start()
p3.start()
p4.start()
p1.join()
p2.join()
p3.join()
p4.join()

m[:a, :b // 2, :c // 2] = np.load('./output_random/xiufu_zone1.npy')
m[:a, :b // 2, c // 2:c] = np.load('./output_random/xiufu_zone2.npy')
m[:a, b // 2:b, :c // 2] = np.load('./output_random/xiufu_zone3.npy')
m[:a, b // 2:b, c // 2:c] = np.load('./output_random/xiufu_zone4.npy')

np.save('./output_random/xiufu_zone.npy',m)
data=m.transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),  dimensions=data.shape)
grid.point_data.scalars = np.ravel(data,order='F')
grid.point_data.scalars.name = 'lithology'
write_data(grid, './output_random/xiufu_zone.vtk')
print('output')
time_end=time.time()
print(time_end-time_strat)

'''
