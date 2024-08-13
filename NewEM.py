#!/usr/bin/env python
# coding: utf-8

# In[4]:


######################ver0.5
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pylab
import numpy as np
import time
from PIL import Image
import random
from pathlib2 import Path#python3环境下
#from pathlib import Path  #python2环境下
import os
from tvtk.api import tvtk, write_data 
import threading
from time import sleep
from tqdm import tqdm
#################################本程序自带子程序########################
from zuobiao import*
from Pdatabase import*
from patterntool import*
from editdistance import*
from roadlist import*
import heapq
from listheap import*  
from initialGrid import*
from Patchmatch import*
from cluster import*
from PatchmatchZ import*
from entropyweight import*
from AIinitial import*

def cal_distanceZ(a, b, A_padding, B, p_size):
    #print(b)
    p = p_size // 2
    patch_a = A_padding[:, a[0]-p:a[0]+p+1, a[1]-p:a[1]+p+1]
    #print(a[0]-p,a[0]+p+1,patch_a.shape[0])
    patch_b = B[: b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1]
    #print(patch_a.shape[2])

    temp = patch_b - patch_a
    smstr=np.nonzero(temp)
    #print smstr
    dist=np.shape(smstr[0])[0]
    #print dist
    return dist

def databaseclusterAIplus(database,U):#模式分类forEM
    Cdatabase=Simplecluster2(database,U) #多尺度并行
    np.save('./database/Adatabase',Cdatabase)
    return Cdatabase

def databaseclusterAIplusforGosim(cdatabase,U):#模式分类
    print('start')


    p1= multiprocessing.Process(target=Simplecluster, args=(cdatabase[0],U,0)) 
    print('process start') 
    p1.start()

    p2= multiprocessing.Process(target=Simplecluster, args=(cdatabase[1],U,1)) 
    print('process start') 
    p2.start()

    p3= multiprocessing.Process(target=Simplecluster, args=(cdatabase[2],U,2)) 
    print('process start') 
    p3.start()

    p4= multiprocessing.Process(target=Simplecluster, args=(cdatabase[3],U,3)) 
    print('process start') 
    p4.start()

    p5= multiprocessing.Process(target=Simplecluster, args=(cdatabase[4],U,4)) 
    print('process start') 
    p5.start()

    p6= multiprocessing.Process(target=Simplecluster, args=(cdatabase[5],U,5)) 
    print('process start') 
    p6.start()
    
    p7= multiprocessing.Process(target=Simplecluster, args=(cdatabase[6],U,6)) 
    print('process start') 
    p7.start()

    

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    print('process end')

    Cdatabase=[]
    cc1=np.load('./database/clusters0.npy')
    cc2=np.load('./database/clusters1.npy')
    cc3=np.load('./database/clusters2.npy')
    cc4=np.load('./database/clusters3.npy')
    cc5=np.load('./database/clusters4.npy')
    cc6=np.load('./database/clusters5.npy')
    cc7=np.load('./database/clusters6.npy')
    

    Cdatabase.append(cc1)
    Cdatabase.append(cc2)
    Cdatabase.append(cc3)
    Cdatabase.append(cc4)
    Cdatabase.append(cc5)
    Cdatabase.append(cc6)
    Cdatabase.append(cc7)

    os.remove('./database/clusters0.npy')
    os.remove('./database/clusters1.npy')
    os.remove('./database/clusters2.npy')
    os.remove('./database/clusters3.npy')
    os.remove('./database/clusters4.npy')
    os.remove('./database/clusters5.npy')
    os.remove('./database/clusters6.npy')

    #np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase






def initializationEMold(A, Tidatabase,Cdatabase,Tizuobiaolist, p_size):#初始化，构建映射
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    p = p_size // 2
    Lti=len(Cdatabase)#经过聚类后模板数据库的总长度
    #A_padding = np.ones([A_h+p*2, A_l+p*2,A_w+p*2]) * np.nan
    A_padding = -np.ones([A_h+p*2, A_l+p*2,A_w+p*2])
    A_padding[p:A_h+p, p:A_l+p,p:A_w+p] = A
    f = np.zeros([A_h, A_l,A_w],int)
    dist = np.zeros([A_h, A_l,A_w])
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                a = np.array([i+p, j+p,k+p])
                b=random.randint(0,Lti-1)
                c=len(Cdatabase[b])
                d=random.randint(0,c-1)
                f[i, j,k] = Cdatabase[b][d]
                dist[i, j,k] = calDIST(a,A_padding, Tidabase[f[i, j,k]], p_size)
    for n in range(len(Tizuobiaolist)):
        f[Tizuobiaolist[n]]=n
        dist[Tizuobiaolist[n]]=0
        #print calDIST(Tizuobiaolist[n]+p,A_padding, Tidabase[ f[Tizuobiaolist[n]]], p_size)
        #测试是否俱合
    #print f
    #加上硬数据
    return f,dist, A_padding

def propagationEMold(f, a, dist, A_padding, Tidatabase, p_size, is_odd):#传播
    A_h = np.size(A_padding, 0) - p_size + 1
    A_l = np.size(A_padding, 1) - p_size + 1
    A_w = np.size(A_padding, 2) - p_size + 1
    p=p_size//2
    x = a[0]-p
    y = a[1]-p
    z = a[2]-p
    if is_odd:
        d_left = dist[max(x-1, 0), y,z]
        d_up = dist[x, max(y-1, 0),z]
        d_forward=dist[x,y,max(z-1, 0)]
        d_current = dist[x, y,z]
        idx = np.argmin(np.array([d_current, d_left, d_up,d_forward]))
        if idx == 1:
            f[x, y,z] = f[max(x - 1, 0), y,z]
            dist[x, y,z] = calDIST(a,A_padding,Tidabase[f[i, j,k]], p_size)
        if idx == 2:
            f[x, y,z] = f[x, max(y - 1, 0),z]
            dist[x, y,z] = calDIST(a,A_padding,Tidabase[f[i, j,k]], p_size)
        if idx == 3:
            f[x, y,z] = f[x, y,max(z - 1, 0)]
            dist[x, y,z] = calDIST(a,A_padding,Tidabase[f[i, j,k]], p_size)
    else:
        d_right = dist[min(x + 1, A_h-1), y,z]
        d_down = dist[x, min(y + 1, A_l-1),z]
        d_current = dist[x, y,z]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y,z] = f[min(x + 1, A_h-1), y,z]
            dist[x, y,z] = calDIST(a,A_padding,Tidabase[f[i, j,k]], p_size)
        if idx == 2:
            f[x, y,z] = f[x, min(y + 1, A_l-1),z]
            dist[x, y,z] = calDIST(a,A_padding,Tidabase[f[i, j,k]], p_size)
        if idx == 3:
            f[x, y,z] = f[x,y, min(z + 1, A_w-1)]
            dist[x, y,z] = calDIST(a,A_padding,Tidabase[f[i, j,k]], p_size)
    return f,dist

def random_searchEMold(f, a, dist, A_padding, Tidatabase, Cdatabase,p_size):#随机搜索
    p = p_size // 2
    x = a[0]-p
    y = a[1]-p
    z = a[2]-p
    arrylist=[]
    for n in range(len(Cdatabase)):
        nigger=random.randint(0,len(Cdatabase[n])-1)
        distcal= calDIST(a,A_padding, Tidabase[f[i, j,k]], p_size)
        arrylist.append(distcal)
    idx = np.argmin (arrylist)
    arrylist2=[]
    for n in range(len(Cdatabase[idx])):
        distcal= calDIST(a,A_padding, Tidabase[Cdatabase[idx][n]], p_size)
        arrylist2.append(distcal)
    idy = np.argmin (arrylist)
    dist[x,y,z]= arrylist[idy]
    f[x,y,z]=Cdatabase[idx][idy]
    return f,dist
 

def NNSBEMold(img, Tidatabase,Cdatabase,Tizuobiaolist, p_size, itr):#寻找最近零并行版
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding= initializationEM(img, Tidatabase,Cdatabase,Tizuobiaolist, p_size)
    p=p_size//2
    print("initialization done")
    for itr in range(1, itr+1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_l - 1, -1, -1):
                    for k in range(A_w - 1, -1, -1):
                        a = np.array([i+p, j+p,k+p])
                        f,dist=propagationEM(f, a, dist, img_padding, Tidatabase, p_size, False)
                        f,dist=random_searchEM(f, a, dist, img_padding, Tidatabase, Cdatabase, p_size)
        else:
            for i in range(A_h):
                for j in range(A_l):
                    for k in range(A_w):
                        a = np.array([i+p, j+p,k+p])
                        f,dist=propagationEM(f, a, dist, img_padding, Tidatabase, p_size, True)
                        f,dist=random_searchEM(f, a, dist, img_padding, Tidatabase, Cdatabase, p_size)
        print("iteration: %d"%(itr))
    return f,dist

def reconstructionEMold(A,f,dist,Tidatabase,size):
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                temp[i,j,k] = Tidatabase[f[i,j,k]][size//2,size//2,size//2]
    return temp

def NewEMold(m,Tidatabase,Cdatabase,Tizuobiaolist,size,itr):#new em-like iteration method
    #size为模板大小，itr为迭代次数
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    f,dist=NNSBEM(m,Tidatabase,Cdatabase,Tizuobiaolist,p_size, itr)
    print("搜索步骤结束，开始更新")
    Re=reconstructionEM(m,f,dist,Tidatabase)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re

def newinitialold(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U):
    #m为空白模拟网格
    time_start1=time.time()
    m,Tilist,zuobiaolist=sectionloadandextend(m,template_x,template_y,0,1)
    #导入剖面
    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial1.vtk') 
    
    m=extendTimodel(m,template_h,template_x,template_y)#拓展模拟网格
    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial2.vtk') 
    print('extend done')
    
    my_file = Path("./database/Cdatabase.npy")
    if my_file.exists():
        Cdatabase=np.load('./database/Cdatabase.npy')
        Tidatabase=np.load('./database/Tidatabase.npy')
        zuobiaolist=np.load('./database/zuobiaolist.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Tidatabase,zuobiaolist=databasebuildAI(m,template_h,template_x,template_y)#数据库构建
        np.save('./database/Tidatabase.npy',Tidatabase)
        np.save('./database/zuobiaolist.npy',zuobiaolist)
        Cdatabase=databaseclusterAIplus(Tidatabase,U)
        np.save('./database/Cdatabase.npy',Cdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)
    return m,Tidatabase,Cdatabase,zuobiaolist









#######################################################new
def databaseclusterAIplussub(Exm,template_h,template_x,template_y,zuobiaolist,U,name):#简单聚类方法for new EM
    #U为阈值
    Cdatabase=[]
    d=[]
    c=[]
    lag=max(template_h,template_x,template_y)
    Exm2=np.pad(Exm,lag,'edge')#拓展
    for n in range(len(zuobiaolist)):
        if n not in c:
            d=[]
            for m in range(n,len(zuobiaolist)):
                a=template1(Exm2,template_h,template_x,template_y,zuobiaolist[n][0]+lag,zuobiaolist[n][1]+lag,zuobiaolist[n][2]+lag)
                b=template1(Exm2,template_h,template_x,template_y,zuobiaolist[m][0]+lag,zuobiaolist[m][1]+lag,zuobiaolist[m][2]+lag)
                if cluster_distance(a, b)<=U:
                    d.append(m)
                    c.append(m)
            Cdatabase.append(d)
    #print len(c)
    path='./database/cdatabase'+str(name)+'.npy'
    np.save(path,Cdatabase)
    return Cdatabase

def reNNSBsub(N):#重组
    p=p_size//2
    for n in range(len(items)):
        path='./database/patchmatchprocess('+str(name)+')('+str(n)+')f.npy'
        np.load
    return f,dist

def databaseclusterAIplus2(Exm,template_h,template_x,template_y,zuobiaolist,U):#简单聚类方法for new EM
    #U为阈值
    core=20
    items=apartlist(zuobiaolist, int(len(zuobiaolist)/core))
    processes=list()
    for n in range(len(items)):
        s=multiprocessing.Process(target=databaseclusterAIplussub, args=(Exm,template_h,template_x,template_y,items[n],U,n)) 
        print('process:',n)
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    Cdatabase=[]
    for n in range(len(items)):
        path='./database/cdatabase'+str(n)+'.npy'
        cdatabase=np.load(path)
        Cdatabase.append(cdatabase)
    return Cdatabase


def databasebuildNewEM(Exm,template_h,template_x,template_y):#智能构建模式库
    #Exm为已经完成了拓展的模拟网格
    lag=max(template_h,template_x,template_y)
    Exm2=np.pad(Exm,lag,'edge')#拓展
    zuobiaolist=[]
    for h in range(Exm.shape[0]):
        for x in range(Exm.shape[1]):
            for y in range(Exm.shape[2]):
                if Exm[h,x,y]!=-1:
                    h0=h+lag
                    x0=x+lag
                    y0=y+lag
                    tem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
                    if temdetect(tem):#如果不包含待模拟点则为模板
                        zuobiaolist.append((h,x,y))
    return zuobiaolist

def newinitial(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U):
    #m为空白模拟网格
    time_start1=time.time()
    m,Tilist,zuobiaolist=sectionloadandextend(m,template_x,template_y,0,1)
    #导入剖面
    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial1.vtk') 
    
    m=extendTimodel(m,template_h,template_x,template_y)#拓展模拟网格
    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial2.vtk') 
    print('extend done')
    
    my_file = Path("./database/Cdatabase.npy")
    if my_file.exists():
        Cdatabase=np.load('./database/Cdatabase.npy')
        zuobiaolist=np.load('./database/zuobiaolist.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        zuobiaolist=databasebuildNewEM(m,template_h,template_x,template_y)#数据库构建
        np.save('./database/zuobiaolist.npy',zuobiaolist)
        Cdatabase=databaseclusterAIplus2(m,template_h,template_x,template_y,zuobiaolist,U)
        np.save('./database/Cdatabase.npy',Cdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)
    return m,Cdatabase,zuobiaolist



def initializationEM(A, Cdatabase,zuobiaolist, p_size):#初始化，构建映射
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    p = p_size // 2
    A_padding = np.ones([A_h+p*2, A_l+p*2,A_w+p*2]) * np.nan
    A_padding[p:A_h+p, p:A_l+p,p:A_w+p] = A
    f = np.zeros([A_h, A_l,A_w],int)
    dist = np.zeros([A_h, A_l,A_w])
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                a = np.array([i+p, j+p,k+p])
                e=random.randint(0,len(Cdatabase)-1)
                b=random.randint(0,len(Cdatabase[e])-1)
                c=len(Cdatabase[e][b])
                d=random.randint(0,c-1)
                f[i, j,k] = Cdatabase[e][b][d]
                Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[f[i, j,k]][0]+p,zuobiaolist[f[i, j,k]][1]+p,zuobiaolist[f[i, j,k]][2]+p)
                dist[i, j,k] = calDIST(a,A_padding, Ti, p_size)
    for n in range(len(zuobiaolist)):
        f[zuobiaolist[n][0],zuobiaolist[n][1],zuobiaolist[n][2]]=n
        dist[zuobiaolist[n][0],zuobiaolist[n][1],zuobiaolist[n][2]]=0
        #print calDIST(Tizuobiaolist[n]+p,A_padding, Tidabase[ f[Tizuobiaolist[n]]], p_size)
        #测试是否俱合
    #print f
    #加上硬数据
    return f,dist, A_padding

def propagationEM(A,f, a, dist, A_padding, zuobiaolist, p_size, is_odd):#传播
    A_h = np.size(A_padding, 0) - p_size + 1
    A_l = np.size(A_padding, 1) - p_size + 1
    A_w = np.size(A_padding, 2) - p_size + 1
    p=p_size//2
    x = a[0]-p
    y = a[1]-p
    z = a[2]-p
    if is_odd:
        d_left = dist[max(x-1, 0), y,z]
        d_up = dist[x, max(y-1, 0),z]
        d_forward=dist[x,y,max(z-1, 0)]
        d_current = dist[x, y,z]
        idx = np.argmin(np.array([d_current, d_left, d_up,d_forward]))
        if idx == 1:
            f[x, y,z] = f[max(x - 1, 0), y,z]
            Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[f[x, y,z]][0]+p,zuobiaolist[f[x, y,z]][1]+p,zuobiaolist[f[x, y,z]][2]+p)
            dist[x, y,z] = calDIST(a,A_padding,Ti, p_size)
        if idx == 2:
            f[x, y,z] = f[x, max(y - 1, 0),z]
            Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[f[x, y,z]][0]+p,zuobiaolist[f[x, y,z]][1]+p,zuobiaolist[f[x, y,z]][2]+p)
            dist[x, y,z] = calDIST(a,A_padding,Ti, p_size)
        if idx == 3:
            f[x, y,z] = f[x, y,max(z - 1, 0)]
            Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[f[x, y,z]][0]+p,zuobiaolist[f[x, y,z]][1]+p,zuobiaolist[f[x, y,z]][2]+p)
            dist[x, y,z] = calDIST(a,A_padding,Ti, p_size)
    else:
        d_right = dist[min(x + 1, A_h-1), y,z]
        d_down = dist[x, min(y + 1, A_l-1),z]
        d_current = dist[x, y,z]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y,z] = f[min(x + 1, A_h-1), y,z]
            Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[f[x, y,z]][0]+p,zuobiaolist[f[x, y,z]][1]+p,zuobiaolist[f[x, y,z]][2]+p)
            dist[x, y,z] = calDIST(a,A_padding,Ti, p_size)
        if idx == 2:
            f[x, y,z] = f[x, min(y + 1, A_l-1),z]
            Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[f[x, y,z]][0]+p,zuobiaolist[f[x, y,z]][1]+p,zuobiaolist[f[x, y,z]][2]+p)
            dist[x, y,z] = calDIST(a,A_padding,Ti, p_size)
        if idx == 3:
            f[x, y,z] = f[x,y, min(z + 1, A_w-1)]
            Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[f[x, y,z]][0]+p,zuobiaolist[f[x, y,z]][1]+p,zuobiaolist[f[x, y,z]][2]+p)
            dist[x, y,z] = calDIST(a,A_padding,Ti, p_size)
    return f,dist

def subrandom_searchEM(A,f, a, dist, A_padding,  Cdatabase,zuobiaolist,p_size,q):#随机搜索
    p = p_size // 2
    arrylist=[]
    arrylist2=[]
    for n in range(len(Cdatabase)):
        nigger=random.randint(0,len(Cdatabase[n])-1)
        Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[Cdatabase[n][nigger]][0]+p,zuobiaolist[Cdatabase[n][nigger]][1]+p,zuobiaolist[Cdatabase[n][nigger]][2]+p)
        distcal= calDIST(a,A_padding,Ti, p_size )
        arrylist.append(distcal)
        arrylist2.append(nigger)
    idx = np.argmin (arrylist)
    idy=arrylist2[idx]
    a=np.zeros((2),int)
    a[0]= arrylist[idx]
    a[1]=Cdatabase[idx][idy]
    q.put(a)

def random_searchEM(A,f, a, dist, A_padding,  Cdatabase,zuobiaolist,p_size):#随机搜索
    p = p_size // 2
    x = a[0]-p
    y = a[1]-p
    z = a[2]-p
    arrylist=[]
    processes=list()
    q = multiprocessing.Queue()
    for n in range(len(Cdatabase)):
        s=multiprocessing.Process(target=subrandom_searchEM, args=(A,f, a, dist, A_padding,  Cdatabase[n],zuobiaolist,p_size,q)) 
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    results = [q.get() for s in processes]
    sfa=99999
    for n in range(len(results)):
        if results[n][0]<=sfa:
            f[x,y,z]=results[n][1]
            dist[x,y,z]=results[n][0]
            sfa=results[n][0]
    print('one done')
    return f,dist
 
def NNSBEM(img, Cdatabase,zuobiaolist, p_size, itr):#寻找最近零并行版
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding= initializationEM(img, Cdatabase,zuobiaolist, p_size)
    p=p_size//2
    print("initialization done")
    for itr in range(1, itr+1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_l - 1, -1, -1):
                    for k in range(A_w - 1, -1, -1):
                        a = np.array([i+p, j+p,k+p])
                        f,dist=propagationEM(img,f, a, dist, img_padding, zuobiaolist, p_size, False)
                        f,dist=random_searchEM(img,f, a, dist, img_padding, Cdatabase,zuobiaolist, p_size)
        else:
            for i in range(A_h):
                for j in range(A_l):
                    for k in range(A_w):
                        a = np.array([i+p, j+p,k+p])
                        f,dist=propagationEM(img,f, a, dist, img_padding, zuobiaolist, p_size, True)
                        f,dist=random_searchEM(img,f, a, dist, img_padding, Cdatabase, zuobiaolist,p_size)
        print("iteration: %d"%(itr))
    return f,dist

def reconstructionEM(A,f,dist,zuobiaolist,size):
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                temp[i,j,k] = A[zuobiaolist[f[i,j,k]][0],zuobiaolist[f[i,j,k]][1],zuobiaolist[f[i,j,k]][2]][size//2,size//2,size//2]
    return temp

def NewEM(m,Cdatabase,Tizuobiaolist,size,itr):#new em-like iteration method
    #size为模板大小，itr为迭代次数
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    f,dist=NNSBEM(m,Cdatabase,Tizuobiaolist,size, itr)
    print("搜索步骤结束，开始更新")
    Re=reconstructionEM(m,f,dist,zuobiaolist,size)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re
#######################################################################
def patternSelhi(pattern):#提取模版中的元素种类
    yuansulist=[]
    for h in range(pattern.shape[0]):
        for x in range(pattern.shape[1]):
            for y in range(pattern.shape[2]):
                if pattern[h,x,y] not in yuansulist:
                    yuansulist.append(pattern[h,x,y])
    yuansulist.sort()#防止顺序导致判断异常
    return yuansulist

def patternSelhi2(pattern):#提取模版中的元素种类,无视未模拟区域ver
    yuansulist=[]
    for h in range(pattern.shape[0]):
        for x in range(pattern.shape[1]):
            for y in range(pattern.shape[2]):
                if pattern[h,x,y] not in yuansulist:
                    if pattern[h,x,y]!=-1:
                        yuansulist.append(pattern[h,x,y])
    yuansulist.sort()#防止顺序导致判断异常
    return yuansulist
def listcompair(yuansulist,listdatabase):#对比是否存在于类型列表中--listdatabase构建阶段
    if yuansulist not in listdatabase:
        listdatabase.append(yuansulist)
    return listdatabase

def listdetect(yuansulist,listdatabase):#对比是否存在于类型列表中，是则返回真
    if yuansulist not in listdatabase:
        return False
    return True

def listdetect2(yuansulist,listdatabase):#对比是否存在于类型列表中返还列表所在序号
    return listdatabase.index(yuansulist)

def C1newdatabasebuildAI(Exm,template_h,template_x,template_y,):#检查模版中包含类型，据此分类成基础数据库
    #Exm为已经完成了拓展的模拟网格
    lag=max(template_h,template_x,template_y)
    Exm2=np.pad(Exm,lag,'edge')#拓展
    listdatabase=[]#构建元素列表数据库
    C1database=[]
    zuobiaolist=[]
    for h in range(Exm.shape[0]):
        for x in range(Exm.shape[1]):
            for y in range(Exm.shape[2]):
                if Exm[h,x,y]!=-1:
                    h0=h+lag
                    x0=x+lag
                    y0=y+lag
                    tem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
                    yuansulist=patternSelhi(tem)
                    if temdetect(tem):#如果不包含待模拟点则为模板
                        if listdetect(yuansulist,listdatabase):
                            youccyou=listdetect2(yuansulist,listdatabase)
                            C1database[youccyou].append(tem)
                            zuobiaolist[youccyou].append((h,x,y))
                        else:
                            listdatabase=listcompair(yuansulist,listdatabase)
                            newdatabase=[]
                            newzuobiaobase=[]
                            newdatabase.append(tem)
                            newzuobiaobase.append((h,x,y))
                            C1database.append(newdatabase)
                            zuobiaolist.append(newzuobiaobase)
    return C1database,listdatabase,zuobiaolist

def C1newdatabasebuildAI2(Exm,template_h,template_x,template_y,):#检查模版中包含类型，据此分类成基础数据库
    #Exm为已经完成了拓展的模拟网格
    lag=max(template_h,template_x,template_y)
    Exm2=np.pad(Exm,lag,'edge')#拓展
    listdatabase=[]#构建元素列表数据库
    zuobiaolist=[]
    for h in range(Exm.shape[0]):
        for x in range(Exm.shape[1]):
            for y in range(Exm.shape[2]):
                if Exm[h,x,y]!=-1:
                    h0=h+lag
                    x0=x+lag
                    y0=y+lag
                    tem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
                    yuansulist=patternSelhi(tem)
                    if temdetect(tem):#如果不包含待模拟点则为模板
                        if listdetect(yuansulist,listdatabase):
                            youccyou=listdetect2(yuansulist,listdatabase)
                            zuobiaolist[youccyou].append((h,x,y))
                        else:
                            listdatabase=listcompair(yuansulist,listdatabase)
                            newzuobiaobase=[]
                            newzuobiaobase.append((h,x,y))
                            zuobiaolist.append(newzuobiaobase)
    return listdatabase,zuobiaolist

def C1Newdatabasebuidcluster(Exm,template_h,template_x,template_y,zuobiaolist,U):#对提取到的zuobiaolistdatabase进行子类别聚类
    Cdatabase=[]
    for n in range(len(zuobiaolist)):
        path='./database/zuobiaolist'+str(n)+'.npy'
        np.save(path,zuobiaolist[n])
    for n in range(len(zuobiaolist)):
        path='./database/zuobiaolist'+str(n)+'.npy'
        a=np.load(path)
        print('No.')
        print(n)
        print('start')
        data=databaseclusterAIplus2(Exm,template_h,template_x,template_y,a,U)
        Cdatabase.append(data)
    return Cdatabase

def C1newinitial(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U):
    #m为空白模拟网格
    time_start1=time.time()
    m,Tilist,zuobiaolist=sectionloadandextend(m,template_x,template_y,0,1)
    #导入剖面
    
    
    m=extendTimodel(m,template_h,template_x,template_y)#拓展模拟网格
    
    
    my_file = Path("./database/Cdatabase.npy")
    if my_file.exists():
        Cdatabase=np.load('./database/Cdatabase.npy')
        zuobiaolist=np.load('./database/zuobiaolist.npy')
        listdatabase=np.load('./database/listdatabase.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        listdatabase,zuobiaolist=C1newdatabasebuildAI2(m,template_h,template_x,template_y,)#数据库构建
        np.save('./database/zuobiaolist.npy',zuobiaolist)
        np.save('./database/listdatabase.npy',listdatabase)
        Cdatabase=[]
        #Cdatabase=C1Newdatabasebuidcluster(m,template_h,template_x,template_y,zuobiaolist,U)
        np.save('./database/Cdatabase.npy',Cdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)
    return m,Cdatabase,zuobiaolist

def C1initializationEM(A, Cdatabase,zuobiaolist, p_size):#初始化，构建映射
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    p = p_size // 2
    A_padding = np.ones([A_h+p*2, A_l+p*2,A_w+p*2]) * np.nan
    A_padding[p:A_h+p, p:A_l+p,p:A_w+p] = A
    f = np.zeros([A_h, A_l,A_w],int)
    dist = np.zeros([A_h, A_l,A_w])
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                a = np.array([i+p, j+p,k+p])
                e=random.randint(0,len(Cdatabase)-1)
                b=random.randint(0,len(Cdatabase[e])-1)
                c=len(Cdatabase[e][b])
                d=random.randint(0,c-1)
                f[i, j,k] = Cdatabase[e][b][d]
                Ti=template1(A_padding,p_size,p_size,p_size,zuobiaolist[f[i, j,k]][0]+p,zuobiaolist[f[i, j,k]][1]+p,zuobiaolist[f[i, j,k]][2]+p)
                dist[i, j,k] = calDIST(a,A_padding, Ti, p_size)
    for n in range(len(zuobiaolist)):
        f[zuobiaolist[n][0],zuobiaolist[n][1],zuobiaolist[n][2]]=n
        dist[zuobiaolist[n][0],zuobiaolist[n][1],zuobiaolist[n][2]]=0
        #print calDIST(Tizuobiaolist[n]+p,A_padding, Tidabase[ f[Tizuobiaolist[n]]], p_size)
        #测试是否俱合
    #print f
    #加上硬数据
    return f,dist, A_padding

#########################################Patchmatchrebuild################
def RecodeTIextendforEM(section,m,template_x,template_y,x1,y1,x2,y2):#EM迭代用剖面提取，(x,y）为剖面定位的一组坐标
    #m为已经完成扩展的模拟网格
    dx=[]
    dy=[]
    
    lag=max(template_x,template_y)//2
    ms=-np.ones((m.shape[0],m.shape[1],m.shape[2]),int)
    sectionload_x(ms,section,x1,y1,x2,y2)#独立载入防止造成多剖面混乱
    Tizuobiaox=-np.ones((m.shape[0],m.shape[1],m.shape[2]), int)
    Tizuobiaoy=-np.ones((m.shape[0],m.shape[1],m.shape[2]), int)
    
    if abs(x1-x2)>=lag:
        for n1 in range(min(x1,x2),max(x1,x2)+1):
            dx.append(n1)
    else:
        for n1 in range(max(0,min(x1,x2)-lag),min(max(x1,x2)+lag,m.shape[1]-1)+1):
            dx.append(n1)
            
    if abs(y1-y2)>=lag:
        for n1 in range(min(y1,y2),max(y1,y2)+1):
            dy.append(n1)
    else:
        for n1 in range(max(0,min(y1,y2)-lag),min(max(y1,y2)+lag,m.shape[2]-1)+1):
            dy.append(n1)
    
    for h in range(ms.shape[0]):
        for x in range(ms.shape[1]):
            for y in range(ms.shape[2]):
                if ms[h,x,y]!=-1:
                    Tizuobiaox[h,x,y]=x
                    Tizuobiaoy[h,x,y]=y
    temp=ms[:,dx,:]
    fow=temp[:,:,dy]
    Tizuobiaoxt=Tizuobiaox[:,dx,:]
    Tizuobiaox=Tizuobiaoxt[:,:,dy]
    Tizuobiaoyt=Tizuobiaoy[:,dx,:]
    Tizuobiaoy=Tizuobiaoyt[:,:,dy]
    c=max(fow.shape[1],fow.shape[2])
    Tizuobiaoh=-np.ones((fow.shape[0],fow.shape[1],fow.shape[2]), int)
    for h in range(Tizuobiaoh.shape[0]):
        for x in range(Tizuobiaoh.shape[1]):
            for y in range(Tizuobiaoh.shape[2]):
                Tizuobiaoh[h,x,y]=h
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=extendTimodelsave,args=(fow,c,c,c,1))
    
    p2 = multiprocessing.Process(target=extendTimodelsave,args=(Tizuobiaox,c,c,c,2))
    p3 = multiprocessing.Process(target=extendTimodelsave,args=(Tizuobiaoy,c,c,c,3))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    
    Tizuobiao=[]#坐标矩阵
    Tizuobiaox=np.load('./output/ext2.npy')
    Tizuobiaoy=np.load('./output/ext3.npy')
    for h in range(Tizuobiaox.shape[0]):
        for x in range(Tizuobiaox.shape[1]):
            for y in range(Tizuobiaox.shape[2]):
                sodoi=np.array([Tizuobiaoh[h,x,y],Tizuobiaox[h,x,y],Tizuobiaoy[h,x,y]])
                Tizuobiao.append(sodoi)
    #Ti=extendTimodel(fow,c,c,c)
    Ti=np.load('./output/ext1.npy')
    return Ti,Tizuobiao

def Recodesectionloadandextend(m,template_x,template_y,flag,scale):#flag==1为patchmatch步骤，0为initial步骤
    #对剖面进行导入和Ti提取的函数  #scale为当前倍率
    Tilist=[]
    Tizuobiaolist=[]
    file1=open('./Ti/Tiparameter.txt')
    content=file1.readline()
    string1=[i for i in content if str.isdigit(i)]
    num=int(''.join(string1))
    print('剖面数目：')
    print (num)
    for n in range(num):
        guding=[]
        for aa in range(4):
            content=file1.readline()
            string1=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string1))
            guding.append(xx)
        path='./Ti/'+str(n+1)+'.bmp'
        section=cv2.imread(path,0)
        #print guding[0],guding[1],guding[2],guding[3]
        sectionload_x(m,section,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)#载入剖面
        if flag==1:
            Ti,Tizuobiao=RecodeTIextendforEM(section,m,template_x,template_y,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)
            Tilist.append(Ti)
            Tizuobiaolist.append(Tizuobiao)
    #都执行完后可进行gosiminitialAI
    return m,Tilist,Tizuobiaolist




def transhxy(h0,x0,y0,lag):#坐标转换
    hz=int(h0/(lag+1))
    xz=int(x0/(lag+1))
    yz=int(y0/(lag+1))
    return hz,xz,yz


def Recodeinitialization(A,B,Bzuobiao,p_size,lag):#初始化，构建映射 flag为是否超过计算规格，lag为超出计算规格时所用的空洞步长
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    p = p_size // 2
    #B_padding = np.ones([B_h+p*2, B_l+p*2,B_w+p*2]) * np.nan
    #B_padding = -np.ones([B_h+p*2, B_l+p*2,B_w+p*2])
    #B_padding[p:B_h+p, p:B_l+p,p:B_w+p] = B
    B=np.pad(B,p,'edge')
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    random_B_r = np.random.randint(p, B_h-p, [A_h, A_l,A_w])#分别储存对应B的坐标
    #print random_B_r
    random_B_c = np.random.randint(p, B_l-p, [A_h, A_l,A_w])
    random_B_v = np.random.randint(p, B_w-p, [A_h, A_l,A_w])
    #A_padding = np.ones([A_h+p*2, A_l+p*2, 1]) * np.nan
    #A_padding[p:A_h+p, p:A_l+p, :] = A
    #A_padding = np.ones([A_h+p*2, A_l+p*2,A_w+p*2]) * np.nan
    A_padding = -np.ones([A_h+p*2, A_l+p*2,A_w+p*2])
    A_padding[p:A_h+p, p:A_l+p,p:A_w+p] = A
    f = np.zeros([A_h//(lag+1), A_l//(lag+1),A_w//(lag+1)], dtype=object)
    dist = np.zeros([A_h//(lag+1), A_l//(lag+1),A_w//(lag+1)])
        
        
    for i in range(A_h//(lag+1)):
        for j in range(A_l//(lag+1)):
            for k in range(A_w//(lag+1)):
                a = np.array([i+p, j+p,k+p])
                b = np.array([random_B_r[i, j,k], random_B_c[i, j,k],random_B_v[i, j,k]], dtype=np.int32)
                #print b
                f[i, j,k] = b
                dist[i, j,k] = cal_distance(a, b, A_padding, B, p_size)
        
    number=0
    for i in range(B_h-2*p):
        for j in range(B_l-2*p):
            for k in range(B_w-2*p):
                #print(Bzuobiao[number],np.array([i+p,j+p,k+p]))
                first=Bzuobiao[number]//(lag+1)
                f[first[0],first[1],first[2]]=np.array([i+p,j+p,k+p])
                number=number+1
                dist[first[0],first[1],first[2]]=0
        
    return f, dist, A_padding,B

def Recodepropagationsub(f, a, dist, A_padding, B, p_size, is_odd,lag):#传播,a为待传播点的坐标

    A_h = f.shape[0]
    A_l = f.shape[1]
    A_w = f.shape[2]
    p=p_size//2
    x = int((a[0]-p)/(lag+1))
    y = int((a[1]-p)/(lag+1))
    z = int((a[2]-p)/(lag+1))
        
    if is_odd:
       d_left = dist[max(x-1, 0), y,z]
       d_up = dist[x, max(y-1, 0),z]
       d_forward=dist[x,y,max(z-1, 0)]
       d_current = dist[x, y,z]
       idx = np.argmin(np.array([d_current, d_left, d_up,d_forward]))# 可以多加几个传播值
       if idx == 1:
          f[x, y,z] = f[max(x - 1, 0), y,z]
          dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
       if idx == 2:
          f[x, y,z] = f[x, max(y - 1, 0),z]
          dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
       if idx == 3:
          f[x, y,z] = f[x, y,max(z - 1, 0)]
          dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
    else:
       d_right = dist[min(x + 1, A_h-1), y,z]
       d_down = dist[x, min(y + 1, A_l-1),z]
       d_back = dist[x, y,min(z + 1, A_w-1)]
       d_current = dist[x, y,z]
       idx = np.argmin(np.array([d_current, d_right, d_down,d_back]))
       if idx == 1:
          f[x, y,z] = f[min(x + 1, A_h-1), y,z]
          dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
       if idx == 2:
          f[x, y,z] = f[x, min(y + 1, A_l-1),z]
          dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
       if idx == 3:
          f[x, y,z] = f[x,y, min(z + 1, A_w-1)]
          dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)        
    return f,dist

def Recoderandom_searchsub(f, a, dist, A_padding, B, p_size,lag ,alpha=0.5):#随机搜索

    p = p_size // 2
    x = int((a[0]-p)/(lag+1))
    y = int((a[1]-p)/(lag+1))
    z = int((a[2]-p)/(lag+1))
        
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    
    i = 2
    search_h = B_h * alpha ** i
    search_l = B_l * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y,z][0]
    b_y = f[x, y,z][1]
    b_z = f[x, y,z][2]
    while search_h > 1 and search_l > 1 and search_w > 1:
        #print('ssssss')
        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h-p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        if (B_l>(3*p_size)) and (B_w>(3*p_size)):
            print('and')
            print(B_l,B_h)
            search_min_c = max(b_y - search_l, p)
            search_max_c = min(b_y + search_l, B_l - p)
            #print search_min_c, search_max_c
            random_b_y = np.random.randint(search_min_c, search_max_c)
            search_min_v = max(b_z - search_w, p)
            search_max_v = min(b_z + search_w, B_w - p)
            #print search_min_v, search_max_v
            random_b_z = np.random.randint(search_min_v, search_max_v) 
            earch_l = B_l * alpha ** i
            search_w = B_w * alpha ** i
        
        else:
            if B_l>=B_w:
               search_min_c = max(b_y - search_l, p)
               search_max_c = min(b_y + search_l, B_l - p)
               #print search_min_c, search_max_c
               random_b_y = np.random.randint(search_min_c, search_max_c)
               random_b_z = np.random.randint(p,B_w-1-p)
               search_l = B_l * alpha ** i
               search_w = B_w
            else:
               random_b_y = np.random.randint(p,B_l-1-p)
               search_min_v = max(b_z - search_w, p)
               search_max_v = min(b_z + search_w, B_w - p)
               #print search_min_v, search_max_v
               random_b_z = np.random.randint(search_min_v, search_max_v)
               search_l = B_l
               search_w = B_w * alpha ** i
       
        search_h = B_h * alpha ** i
        #search_l = B_l * alpha ** i
        #search_w = B_w * alpha ** i
        b = np.array([random_b_x, random_b_y,random_b_z])
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < dist[x, y,z]:
           dist[x, y,z] = d
           f[x, y,z] = b
        i += 1
    #print('stop')
    return f,dist






def RecodeNNSBsub(f,dist,img_padding,Bref,p_size,item,name,core,lag):
    for n in range(len(item)):
        f,dist=Recodepropagationsub(f, item[n], dist, img_padding, Bref, p_size, bool(random.getrandbits(1)),lag)
        f,dist=Recoderandom_searchsub(f, item[n], dist, img_padding, Bref, p_size,lag)
    path1='./database/patchmatchprocess('+str(name)+')('+str(core)+')f.npy'
    path2='./database/patchmatchprocess('+str(name)+')('+str(core)+')dist.npy'
    np.save(path1,f)
    np.save(path2,dist)
def RecodereNNSBsub(f,dist,items,name,core,p_size,lag):#重组
    p=p_size//2
    for n in range(len(items)):
        path1='./database/patchmatchprocess('+str(name)+')('+str(n)+')f.npy'
        path2='./database/patchmatchprocess('+str(name)+')('+str(n)+')dist.npy'
        fff=np.load(path1)
        distdistdist=np.load(path2)
        for xixi in range(len(items[n])):
            f[(items[n][xixi][0]-p)//(lag+1),(items[n][xixi][1]-p)//(lag+1),(items[n][xixi][2]-p)//(lag+1)]=fff[(items[n][xixi][0]-p)//(lag+1),(items[n][xixi][1]-p)//(lag+1),(items[n][xixi][2]-p)//(lag+1)]
            dist[(items[n][xixi][0]-p)//(lag+1),(items[n][xixi][1]-p)//(lag+1),(items[n][xixi][2]-p)//(lag+1)]=distdistdist[(items[n][xixi][0]-p)//(lag+1),(items[n][xixi][1]-p)//(lag+1),(items[n][xixi][2]-p)//(lag+1)]
    return f,dist

def RecodeNNSB(img, ref, refzuobiao,p_size, itr,name,core,lag):
    #寻找最近零并行版改进 name 为ti编号，core为并行模拟核数,lag为坐标与实际坐标间隔
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding ,Bref= Recodeinitialization(img,ref,refzuobiao, p_size,lag)
    p=p_size//2
    print("initialization done")
    zuobiaoarr=[]
    for h in range(0,A_h,lag+1):
        for x in range(0,A_l,lag+1):
            for y in range(0,A_w,lag+1):
                zuobiaoarr.append(np.array([h+p,x+p,y+p]))
    items=apartlist(zuobiaoarr, int(len(zuobiaoarr)/core))
    #内嵌式多进程
    for itr in range(1,itr+1):
        if itr % 2 == 0:
            processes=list()
            print(len(items))
            for n in range(len(items)):
                s=multiprocessing.Process(target=RecodeNNSBsub, args=(f, dist, img_padding, Bref, p_size,items[n],name,n,lag)) 
                print('process:',n)
                s.start()
                processes.append(s)
            for s in processes:
                s.join()
            f,dist=RecodereNNSBsub(f,dist,items,name,core,p_size,lag)
        else:
            processes=list()
            for n in range(len(items)):
                newList = list(reversed(items[n]))#倒转列表顺序
                s=multiprocessing.Process(target=RecodeNNSBsub, args=(f, dist, img_padding, Bref, p_size,newList,name,n,lag)) 
                print('process:',n)
                s.start()
                processes.append(s)
            for s in processes:
                s.join()
            f,dist=RecodereNNSBsub(f,dist,items,name,core,p_size,lag)
        print("iteration: %d"%(itr))
    return f,dist,Bref

def RecodeprojectTTT(m,F,DIST,lag):
    Fore=np.zeros((m.shape[0]//(lag+1), m.shape[1]//(lag+1),m.shape[2]//(lag+1)),int)
    sw=999999
    for i in range(Fore.shape[0]):
        for j in range(Fore.shape[1]):
            for k in range(Fore.shape[2]):
                for n in range(len(F)):
                    if DIST[n][i,j,k]<=sw:
                       Thor=n
                       sw=DIST[n][i,j,k]

                Fore[i,j,k]=Thor
                sw=999999
    return Fore
                                
def RecodereconstructionTTT(A,mm,F,Fore,BTilist,p_size,lag):
    p=p_size//2
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = -np.ones([A_h+p*2, A_l+p*2,A_w+p*2])
    CTilist=[]#构建模板使用统计TI
    for n in range(len(BTilist)):
        ti = np.zeros_like(BTilist[n])
        CTilist.append(ti)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                for n in range(len(F)):
                    if mm[i,j,k]==-1:
                       if Fore[i//(lag+1),j//(lag+1),k//(lag+1)]==n:
                           #temp[i,j,k] = BTilist[n][F[n][i/(lag+1), j/(lag+1),k/(lag+1)][0], F[n][i/(lag+1), j/(lag+1),k/(lag+1)][1],F[n][i/(lag+1), j/(lag+1),k/(lag+1)][2]]
                           Retem=template1(BTilist[n],2*lag+1,2*lag+1,2*lag+1,F[n][i//(lag+1), j//(lag+1),k//(lag+1)][0], F[n][i//(lag+1), j//(lag+1),k//(lag+1)][1],F[n][i//(lag+1), j//(lag+1),k//(lag+1)][2])
                           temp=template1RAI(temp,Retem,i+p,j+p,k+p)
                           CTilist[n][F[n][i//(lag+1), j//(lag+1),k//(lag+1)][0], F[n][i//(lag+1), j//(lag+1),k//(lag+1)][1],F[n][i//(lag+1), j//(lag+1),k//(lag+1)][2]]=CTilist[n][F[n][i//(lag+1), j//(lag+1),k//(lag+1)][0], F[n][i//(lag+1), j//(lag+1),k//(lag+1)][1],F[n][i//(lag+1), j//(lag+1),k//(lag+1)][2]]+1
                    else:
                       temp[i+p,j+p,k+p]=mm[i,j,k]
    #Image.fromarray(temp).show()
    Re=temp[p:A_h+p, p:A_l+p,p:A_w+p]
    return Re,CTilist
                                   
def Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,core,lag):#patchmatch重新优化版，core为并行ver,flag为坐标与实际坐标间隔
    #size为模板大小，itr为迭代次数
    #增加新加速模式，首先提取列表，按照一定重叠区选择待模拟点
    #重构过程根据多源融合结果整块复制
    
    Fore = np.zeros([m.shape[0], m.shape[1],m.shape[2]])
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    F=[]
    DIST=[]
    BTilist=[]
    processes=list()
    for n in range(len(Tilist)):
        print(Tilist[n].shape)
    for n in range(len(Tilist)):
        print(Tilist[n].shape)
        f,dist,Bref=RecodeNNSB(m,Tilist[n],Tizuobiaolist[n],size,itr,n,core,lag)
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)
    print("Searching done!")
    #print F[0],F[1],F[2],F[3]
    Fore=RecodeprojectTTT(m,F,DIST,lag)
    #print Fore
    print("Pick done!")
    Re,CTilist=RecodereconstructionTTT(m,mm,F,Fore,BTilist,size,lag)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re,CTilist




























########################################################柱状patchmatch
def extend2d2d(m,x1,y1):#9格子内随机选取一个值 
    listcs=[]
    for ss2 in range(-1,2):
        for ss3 in range(-1,2):
            c=m[x1+ss2,y1+ss3]
            if c!=-1:#默认空值为-1
                listcs.append(c)

    if len(listcs)>=2:
    #if len(listcs)!=0:
        value= max_list(listcs)
    else:
        value=-1
    return value
def extendTimodelsave2d(m,template_x,template_y,ids):#全自动拓展插入硬数据的待模拟网格
    lag=max(template_h,template_x,template_y)//2
    m2=np.pad(m,lag,'edge')
    d=[]

    for x in range(lag,m2.shape[0]-lag):
        for y in range(lag,m2.shape[1]-lag):
            d.append((x,y))
    
    for cc in range(lag):
        #random.shuffle(d)
        flag=0
        for n in range(len(d)):
            x=d[n][0]
            y=d[n][1]

            if m2[x,y]==-1:
                value=extend2d2d(m2,x,y)
                flag=1
                if value!=-1:
                    #print value
                    m[h-lag,x-lag,y-lag]=value
                
                else:
                    if cc==lag-1:
                        m[x-lag,y-lag]=value
        if flag==0:
            break
        m2=np.pad(m,lag,'edge')
        #填充为1的 
    path='./output/ext'+str(ids)+'.npy'
    np.save(path,m)
    return m


def Recode2TIextendforEM(section,m,template_x,template_y,x1,y1,x2,y2):#EM迭代用剖面提取，(x,y）为剖面定位的一组坐标
    #m为已经完成扩展的模拟网格 #柱状版
    dx=[]
    dy=[]
    
    lag=max(template_x,template_y)//2
    ms=-np.ones((m.shape[0],m.shape[1],m.shape[2]),int)
    sectionload_x(ms,section,x1,y1,x2,y2)#独立载入防止造成多剖面混乱
    Tizuobiaox=-np.ones((m.shape[1],m.shape[2]), int)
    Tizuobiaoy=-np.ones((m.shape[1],m.shape[2]), int)
    
    if abs(x1-x2)>=lag:
        for n1 in range(min(x1,x2),max(x1,x2)+1):
            dx.append(n1)
    else:
        for n1 in range(max(0,min(x1,x2)-lag),min(max(x1,x2)+lag,m.shape[1]-1)+1):
            dx.append(n1)
            
    if abs(y1-y2)>=lag:
        for n1 in range(min(y1,y2),max(y1,y2)+1):
            dy.append(n1)
    else:
        for n1 in range(max(0,min(y1,y2)-lag),min(max(y1,y2)+lag,m.shape[2]-1)+1):
            dy.append(n1)
    
    for h in range(ms.shape[0]):
        for x in range(ms.shape[1]):
            for y in range(ms.shape[2]):
                if ms[h,x,y]!=-1:
                    Tizuobiaox[x,y]=x
                    Tizuobiaoy[x,y]=y
    temp=ms[:,dx,:]
    fow=temp[:,:,dy]
    Tizuobiaoxt=Tizuobiaox[dx,:]
    Tizuobiaox=Tizuobiaoxt[:,dy]
    Tizuobiaoyt=Tizuobiaoy[dx,:]
    Tizuobiaoy=Tizuobiaoyt[:,dy]
    c=max(fow.shape[1],fow.shape[2])
   

    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=extendTimodelsave,args=(fow,c,c,c,1))
    
    p2 = multiprocessing.Process(target=extendTimodelsave2d,args=(Tizuobiaox,c,c,2))
    p3 = multiprocessing.Process(target=extendTimodelsave2d,args=(Tizuobiaoy,c,c,3))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    
    Tizuobiao=[]#坐标矩阵
    Tizuobiaox=np.load('./output/ext2.npy')
    Tizuobiaoy=np.load('./output/ext3.npy')

    for x in range(Tizuobiaox.shape[0]):
        for y in range(Tizuobiaox.shape[1]):
            sodoi=np.array([Tizuobiaox[x,y],Tizuobiaoy[x,y]])
            Tizuobiao.append(sodoi)
    #Ti=extendTimodel(fow,c,c,c)
    Ti=np.load('./output/ext1.npy')
    return Ti,Tizuobiao




def Recode2sectionloadandextend(m,template_x,template_y,flag,scale):#flag==1为patchmatch步骤，0为initial步骤
    #对剖面进行导入和Ti提取的函数  #scale为当前倍率
    Tilist=[]
    Tizuobiaolist=[]
    file1=open('./Ti/Tiparameter.txt')
    content=file1.readline()
    string1=[i for i in content if str.isdigit(i)]
    num=int(''.join(string1))
    print('剖面数目：')
    print (num)
    for n in range(num):
        guding=[]
        for aa in range(4):
            content=file1.readline()
            string1=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string1))
            guding.append(xx)
        path='./Ti/'+str(n+1)+'.bmp'
        section=cv2.imread(path,0)
        #print guding[0],guding[1],guding[2],guding[3]
        sectionload_x(m,section,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)#载入剖面
        if flag==1:
            Ti,Tizuobiao=Recode2TIextendforEM(section,m,template_x,template_y,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)
            Tilist.append(Ti)
            Tizuobiaolist.append(Tizuobiao)
    #都执行完后可进行gosiminitialAI
    return m,Tilist,Tizuobiaolist






def Recode2initialization(A,B,Bzuobiao,p_size,lag):#初始化，构建映射 flag为是否超过计算规格，lag为超出计算规格时所用的空洞步长
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    p = p_size // 2

    
    B=np.pad(B,((0,0),(p,p),(p,p)),'edge')
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)

    
    random_B_c = np.random.randint(p, B_l-p, [ A_l,A_w])
    random_B_v = np.random.randint(p, B_w-p, [ A_l,A_w])

    
    A_padding = -np.ones([A_h, A_l+p*2,A_w+p*2])
    A_padding[:, p:A_l+p,p:A_w+p] = A
    
    f = np.zeros([ A_l//(lag+1),A_w//(lag+1)], dtype=object)
    dist = np.zeros([ A_l//(lag+1),A_w//(lag+1)])
        
        

    for j in range(A_l//(lag+1)):
        for k in range(A_w//(lag+1)):
            a = np.array([j+p,k+p])
            b = np.array([random_B_c[j,k],random_B_v[j,k]], dtype=np.int32)
            #print b
            f[j,k] = b
            dist[j,k] = cal_distanceZ(a, b, A_padding, B, p_size)
        
    number=0

    for j in range(B_l-2*p):
        for k in range(B_w-2*p):
            #print(Bzuobiao[number],np.array([i+p,j+p,k+p]))
            first=Bzuobiao[number]//(lag+1)
            f[first[0],first[1]]=np.array([j+p,k+p])
            number=number+1
            dist[first[0],first[1]]=0
        
    return f, dist, A_padding,B

def Recode2propagationsub(f, a, dist, A_padding, B, p_size, is_odd,lag):#传播,a为待传播点的坐标

    A_h = f.shape[0]
    A_l = f.shape[1]
    A_w = f.shape[2]
    p=p_size//2
    x = int((a[0]-p)/(lag+1))
    y = int((a[1]-p)/(lag+1))
        
    if is_odd:
        d_left = dist[max(x-1, 0), y]
        d_up = dist[x, max(y-1, 0)]

        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))# 可以多加几个传播值
        if idx == 1:
            f[x,y] = f[max(x - 1, 0), y]
            dist[x, y] = cal_distanceZ(a, f[x, y], A_padding, B, p_size)
        if idx == 1:
            f[x, y] = f[x, max(y - 1, 0)]
            dist[x,y] = cal_distanceZ(a, f[x, y], A_padding, B, p_size)

    else:
        d_right = dist[min(x + 1, A_l-1), y]
        d_down = dist[x, min(y + 1, A_w-1)]
        d_current = dist[x, y,z]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y] = f[min(x + 1, A_l-1), y]
            dist[x, y] = cal_distanceZ(a, f[x, y], A_padding, B, p_size)
        if idx == 2:
            f[x, y] = f[x, min(y + 1, A_w-1)]
            dist[x, y] = cal_distanceZ(a, f[x, y], A_padding, B, p_size)
        
    return f,dist

def Recode2random_searchsub(f, a, dist, A_padding, B, p_size,lag ,alpha=0.5):#随机搜索

    p = p_size // 2
    x = int((a[0]-p)/(lag+1))
    y = int((a[1]-p)/(lag+1))

        
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    
    i = 2

    search_l = B_l * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y][0]
    b_y = f[x, y][1]

    while  search_l > 1 and search_w > 1:
        #print('ssssss')
        
        if (B_l>(3*p_size)) and (B_w>(3*p_size)):
            search_min_r = max(b_x - search_l, p)
            search_max_r = min(b_x + search_l, B_l-p)
            random_b_x = np.random.randint(search_min_r, search_max_r)
            #print(B_l,B_w)
            search_min_c = max(b_y - search_w, p)
            search_max_c = min(b_y + search_w, B_w - p)
            #print search_min_c, search_max_c
            random_b_y = np.random.randint(search_min_c, search_max_c)

             
            search_l = B_l * alpha ** i
            search_w = B_w * alpha ** i
        
        else:
            if B_l>=B_w:
                search_min_r = max(b_x - search_l, p)
                search_max_r = min(b_x + search_l, B_l-p)
                random_b_x = np.random.randint(search_min_r, search_max_r)
                #print(B_l,B_w)
                random_b_y = np.random.randint(p,B_w-p)
                search_l = B_l * alpha ** i
                search_w = B_w
                
            else:
                search_min_c = max(b_y - search_w, p)
                search_max_c = min(b_y + search_w, B_w-p)
                random_b_y = np.random.randint(search_min_c, search_max_c)
                #print(B_l,B_w)
                random_b_x = np.random.randint(p,B_l-p)
                search_l = B_l
                search_w = B_w * alpha ** i
       

        b = np.array([random_b_x, random_b_y])
        d = cal_distanceZ(a, b, A_padding, B, p_size)
        if d < dist[x, y]:
            dist[x, y] = d
            f[x, y] = b
        i += 1
    #print('stop')
    return f,dist






def Recode2NNSBsub(f,dist,img_padding,Bref,p_size,item,name,core,lag):
    for n in range(len(item)):
        f,dist=Recode2propagationsub(f, item[n], dist, img_padding, Bref, p_size, bool(random.getrandbits(1)),lag)
        f,dist=Recode2random_searchsub(f, item[n], dist, img_padding, Bref, p_size,lag)
    path1='./database/patchmatchprocess('+str(name)+')('+str(core)+')f.npy'
    path2='./database/patchmatchprocess('+str(name)+')('+str(core)+')dist.npy'
    np.save(path1,f)
    np.save(path2,dist)
def Recode2reNNSBsub(f,dist,items,name,core,p_size,lag):#重组
    p=p_size//2
    for n in range(len(items)):
        path1='./database/patchmatchprocess('+str(name)+')('+str(n)+')f.npy'
        path2='./database/patchmatchprocess('+str(name)+')('+str(n)+')dist.npy'
        fff=np.load(path1)
        distdistdist=np.load(path2)
        for xixi in range(len(items[n])):
            f[(items[n][xixi][0]-p)//(lag+1),(items[n][xixi][1]-p)//(lag+1)]=fff[(items[n][xixi][0]-p)//(lag+1),(items[n][xixi][1]-p)//(lag+1)]
            dist[(items[n][xixi][0]-p)//(lag+1),(items[n][xixi][1]-p)//(lag+1)]=distdistdist[(items[n][xixi][0]-p)//(lag+1),(items[n][xixi][1]-p)//(lag+1)]
    return f,dist

def Recode2NNSB(img, ref, refzuobiao,p_size, itr,name,core,lag):
    #寻找最近零并行版改进 name 为ti编号，core为并行模拟核数,lag为坐标与实际坐标间隔
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding ,Bref= Recode2initialization(img,ref,refzuobiao, p_size,lag)
    p=p_size//2
    print("initialization done")
    zuobiaoarr=[]

    for x in range(0,A_l,lag+1):
        for y in range(0,A_w,lag+1):
            zuobiaoarr.append(np.array([x+p,y+p]))
    items=apartlist(zuobiaoarr, int(len(zuobiaoarr)/core))
    #内嵌式多进程
    for itr in range(1,itr+1):
        if itr % 2 == 0:
            processes=list()
            print(len(items))
            for n in range(len(items)):
                s=multiprocessing.Process(target=Recode2NNSBsub, args=(f, dist, img_padding, Bref, p_size,items[n],name,n,lag)) 
                print('process:',n)
                s.start()
                processes.append(s)
            for s in processes:
                s.join()
            f,dist=Recode2reNNSBsub(f,dist,items,name,core,p_size,lag)
        else:
            processes=list()
            for n in range(len(items)):
                newList = list(reversed(items[n]))#倒转列表顺序
                s=multiprocessing.Process(target=Recode2NNSBsub, args=(f, dist, img_padding, Bref, p_size,newList,name,n,lag)) 
                print('process:',n)
                s.start()
                processes.append(s)
            for s in processes:
                s.join()
            f,dist=Recode2reNNSBsub(f,dist,items,name,core,p_size,lag)
        print("iteration: %d"%(itr))
    return f,dist,Bref

def Recode2projectTTT(m,F,DIST,lag):
    Fore=np.zeros(( m.shape[1]//(lag+1),m.shape[2]//(lag+1)),int)
    sw=999999

    for j in range(Fore.shape[1]):
        for k in range(Fore.shape[2]):
            for n in range(len(F)):
                if DIST[n][j,k]<=sw:
                    Thor=n
                    sw=DIST[n][j,k]

            Fore[j,k]=Thor
            sw=999999
    return Fore
                                
def Recode2reconstructionTTT(A,mm,F,Fore,BTilist,p_size,lag):
    p=p_size//2
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = -np.ones([A_h, A_l+p*2,A_w+p*2])
    '''
    CTilist=[]#构建模板使用统计TI
    for n in range(len(BTilist)):
        ti = np.zeros_like(BTilist[n])
        CTilist.append(ti)
    '''
    for j in range(A_l):
        for k in range(A_w):
            for n in range(len(F)):
                if mm[0,j,k]==-1:
                    if Fore[j//(lag+1),k//(lag+1)]==n:
                        #Retem=template1(BTilist[n],2*lag+1,2*lag+1,2*lag+1,F[n][i//(lag+1), j//(lag+1),k//(lag+1)][0], F[n][i//(lag+1), j//(lag+1),k//(lag+1)][1],F[n][i//(lag+1), j//(lag+1),k//(lag+1)][2])
                        Retem=template2(BTilist[n],2*lag+1,2*lag+1,F[n][ j//(lag+1),k//(lag+1)][0], F[n][ j//(lag+1),k//(lag+1)][1])
                        temp=template2RAI(temp,Retem,j+p,k+p)
                        #CTilist[n][F[n][i//(lag+1), j//(lag+1),k//(lag+1)][0], F[n][i//(lag+1), j//(lag+1),k//(lag+1)][1],F[n][i//(lag+1), j//(lag+1),k//(lag+1)][2]]=CTilist[n][F[n][i//(lag+1), j//(lag+1),k//(lag+1)][0], F[n][i//(lag+1), j//(lag+1),k//(lag+1)][1],F[n][i//(lag+1), j//(lag+1),k//(lag+1)][2]]+1
                else:
                    temp[:,j+p,k+p]=mm[:,j,k]
    #Image.fromarray(temp).show()
    Re=temp[:, p:A_l+p,p:A_w+p]
    return Re,CTilist
                                   
def Recode2patchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,core,lag):#patchmatch重新优化版，core为并行ver,flag为坐标与实际坐标间隔
    #size为模板大小，itr为迭代次数
    #增加新加速模式，首先提取列表，按照一定重叠区选择待模拟点
    #重构过程根据多源融合结果整块复制
    
    Fore = np.zeros([m.shape[0], m.shape[1],m.shape[2]])
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    F=[]
    DIST=[]
    BTilist=[]
    processes=list()
    for n in range(len(Tilist)):
        print(Tilist[n].shape)
    for n in range(len(Tilist)):
        print(Tilist[n].shape)
        f,dist,Bref=Recode2NNSB(m,Tilist[n],Tizuobiaolist[n],size,itr,n,core,lag)
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)
    print("Searching done!")
    #print F[0],F[1],F[2],F[3]
    Fore=RecodeprojectTTT(m,F,DIST,lag)
    #print Fore
    print("Pick done!")
    Re=Recode2reconstructionTTT(m,mm,F,Fore,BTilist,size,lag)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re








