#!/usr/bin/env python
# coding: utf-8

# In[24]:


######################ver1.4 柱状初始化，保证垂直层序
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
from AIinitial import*
from PatchmatchZ import*
####################################################

def lujinglistAIP(m,template_x,template_y):#将所有待模拟网格中空值1的待模拟点加入模拟路径
    roadlist=[]
    #lagh=template_h//2+1
    #lagx=template_x//2+1
    #lagy=template_y//2+1

    for x in range(m.shape[1]):
        for y in range(m.shape[2]):
            if m[0,x,y]==-1:
               roadlist.append((x,y))
    return roadlist

def cutP(m,L):#裁剪工具for pillar
    i1=0
    is1=0
    j1=0
    h=m.shape[0]
    a=m.shape[1]
    b=m.shape[2]
    data=np.zeros([h,a-2*L,b-2*L],int)
    for s in range(h):
        for i in range(L,a-L): 
            for j in range(L,b-L):
                data[s,i1,j1]=m[s,i,j]
                j1=j1+1
            i1=i1+1
            j1=0

        i1=0
        j1=0
    m=data
    return m



def databasebuildAIP(Exm,template_x,template_y):#智能构建模式库
    #Exm为已经完成了拓展的模拟网格
    lag=max(template_x,template_y)
    Exm2=np.pad(Exm,((0,0),(lag,lag),(lag,lag)),'edge')#拓展
    database=[]
    zuobiaolist=[]
    for x in range(Exm.shape[1]):
        for y in range(Exm.shape[2]):
            if Exm[0,x,y]!=-1:
               x0=x+lag
               y0=y+lag
               tem=template2(Exm2,template_x,template_y,x0,y0)
               if temdetect(tem):#如果不包含待模拟点则为模板
                  database.append(tem)
                  zuobiaolist.append((x,y))
    return database,zuobiaolist


def databasecataAIP(database,lag):#按照重叠区分类，lag为重叠区,四向面提取
    template_h=database[0].shape[0]
    template_x=database[0].shape[1]
    template_y=database[0].shape[2]
    le=len(database)
    dis=[]#后左
    disx=[]#前
    disy=[]#右


    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)

    cdatabase=[] 
    
    #左
    d1=[]
    for s in range(le):#遍历模式库
        #数据库0
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)
    
    #右
    d1=[]
    for s in range(le):#遍历模式库
        #数据库1
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)
    
    #后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库2
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)  
    
    #前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库3
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)
    
    

  
    
    cdatabase.append(database)#数据库4为本体
    print('done')
    return cdatabase

def databaseclusterAIP(cdatabase,U):#模式分类
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

    


    

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    
    
    print('process end')

    Cdatabase=[]
    cc1=np.load('./database/clusters0.npy')
    cc2=np.load('./database/clusters1.npy')
    cc3=np.load('./database/clusters2.npy')
    cc4=np.load('./database/clusters3.npy')


    Cdatabase.append(cc1)
    Cdatabase.append(cc2)
    Cdatabase.append(cc3)
    Cdatabase.append(cc4)


    
    
    np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase




################################################################
################################################################
################################################################
def initialroadlistAIP(m,template_x,template_y,lag):
    #自动初始化网格系统
    lujing=[]
    Roadlist=[]#最终路径名单
    lujing=lujinglistAIP(m,template_x,template_y)
    random.shuffle(lujing)
    #print len(lujing)
    
    #print len(lujing)
    m2=np.pad(m,((0,0),(lag,lag),(lag,lag)),'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]
    Fin=Fin*1000#最大循环次数

    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    Banlujing=[]
    ################重叠区选取器#####################
    ss=H*template_x*template_y
    dis=[]#后左
    disx=[]#前
    disy=[]#右

    b=np.zeros((H,template_x,template_y),int)
    
    d=[]#待候选列表

    cc=0#对比序号
    
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)

        
        
    #############################################
    
    
    while n<len(lujing):
        if m2[0,lujing[n][0]+lag,lujing[n][1]+lag]==-1:
            #if lujing[n] not in Banlujing:

            x1=lujing[n][0]+lag
            y1=lujing[n][1]+lag
            o1=template2(m2,template_x,template_y,x1,y1)
            k=0#重叠区计数器

            if o1[0,0,template_y/2]!=-1:
                #后
                k=k+1
            if o1[0,template_x-1,template_y/2]!=-1:
                #前 
                k=k+1
            if o1[0,template_x/2,0]!=-1:
                #左
                k=k+1
            if o1[0,template_x/2,template_y-1]!=-1:
                #右
                k=k+1
            if k!=0:
                m2=template2R(m2,b,x1,y1)
                Roadlist.append((x1-lag,y1-lag))
            else:
                lujing.append(lujing[n])
        #print len(Roadlist),n,len(lujing)
        n=n+1
        #print len(Roadlist)
    return Roadlist


################################################################
################################################################
################################################################



def initialgridAIP(m,template_x,template_y,lag,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIP(m,template_x,template_y,lag)

    #print len(lujing)
    m2=np.pad(m,((0,0),(lag,lag),(lag,lag)),'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]*10#最大循环次数

    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    ################重叠区选取器#####################
    ss=H*template_x*template_y
    dis=[]#后左
    disx=[]#前
    disy=[]#右

    b=np.zeros((H,template_x,template_y),int)
    
    d=[]#待候选列表
    c1=99999#对比数
    cc=0#对比序号
    
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)

        
    #############################################
    
    
    for n in range(len(lujing)):
        x1=lujing[n][0]+lag
        y1=lujing[n][1]+lag
        o1=template2(m2,template_x,template_y,x1,y1)
        canpatternlist0=[]
        canpatternlist1=[]
        canpatternlist2=[]
        canpatternlist3=[]
        c=np.zeros((H,template_x,template_y),int)

        
        if o1[0,0,template_y/2]!=-1:
            #后
            b=np.zeros((H,template_x,template_y),int)
            b[:,dis,:]=1
            c[:,dis,:]=1
            temo=o1*b
            canpatternlist0=patternsearchDi(Cdatabase[2],cdatabase[2],temo)



        if o1[0,template_x-1,template_y/2]!=-1:
            #前
            b=np.zeros((H,template_x,template_y),int)
            b[:,disx,:]=1
            c[:,disx,:]=1
            temo=o1*b
            canpatternlist1=patternsearchDi(Cdatabase[3],cdatabase[3],temo)

 
        if o1[0,template_x/2,0]!=-1:
            #左
            b=np.zeros((H,template_x,template_y),int)
            b[:,:,dis]=1
            c[:,:,dis]=1
            temo=o1*b
            canpatternlist2=patternsearchDi(Cdatabase[0],cdatabase[0],temo)


        if o1[0,template_x/2,template_y-1]!=-1:
            #右
            b=np.zeros((H,template_x,template_y),int)
            b[:,:,disy]=1
            c[:,:,disy]=1
            temo=o1*b
            canpatternlist3=patternsearchDi(Cdatabase[1],cdatabase[1],temo)


                
                
                

        canpatternlist=[]
        canpatternlist=list(set(canpatternlist0).union(set(canpatternlist1)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))

        canpatternlist=list(set(canpatternlist))
        #print len(canpatternlist),canpatternlist


        temo=o1*c
        tem=patternsearchAI2(temo,c,cdatabase[4],canpatternlist,N)
        m2=template2RAI(m2,tem,x1,y1)
            
        
    m=cutP(m2,lag)
    return m

def gosiminitialAIP(m,template_h,template_x,template_y,lag,lag_x,lag_y,N,U):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格
    #m为已经导入了Ti的模拟网格
    time_start1=time.time()
    m=extendTimodel(m,template_h,template_x,template_y)#拓展模拟网格


    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial1.vtk') 
    print('extend done')

 
    my_file = Path("./database/Cdatabase.npy")
    if my_file.exists():
        Cdatabase=np.load('./database/Cdatabase.npy')
        cdatabase=np.load('./database/cdatabase.npy')
        database=np.load('./database/database.npy')
        zuobiaolist=np.load('./database/zuobiaolist.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        database,zuobiaolist=databasebuildAIP(m,template_x,template_y)#数据库构建
        np.save('./database/database.npy',database)
        np.save('./database/zuobiaolist.npy',zuobiaolist)
        cdatabase=databasecataAIP(database,lag)
        np.save('./database/cdatabase.npy',cdatabase)
        Cdatabase=databaseclusterAIP(cdatabase,U)
        np.save('./database/Cdatabase.npy',Cdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    m=initialgridAIP(m,template_x,template_y,lag,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N)
    
    time_end=time.time()
    
    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial2.vtk') 
    
    print('initial done')
    print('timecost:')
    print(time_end-time_start)
    #初始化
    return m
















def GosimAIP(m,patternSizeh,patternSizex,patternSizey,lag,lag_x,lag_y,N,U,scale,size,itr):
    time_starts=time.time()#计时开始
    #初始模型构建 m为已经构建好的模拟网格
    m,Tilist,zuobiaolist=sectionloadandextend(m,patternSizex,patternSizey,0,1)
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIP(m,patternSizeh,patternSizex,patternSizey,lag,lag_x,lag_y,N,U)
    print('initial done')


    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)
        patternSix=int(patternSizex*scale[ni])
        patternSiy=int(patternSizey*scale[ni])
        patternSih=int(patternSizeh*scale[ni])
        #TI转换为当前尺度
        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=sectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)
        mm=extendTimodel(mm,patternSizeh,patternSizex,patternSizey)#mm亦为判断硬数据矩阵
        #上一个尺度升采样
        m=simgridex(m,scale[ni])
        #重新导入
        for hi in range(m.shape[0]):
            for xi in range(m.shape[1]):
                for yi in range(m.shape[2]):
                    if mm[hi,xi,yi]!=-1:
                        m[hi,xi,yi]=mm[hi,xi,yi]
        print("该尺度扩展Ti完成")
        time_start=time.time()#计时开始
        #np.save('./output/m1.npy',m)
        #m= patchmatchmultiTiB(m,Tilist,size,itr,name)
        m= patchmatchmultiTiBZ(m,mm,Tilist,size,itr,1)
        path="./output/reconstruction"+str(id)+".npy"
        np.save(path,m)
        time_end=time.time()
        #size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end-time_start)
    time_ends=time.time()
    print('总消耗时间:')
    print(time_ends-time_starts)
    print('Simulating done!')
    ##########################转换vtk格式########################

    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    path="./output/output"+str(id)+".vtk"
    write_data(grid, path) 

    print('并行程序结束')

