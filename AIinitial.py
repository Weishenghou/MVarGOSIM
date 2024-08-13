#!/usr/bin/env python
# coding: utf-8

# In[24]:


######################ver1.5
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
####################################################
def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str
def extend2dAI(m,h1,x1,y1):#9格子内随机选取一个值 
    listcs=[]
    for ss2 in range(-1,2):
        for ss3 in range(-1,2):
            c=m[h1,x1+ss2,y1+ss3]
            if c!=-1:#默认空值为-1
                listcs.append(c)

    if len(listcs)>=2:
    #if len(listcs)!=0:
        value= max_list(listcs)
    else:
        value=-1
    return value


'''
def extend2dAI(m,h1,x1,y1):#9格子内随机选取一个值 
    listcs=[]
    for ss2 in range(-1,2):
        for ss3 in range(-1,2):
            c=m[h1,x1+ss2,y1+ss3]
            if c!=1:#默认空值为1
                listcs.append(c)
    random.shuffle(listcs)
    if len(listcs)!=0:
        value=listcs[0]
    else:
        value=1
    return value
'''

#old version
def extendTimodel(m,template_h,template_x,template_y):#全自动拓展插入硬数据的待模拟网格
    lag=max(template_h,template_x,template_y)//2
    #lag=max(template_h,template_x,template_y)//2
    m2=np.pad(m,lag,'edge')
    d=[]
    for h in range(lag,m2.shape[0]-lag):
            for x in range(lag,m2.shape[1]-lag):
                for y in range(lag,m2.shape[2]-lag):
                    d.append((h,x,y))
    
    for cc in range(lag):
        #random.shuffle(d)
        flag=0
        for n in range(len(d)):
            h=d[n][0]
            x=d[n][1]
            y=d[n][2]
            if m2[h,x,y]==-1:
                value=extend2dAI(m2,h,x,y)
                flag=-1
                if value!=-1:
                    #print value
                    m[h-lag,x-lag,y-lag]=value

        if flag==0:
            break
        m2=np.pad(m,lag,'edge')
        #填充为1的    
    return m
'''

def extendTimodel(m,template_h,template_x,template_y):#全自动拓展插入硬数据的待模拟网格 new version
    roadlist=[]
    lag=template_h//2
    m2=np.pad(m,lag,'edge')
    for h in range(lag,m.shape[0]+lag):
        for x in range(lag,m.shape[1]+lag):
            for y in range(lag,m.shape[2]+lag):
                if m2[h,x,y]!=-1:
                    for cc1 in range(-lag,lag+1):
                        if (h+cc1>=lag) and ((h+cc1)<(m.shape[0]+lag)):
                            for cc2 in range(-lag,lag+1):
                                if (x+cc2>=lag) and ((x+cc2)<(m.shape[1]+lag)):
                                    for cc3 in range(-lag,lag+1):
                                        if (y+cc3>=lag) and ((y+cc3)<(m.shape[2]+lag)):
                                            if m2[h+cc1,x+cc2,y+cc3]==-1:
                                                roadlist.append([h+cc1,x+cc2,y+cc3])
    
    
    
    for cc in range(lag):
        #random.shuffle(d)
        flag=0
        for n in range(len(roadlist)):
            h=roadlist[n][0]
            x=roadlist[n][1]
            y=roadlist[n][2]
            if m2[h,x,y]==-1:
                value=extend2dAI(m2,h,x,y)
                flag=-1
                if value!=-1:
                    #print value
                    m[h-lag,x-lag,y-lag]=value
                
                #else:
                   # if cc==lag-1:
                        #m[h-lag,x-lag,y-lag]=value

        if flag==0:
            break
        m2=np.pad(m,lag,'edge')
        #填充为1的    
    return m

'''
def extendTimodelsave(m,template_h,template_x,template_y,ids):#全自动拓展插入硬数据的待模拟网格
    lag=max(template_h,template_x,template_y)//2
    m2=np.pad(m,lag,'edge')
    d=[]
    for h in range(lag,m2.shape[0]-lag):
            for x in range(lag,m2.shape[1]-lag):
                for y in range(lag,m2.shape[2]-lag):
                    d.append((h,x,y))
    
    for cc in range(lag):
        #random.shuffle(d)
        flag=0
        for n in range(len(d)):
            h=d[n][0]
            x=d[n][1]
            y=d[n][2]
            if m2[h,x,y]==-1:
                value=extend2dAI(m2,h,x,y)
                flag=1
                if value!=-1:
                    #print value
                    m[h-lag,x-lag,y-lag]=value
                
                else:
                    if cc==lag-1:
                        m[h-lag,x-lag,y-lag]=value
        if flag==0:
            break
        m2=np.pad(m,lag,'edge')
        #填充为1的 
    path='./output/ext'+str(ids)+'.npy'
    np.save(path,m)
    return m



def lujinglistAI(m,template_h,template_x,template_y):#将所有待模拟网格中空值1的待模拟点加入模拟路径
    roadlist=[]
    #lagh=template_h//2+1
    #lagx=template_x//2+1
    #lagy=template_y//2+1

    for h in range(m.shape[0]):
            for x in range(m.shape[1]):
                for y in range(m.shape[2]):
                    if m[h,x,y]==-1:
                        roadlist.append((h,x,y))
    return roadlist
'''
def cut(m,L):#裁剪工具
    i1=0
    is1=0
    j1=0
    h=m.shape[0]
    a=m.shape[1]
    b=m.shape[2]
    data=np.zeros([h-2*L,a-2*L,b-2*L],int)
    for s in range(L,h-L):
        for i in range(L,a-L): 
            for j in range(L,b-L):
                data[is1,i1,j1]=m[s,i,j]
                j1=j1+1
            i1=i1+1
            j1=0
        is1=is1+1
        i1=0
        j1=0
    m=data
    return m
'''
def cut(m,lag):
    return m[lag:m.shape[0]-lag,lag:m.shape[1]-lag,lag:m.shape[2]-lag]
def temdetect(tem):#检测是否包含待模拟点
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]==-1:
                    return False
    return True
def temdetectD(tem):#检测是否包含待模拟点是否大于阈值
    count=0
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]==-1:
                   count=count+1
    if count>=0.5*(tem.shape[0])*(tem.shape[1])*(tem.shape[2]):
       return False
    return True
def temdetect0(tem):#检测剖面是否为0
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]!=0:
                    return False
    return True

def temdetect0d(tem):#检测剖面是否为0,单剖面版
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            if tem[h,x]!=0:
                return False
    return True
def temdetect1(tem):#检测是否为待模拟重叠区
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]!=-1:
                    return False
    return True

def databasebuildAI(Exm,template_h,template_x,template_y):#智能构建模式库
    #Exm为已经完成了拓展的模拟网格
    lag=max(template_h,template_x,template_y)
    Exm2=np.pad(Exm,lag,'edge')#拓展
    database=[]
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
                        database.append(tem)
                        zuobiaolist.append((h,x,y))
    return database,zuobiaolist


def databasecataAI(database,lag):#按照重叠区分类，lag为重叠区,六向面提取
    template_h=database[0].shape[0]
    template_x=database[0].shape[1]
    template_y=database[0].shape[2]
    le=len(database)
    dis=[]#后左上
    disx=[]#前
    disy=[]#右
    dish=[]#下

    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
    cdatabase=[]
    #下
    d1=[]
    for s in range(le):#遍历模式库
        #数据库0
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1        
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1) 
    
    #左
    d1=[]
    for s in range(le):#遍历模式库
        #数据库1
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)
    
    #右
    d1=[]
    for s in range(le):#遍历模式库
        #数据库2
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)
    
    #后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库3
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)  
    
    #前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库4
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)
    
    #上
    d1=[]
    for s in range(le):#遍历模式库
        #数据库5
        b=np.zeros((template_h,template_x,template_y),int)
        b[dis,:,:]=1        
        t=database[s]*b
        d1.append(t)
    cdatabase.append(d1)
  
    
    cdatabase.append(database)#数据库6为本体
    print('done')
    return cdatabase

def databaseclusterAI(cdatabase,U):#模式分类
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
 

    

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    
    time.sleep(5)
    print('process end')

    Cdatabase=[]
    cc1=np.load('./database/clusters0.npy')
    cc2=np.load('./database/clusters1.npy')
    cc3=np.load('./database/clusters2.npy')
    cc4=np.load('./database/clusters3.npy')
    cc5=np.load('./database/clusters4.npy')
    cc6=np.load('./database/clusters5.npy')

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
    
    #np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase


def patternsearchAI(o1,database,N):#根据重叠区选取备选模式 #直接返回候选模板 ver1.0#后续增加坐标返回模式
    #N为备选模板个数
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    template_h=database[0].shape[0]
    template_x=database[0].shape[1]
    template_y=database[0].shape[2]
    d=[]#备选列表    
    drill1=o1.reshape(ss,1)
    for n in range(len(database)):
        ctem=database[n]
        drill2=ctem.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)
    r=random.randint(0,len(si)-1)
    t=si[r]    
    return database[t]
'''
def initialgridAI(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,database,zuobiaolist,N):
    #自动初始化网格系统
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=lujinglistAI(m,template_h,template_x,template_y)
    random.shuffle(lujing)
    
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]*10#最大循环次数
    DevilTrigger=False
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    ################重叠区选取器#####################
    ss=template_h*template_x*template_y
    dis=[]#后左上
    disx=[]#前
    disy=[]#右
    dish=[]#下
    b=np.zeros((template_h,template_x,template_y),int)
    
    d=[]#待候选列表
    c1=99999#对比数
    cc=0#对比序号
    
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
        
        
    #############################################
    
    
    for n in range(Fin):
        if n==len(lujing):
            break
        if lujing[n] not in Banlujing:
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            if o1[0,template_x/2,template_y/2]!=1:
                #上
                b[dis,:,:]=1
                k=k+1
            if o1[template_h-1,template_x/2,template_y/2]!=1:
                #下
                b[dish,:,:]=1
                k=k+1
            if o1[template_h/2,0,template_y/2]!=1:
                #后
                b[:,dis,:]=1
                k=k+1
            if o1[template_h/2,template_x-1,template_y/2]!=1:
                #前 
                b[:,disx,:]=1
                k=k+1
            if o1[template_h/2,template_x/2,0]!=1:
                #左
                b[:,:,dis]=1
                k=k+1
            if o1[template_h/2,template_x/2,template_y-1]!=1:
                #右
                b[:,:,disy]=1
                k=k+1
            if k!=0:
                temo=o1*b
                tem=patternsearchAI(temo,database,N)
                m2=template1RAI(m2,tem,h1,x1,y1)
                for hb in range(h1-lag-lag,h1+1):
                    for xb in range(x1-lag-lag,x1+1):
                        for yb in range(y1-lag-lag,y1+1):
                            Banlujing.append((hb,xb,yb))#将已经模拟的区域坐标加入黑名单
                #print(n)
                #print len(lujing)
            else:
                lujing.append(lujing[n])
                
        
    m=cut(m2,lag)
    return m
'''
################################################################
################################################################
################################################################
def initialroadlistAI(m,template_h,template_x,template_y,lag):
    #自动初始化网格系统
    lujing=[]
    Roadlist=[]#最终路径名单
    lujing=lujinglistAI(m,template_h,template_x,template_y)
    random.shuffle(lujing)
    #print len(lujing)
    
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]
    Fin=Fin*1000#最大循环次数
    DevilTrigger=False
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    Banlujing=[]
    ################重叠区选取器#####################
    ss=template_h*template_x*template_y
    dis=[]#后左上
    disx=[]#前
    disy=[]#右
    dish=[]#下
    b=np.zeros((template_h,template_x,template_y),int)
    
    d=[]#待候选列表

    cc=0#对比序号
    
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
        
        
    #############################################
    n=0
    
    while n<len(lujing):
        if m2[lujing[n][0]+lag,lujing[n][1]+lag,lujing[n][2]+lag]==-1:
            #if lujing[n] not in Banlujing:
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            
            if o1[0,template_x//2,template_y//2]!=-1:
                #上
                k=k+1
           
            if o1[template_h-1,template_x//2,template_y//2]!=-1:
                #下
                k=k+1
            if o1[template_h//2,0,template_y//2]!=-1:
                #后
                k=k+1
            if o1[template_h//2,template_x-1,template_y//2]!=-1:
                #前 
                k=k+1
            if o1[template_h//2,template_x//2,0]!=-1:
                #左
                k=k+1
            if o1[template_h//2,template_x//2,template_y-1]!=-1:
                #右
                k=k+1
            if (h1>template_h-lag) and (k>=2):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            elif (h1<=template_h-lag) and (k!=0):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                '''
                for hb in range(h1-lag-lag,h1+1):
                    for xb in range(x1-lag-lag,x1+1):
                        for yb in range(y1-lag-lag,y1+1):
                            Banlujing.append((hb,xb,yb))#将已经模拟的区域坐标加入黑名单
                '''
            else:
                lujing.append(lujing[n])
        #print len(Roadlist),len(lujing)-n
        n=n+1
        #print len(Roadlist)
    print('roadlist initial done')
    return Roadlist

def initialroadlistAI2(m,template_h,template_x,template_y,lag):#改进版
    #自动初始化网格系统
    lujing=[]
    Roadlist=[]#最终路径名单
    lujing=lujinglistAI(m,template_h,template_x,template_y)
    random.shuffle(lujing)
    #print len(lujing)
    
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]
    Fin=Fin*1000#最大循环次数
    DevilTrigger=False
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    Banlujing=[]
    ################重叠区选取器#####################
    ss=template_h*template_x*template_y
    dis=[]#后左上
    disx=[]#前
    disy=[]#右
    dish=[]#下
    b=np.zeros((template_h,template_x,template_y),int)
    
    d=[]#待候选列表

    cc=0#对比序号
    
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
        
        
    #############################################
    
    
    while n<len(lujing):
        if m2[lujing[n][0]+lag,lujing[n][1]+lag,lujing[n][2]+lag]==-1:
            #if lujing[n] not in Banlujing:
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            if temdetectD(o1[0:lag,:,:]): 
                #上
                k=k+1

            if temdetectD(o1[template_h-lag:template_h,:,:]):
                #下
                k=k+1
            if temdetectD(o1[:,0:lag,:]):
                #后
                k=k+1
            if temdetectD(o1[:,template_x-lag:template_x,:]):
                #前 
                k=k+1
            if temdetectD(o1[:,:,0:lag]):
                #左
                k=k+1
            if temdetectD(o1[:,:,template_y-lag:template_y]):
                #右
                k=k+1
            if (h1>template_h-lag) and (k>=2):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            elif (h1<=template_h-lag) and (k!=0):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                '''
                for hb in range(h1-lag-lag,h1+1):
                    for xb in range(x1-lag-lag,x1+1):
                        for yb in range(y1-lag-lag,y1+1):
                            Banlujing.append((hb,xb,yb))#将已经模拟的区域坐标加入黑名单
                '''
            else:
                lujing.append(lujing[n])
        #print len(Roadlist),len(lujing)-n
        n=n+1
        #print len(Roadlist)
    #print('roadlist initial done')
    return Roadlist

################################################################
################################################################
################################################################
def patternsearchAI2(o1,c,database,canpatternlist,N):#根据重叠区,在候选列表中选取备选模式 #直接返回候选模板 ver2.0 
    #N为备选模板个数
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    template_h=database[0].shape[0]
    template_x=database[0].shape[1]
    template_y=database[0].shape[2]
    d=[]#备选列表    
    drill1=o1.reshape(ss,1)
    #print len(canpatternlist)
    for n in range(len(canpatternlist)):
        ctem=database[canpatternlist[n]]*c
        drill2=ctem.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)
    r=random.randint(0,len(si)-1)
    t=si[r]    
    return database[canpatternlist[t]]

def random_index(rate):
    #
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    start = 0
    index = 0
    randnum = random.randint(1, 1)
    #randnum = random.randint(1, sum(rate))会不足1
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index


def patternsearchAI3(o1,h,x,y,c,database,canpatternlist,zuobiaolist,N):#根据重叠区,在候选列表中选取备选模式 # ver3.0 IDW
    #N为备选模板个数
    #print h,zuobiaolist[canpatternlist[0]]
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    template_h=database[0].shape[0]
    template_x=database[0].shape[1]
    template_y=database[0].shape[2]
    d=[]#备选列表    
    drill1=o1.reshape(ss,1)
    #print len(canpatternlist)
    for n in range(len(canpatternlist)):
        ctem=database[canpatternlist[n]]*c
        drill2=ctem.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)
    zuobiaomin=[]
    zuobiaopro=[]
    acc=0
    for r in range(len(si)):
        a1=h-zuobiaolist[canpatternlist[si[r]]][0]
        a2=x-zuobiaolist[canpatternlist[si[r]]][1]
        a3=y-zuobiaolist[canpatternlist[si[r]]][2]
        sam=math.sqrt((a1**2)+(a2**2)+(a3**2))
        #print a1,a2,a3,sam
        acc=acc+(1/sam)
        zuobiaomin.append(sam)
    for r in range(len(si)):
        zuobiaopro.append(((1/zuobiaomin[r])/acc))
    #print sum(zuobiaopro)
    rs= random_index(zuobiaopro)#IDW
    #print rs,len(si),si[rs]
    return database[canpatternlist[si[rs]]]

def patternsearchAI4(o1,h,x,y,c,database,canpatternlist,zuobiaolist,N):#根据重叠区,在候选列表中选取备选模式 #返回坐标最接近的候选模板 ver4.0 信息熵
    #N为备选模板个数
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    template_h=database[0].shape[0]
    template_x=database[0].shape[1]
    template_y=database[0].shape[2]
    d=[]#备选列表    
    drill1=o1.reshape(ss,1)
    #print len(canpatternlist)
    for n in range(len(canpatternlist)):
        ctem=database[canpatternlist[n]]*c
        drill2=ctem.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)
    score=np.zeros((len(si),2),float)

    for r in range(len(si)):
        score[r,0]=d[si[r]]#记录下距离
    for r in range(len(si)):
        a1=h-zuobiaolist[canpatternlist[si[r]]][0]
        a2=x-zuobiaolist[canpatternlist[si[r]]][1]
        a3=y-zuobiaolist[canpatternlist[si[r]]][2]
        sam=math.sqrt((a1**2)+(a2**2)+(a3**2))
        score[r,1]=sam#记录下欧式距离
      
    yys= maxscore(score)#信息熵返还最大分数的序号
    rs=si[yys]
    return database[canpatternlist[rs]]
'''

def initialgridAI(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):
    #自动初始化网格系统，分类加速ver
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=lujinglistAI(m,template_h,template_x,template_y)
    random.shuffle(lujing)
    
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]*10#最大循环次数
    DevilTrigger=False
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    ################重叠区选取器#####################
    ss=template_h*template_x*template_y
    dis=[]#后左上
    disx=[]#前
    disy=[]#右
    dish=[]#下
    b=np.zeros((template_h,template_x,template_y),int)
    
    d=[]#待候选列表
    c1=99999#对比数
    cc=0#对比序号
    
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
        
        
    #############################################
    
    
    for n in range(Fin):
        if n==len(lujing):
            break
        if lujing[n] not in Banlujing:
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            canpatternlist0=[]
            canpatternlist1=[]
            canpatternlist2=[]
            canpatternlist3=[]
            canpatternlist4=[]
            canpatternlist5=[]
            c=np.zeros((template_h,template_x,template_y),int)
            if o1[0,template_x/2,template_y/2]!=1:
                #上
                b=np.zeros((template_h,template_x,template_y),int)
                b[dis,:,:]=1
                c[dis,:,:]=1
                temo=o1*b
                canpatternlist0=patternsearchDi(Cdatabase[5],cdatabase[5],temo)
                
                k=k+1
            if o1[template_h-1,template_x/2,template_y/2]!=1:
                #下
                b=np.zeros((template_h,template_x,template_y),int)
                b[dish,:,:]=1
                c[dish,:,:]=1
                temo=o1*b
                canpatternlist1=patternsearchDi(Cdatabase[0],cdatabase[0],temo)
                k=k+1
            if o1[template_h/2,0,template_y/2]!=1:
                #后
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,dis,:]=1
                c[:,dis,:]=1
                temo=o1*b
                canpatternlist2=patternsearchDi(Cdatabase[3],cdatabase[3],temo)
                k=k+1
            if o1[template_h/2,template_x-1,template_y/2]!=1:
                #前
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,disx,:]=1
                c[:,disx,:]=1
                temo=o1*b
                canpatternlist3=patternsearchDi(Cdatabase[4],cdatabase[4],temo)
                k=k+1
            if o1[template_h/2,template_x/2,0]!=1:
                #左
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,dis]=1
                c[:,:,dis]=1
                temo=o1*b
                canpatternlist4=patternsearchDi(Cdatabase[1],cdatabase[1],temo)
                k=k+1
            if o1[template_h/2,template_x/2,template_y-1]!=1:
                #右
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,disy]=1
                c[:,:,disy]=1
                temo=o1*b
                canpatternlist5=patternsearchDi(Cdatabase[2],cdatabase[2],temo)
                k=k+1
                
                
                
            if k!=0:
                canpatternlist=[]
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist0)))
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist1)))
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist4)))
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist5)))
                canpatternlist=list(set(canpatternlist))
                temo=o1*c
                tem=patternsearchAI2(temo,c,cdatabase[6],canpatternlist,N)
                m2=template1RAI(m2,tem,h1,x1,y1)
                for hb in range(h1-lag-lag,h1+1):
                    for xb in range(x1-lag-lag,x1+1):
                        for yb in range(y1-lag-lag,y1+1):
                            Banlujing.append((hb,xb,yb))#将已经模拟的区域坐标加入黑名单
                print(n)
                print len(lujing)
            else:
                lujing.append(lujing[n])
                
        
    m=cut(m2,lag)
    return m
'''
def initialgridAI2(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAI(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]*10#最大循环次数
    DevilTrigger=False
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    ################重叠区选取器#####################
    ss=template_h*template_x*template_y
    dis=[]#后左上
    disx=[]#前
    disy=[]#右
    dish=[]#下
    b=np.zeros((template_h,template_x,template_y),int)
    
    d=[]#待候选列表
    c1=99999#对比数
    cc=0#对比序号
    
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
        
        
    #############################################
    
    
    for n in range(len(lujing)):
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
        k=0#重叠区计数器
        flag=0
        canpatternlist0=[]
        canpatternlist1=[]
        canpatternlist2=[]
        canpatternlist3=[]
        canpatternlist4=[]
        canpatternlist5=[]
        c=np.zeros((template_h,template_x,template_y),int)
        '''
        if o1[0,template_x/2,template_y/2]!=-1:
            #上
            b=np.zeros((template_h,template_x,template_y),int)
            b[dis,:,:]=1
            c[dis,:,:]=1
            temo=o1*b
            canpatternlist0=patternsearchDi(Cdatabase[5],cdatabase[5],temo)
        '''        

        if o1[template_h-1,template_x//2,template_y//2]!=-1:
            #下
            b=np.zeros((template_h,template_x,template_y),int)
            b[dish,:,:]=1
            c[dish,:,:]=1
            temo=o1*b
            canpatternlist1=patternsearchDi(Cdatabase[0],cdatabase[0],temo)
            if temdetect0d(o1[template_h-1,:,:]):
                flag=1

        if o1[template_h//2,0,template_y//2]!=-1:
            #后
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,dis,:]=1
            c[:,dis,:]=1
            temo=o1*b
            canpatternlist2=patternsearchDi(Cdatabase[3],cdatabase[3],temo)
            if temdetect0d(o1[:,0,:]):
                flag=1

        if o1[template_h//2,template_x-1,template_y//2]!=-1:
            #前
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,disx,:]=1
            c[:,disx,:]=1
            temo=o1*b
            canpatternlist3=patternsearchDi(Cdatabase[4],cdatabase[4],temo)
            if temdetect0d(o1[:,template_x-1,:]):
                flag=1
 
        if o1[template_h//2,template_x//2,0]!=-1:
            #左
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,dis]=1
            c[:,:,dis]=1
            temo=o1*b
            canpatternlist4=patternsearchDi(Cdatabase[1],cdatabase[1],temo)
            if temdetect0d(o1[:,:,0]):
                flag=1

        if o1[template_h//2,template_x//2,template_y-1]!=-1:
            #右
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,disy]=1
            c[:,:,disy]=1
            temo=o1*b
            canpatternlist5=patternsearchDi(Cdatabase[2],cdatabase[2],temo)
            if temdetect0d(o1[:,:,template_y-1]):
                flag=1

                
                
                

        canpatternlist=[]
        canpatternlist=list(set(canpatternlist0).union(set(canpatternlist1)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist4)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist5)))
        
        #print len(canpatternlist0),len(canpatternlist1),len(canpatternlist2),len(canpatternlist3),len(canpatternlist4),len(canpatternlist5)
        #print canpatternlist0,canpatternlist1,canpatternlist2,canpatternlist3,canpatternlist4,canpatternlist5
        canpatternlist=list(set(canpatternlist))
        #print len(canpatternlist),canpatternlist
        if flag!=0:
            tem=np.zeros((template_h,template_x,template_y),int)
        else:
            #print("have")
            temo=o1*c
            tem=patternsearchAI2(temo,c,cdatabase[6],canpatternlist,N)
        m2=template1RAI(m2,tem,h1,x1,y1)
            
        
    m=cut(m2,lag)
    return m

def initialgridAI2zuobiaover(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAI(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]*10#最大循环次数
    DevilTrigger=False
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    ################重叠区选取器#####################
    ss=template_h*template_x*template_y
    dis=[]#后左上
    disx=[]#前
    disy=[]#右
    dish=[]#下
    b=np.zeros((template_h,template_x,template_y),int)
    
    d=[]#待候选列表
    c1=99999#对比数
    cc=0#对比序号
    
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
        
        
    #############################################
    
    
    for n in range(len(lujing)):
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
        k=0#重叠区计数器
        flag=0
        canpatternlist0=[]
        canpatternlist1=[]
        canpatternlist2=[]
        canpatternlist3=[]
        canpatternlist4=[]
        canpatternlist5=[]
        c=np.zeros((template_h,template_x,template_y),int)
        '''
        if o1[0,template_x/2,template_y/2]!=-1:
            #上
            b=np.zeros((template_h,template_x,template_y),int)
            b[dis,:,:]=1
            c[dis,:,:]=1
            temo=o1*b
            canpatternlist0=patternsearchDi(Cdatabase[5],cdatabase[5],temo)
        '''        

        if o1[template_h-1,template_x/2,template_y/2]!=-1:
            #下
            b=np.zeros((template_h,template_x,template_y),int)
            b[dish,:,:]=1
            c[dish,:,:]=1
            temo=o1*b
            canpatternlist1=patternsearchDi(Cdatabase[0],cdatabase[0],temo)
            if temdetect0d(o1[template_h-1,:,:]):
                flag=1

        if o1[template_h/2,0,template_y/2]!=-1:
            #后
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,dis,:]=1
            c[:,dis,:]=1
            temo=o1*b
            canpatternlist2=patternsearchDi(Cdatabase[3],cdatabase[3],temo)
            if temdetect0d(o1[:,0,:]):
                flag=1

        if o1[template_h/2,template_x-1,template_y/2]!=-1:
            #前
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,disx,:]=1
            c[:,disx,:]=1
            temo=o1*b
            canpatternlist3=patternsearchDi(Cdatabase[4],cdatabase[4],temo)
            if temdetect0d(o1[:,template_x-1,:]):
                flag=1
 
        if o1[template_h/2,template_x/2,0]!=-1:
            #左
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,dis]=1
            c[:,:,dis]=1
            temo=o1*b
            canpatternlist4=patternsearchDi(Cdatabase[1],cdatabase[1],temo)
            if temdetect0d(o1[:,:,0]):
                flag=1

        if o1[template_h/2,template_x/2,template_y-1]!=-1:
            #右
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,disy]=1
            c[:,:,disy]=1
            temo=o1*b
            canpatternlist5=patternsearchDi(Cdatabase[2],cdatabase[2],temo)
            if temdetect0d(o1[:,:,template_y-1]):
                flag=1

                
                
                

        canpatternlist=[]
        canpatternlist=list(set(canpatternlist0).union(set(canpatternlist1)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist4)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist5)))
        
        #print len(canpatternlist0),len(canpatternlist1),len(canpatternlist2),len(canpatternlist3),len(canpatternlist4),len(canpatternlist5)
        #print canpatternlist0,canpatternlist1,canpatternlist2,canpatternlist3,canpatternlist4,canpatternlist5
        canpatternlist=list(set(canpatternlist))
        #print len(canpatternlist),canpatternlist
        if flag!=0:
            tem=np.zeros((template_h,template_x,template_y),int)
        else:
            #print("have")
            temo=o1*c
            tem=patternsearchAI3(temo,lujing[n][0],lujing[n][1],lujing[n][2],c,cdatabase[6],canpatternlist,zuobiaolist,N)#判断坐标远近返回最近的候选模板
        m2=template1RAI(m2,tem,h1,x1,y1)
            
        
    m=cut(m2,lag)
    return m


def gosiminitialAI(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U):
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
    np.save('./output/outputinitial1.npy',m)

 
    my_file = Path("./database/Cdatabase.npy")
    if my_file.exists():
        Cdatabase=np.load('./database/Cdatabase.npy')
        cdatabase=np.load('./database/cdatabase.npy')
        database=np.load('./database/database.npy')
        zuobiaolist=np.load('./database/zuobiaolist.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        database,zuobiaolist=databasebuildAI(m,template_h,template_x,template_y)#数据库构建
        np.save('./database/database.npy',database)
        np.save('./database/zuobiaolist.npy',zuobiaolist)
        cdatabase=databasecataAI(database,lag)
        np.save('./database/cdatabase.npy',cdatabase)
        Cdatabase=databaseclusterAI(cdatabase,U)
        np.save('./database/Cdatabase.npy',Cdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    m=initialgridAI2(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N)
    
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


def gosiminitialAIzuobiaover(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格
    #m为已经导入了Ti的模拟网格
    time_start1=time.time()
    m=extendTimodel(m,template_h,template_x,template_y)#拓展模拟网格
    #np.save('./output/wtf.npy',m)

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
        database,zuobiaolist=databasebuildAI(m,template_h,template_x,template_y)#数据库构建
        np.save('./database/database.npy',database)
        np.save('./database/zuobiaolist.npy',zuobiaolist)
        cdatabase=databasecataAI(database,lag)
        np.save('./database/cdatabase.npy',cdatabase)
        Cdatabase=databaseclusterAI(cdatabase,U)
        np.save('./database/Cdatabase.npy',Cdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    m=initialgridAI2zuobiaover(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N)
    
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



def TIextendforEM(section,m,template_x,template_y,x1,y1,x2,y2):#EM迭代用剖面提取，(x,y）为剖面定位的一组坐标
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
    Tizuobiao.append(Tizuobiaoh)
    Tizuobiao.append(Tizuobiaox)
    Tizuobiao.append(Tizuobiaoy)
    #Ti=extendTimodel(fow,c,c,c)
    Ti=np.load('./output/ext1.npy')
    return Ti,Tizuobiao

def sectionloadandextend(m,template_x,template_y,flag,scale):#flag==1为patchmatch步骤，0为initial步骤
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
            Ti,Tizuobiao=TIextendforEM(section,m,template_x,template_y,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)
            Tilist.append(Ti)
            Tizuobiaolist.append(Tizuobiao)
    #都执行完后可进行gosiminitialAI
    return m,Tilist,Tizuobiaolist








def GosimAI(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr):
    time_starts=time.time()#计时开始
    #初始模型构建 m为已经构建好的模拟网格
    m,Tilist,zuobiaolist=sectionloadandextend(m,patternSizex,patternSizey,0,1)
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAI(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U)
    print('initial done')


    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)
        patternSizex=int(patternSizex*scale[ni])
        patternSizey=int(patternSizey*scale[ni])
        patternSizeh=int(patternSizeh*scale[ni])
        #TI转换为当前尺度
        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=sectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)
        mm=extendTimodel(mm,patternSizeh,patternSizex,patternSizey)
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
        #m= patchmatchmultiTiBZ(m,mm,Tilist,size,itr,1)
        CTI=[]#检测使用率剖面
        m,CTI= patchmatchmultiTiBZ2(m,mm,Tilist,size,itr,1)
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
#####################################################

def GosimAIzuobiaover(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr):
    time_starts=time.time()#计时开始
    #初始模型构建 m为已经构建好的模拟网格
    m,Tilist,zuobiaolist=sectionloadandextend(m,patternSizex,patternSizey,0,1)
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIzuobiaover(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U)
    print('initial done')


    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)
        patternSizex=int(patternSizex*scale[ni])
        patternSizey=int(patternSizey*scale[ni])
        patternSizeh=int(patternSizeh*scale[ni])
        #TI转换为当前尺度
        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=sectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)
        mm=extendTimodel(mm,patternSizeh,patternSizex,patternSizey)
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
        CTI=[]#检测使用率剖面
        m,CTI= patchmatchmultiTiBZ2(m,mm,Tilist,size,itr,1)
        #计算量加大过多
        #m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path="./output/reconstruction.npy"
        np.save(path,m)
        path="./output/CTI.npy"
        np.save(path,CTI)
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
#####################################################






'''
m=np.ones((100,75,150),int)
m,Tilist,zuobiaolist=sectionloadandextend(m,7,7)
m=extendTimodel(m,7,7,7)
time_e=time.time()
print time_e-time_s

data=m.transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
grid.point_data.scalars = np.ravel(data,order='F') 
grid.point_data.scalars.name = 'lithology' 
write_data(grid, './output/ssss.vtk') 

data=Tilist[0].transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
grid.point_data.scalars = np.ravel(data,order='F') 
grid.point_data.scalars.name = 'lithology' 
write_data(grid, './output/test1.vtk')


data=Tilist[1].transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
grid.point_data.scalars = np.ravel(data,order='F') 
grid.point_data.scalars.name = 'lithology' 
write_data(grid, './output/test2.vtk')


data=Tilist[2].transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
grid.point_data.scalars = np.ravel(data,order='F') 
grid.point_data.scalars.name = 'lithology' 
write_data(grid, './output/test3.vtk') 
'''
'''
m=np.ones((4,15,15),int)

m[:,3,:]=2
m[0,3,:]=0
m[:,12,:]=4
m[:,:,2]=2
s=2*sectionread_x(m,1,2,14,14)
sectionload_x(m,s,1,2,13,14)
#m[:,:,range(12,15)]=2
print m
m=extendTimodel(m,1,1,1)
print m
r=lujinglistAI(m,3,3,3)
print r
#s=2*sectionread_x(m,1,2,14,14)
#sectionload_x(m,s,1,2,13,14)
#sectionload_x(m,s,1,2,14,13)
#print m
m=np.pad(m,1,'edge')
tem=np.zeros((3,3,3),int)
for n in range(len(r)):
    h=r[n][0]+1
    x=r[n][1]+1
    y=r[n][2]+1
    
    template1RAI(m,tem,h,x,y)
m=cut(m,1)
print m
tem=np.zeros((3,3,3),int)
tem[:,0,:]=5
if temdetect(tem):
    print tem
'''

