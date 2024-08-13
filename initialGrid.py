#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from zuobiao import*
from Pdatabase import*
from patterntool import*
from editdistance import*
from roadlist import*
import heapq
from listheap import*
from cluster import*
#######################################################################################################################
def patternsearch1(cdatabase,database,tem,N):#初始模型构建中依据重叠区用编辑距离搜索最适合模板，N为备选模板个数
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    #print drill1
    d=[]
    for n in range(len(cdatabase)):
        tem2=cdatabase[n]
        drill2=tem2.reshape(ss,1)
        #print drill1,drill2
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)
    r=random.randint(0,N-1)
    t=si[r]
    #print si
    #print t
    return database[t]


def patternsearch2(Cdatabase,cdatabase,database,tem,N):#初始模型构建中依据重叠区用编辑距离搜索最适合模板，N为备选模板个数,Cdatabase为分类好的数据库
    #Cdatabase和cdatabase都是重叠区定好的
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    #print drill1
    #print len(Cdatabase),len(cdatabase)
    d=[]
    c1=99999
    cc=0
    for n in range(Cdatabase.shape[0]):
        rrr=random.randint(0,len(Cdatabase[n])-1)
        tem2=cdatabase[Cdatabase[n][rrr]]
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
           c1=fun
           cc=n
    #print tem
    #print c
    for n in range(len(Cdatabase[cc])):
        #rint Cdatabase[cc][n],Cdatabase[cc][1],cdatabase[0]
        tem2=cdatabase[Cdatabase[cc][n]]
        drill2=tem2.reshape(ss,1)
        #print drill1,drill2
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)
    r=random.randint(0,len(si)-1)
    t=si[r]
    #print si
    #print database[Cdatabase[cc][t]]
    return database[Cdatabase[cc][t]]

def patternsearch3(Cdatabase,cdatabase,database,tem,N):#初始模型构建中依据重叠区用编辑距离搜索最适合模板，N为备选模板个数,Cdatabase为分类好的数据库
    #简化版，类内直接选取一类
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    #print drill1
    d=[]
    c=[]
    for n in range(Cdatabase.shape[0]):
        tem2=cdatabase[Cdatabase[n][0]]
        drill2=tem2.reshape(ss,1)
        c.append(hamming_distance(drill1,drill2))
    CC=getListMinNumIndex(c,1)
    cc=CC[0]
    r=random.randint(0,len(Cdatabase[cc])-1)
    t=Cdatabase[cc][r]
    #print si
    #print t
    return database[t]

def patternsearchDi(Cdatabase,cdatabase,tem):#初始模型构建中依据重叠区用编辑距离搜索最适合模板的代号,Cdatabase为分类好的数据库
    #Cdatabase和cdatabase都是重叠区定好的
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    #print drill1
    #print(len(Cdatabase),len(cdatabase))
    c1=99999
    cc=0
    for n in range(Cdatabase.shape[0]):
        rrr=random.randint(0,len(Cdatabase[n])-1)
        tem2=cdatabase[Cdatabase[n][rrr]]
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
           c1=fun
           cc=n
    #print cc
    #print Cdatabase[cc]
    return Cdatabase[cc]

'''
def initialGrid1(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,database,zuobiaolist,N):#非柱状初始模型构建,N为备选模式个数
    #N为备选模式个数
    #lag为重叠区宽度
    d1=[]
    m,d1=lujing1(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    d2=[]
    d2=lujing1pluse(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    #print d1
    cdatabase=temdatabasec_5(database,lag,template_h,template_x,template_y)
    H=m.shape[0]
    hh=int((template_h-1)/2)
    s3=int(((H-1-(2*hh))/lag_h))+1
    gai=len(d2)/s3
    for n in range(gai):
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        #print o1
        #判断是哪个重叠区
        if d2[n]==0:#左后
            tem=patternsearch1(cdatabase[0],database,o1,N)#左后
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==1:#左前
            tem=patternsearch1(cdatabase[1],database,o1,N)#左前
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==2:#右前
            tem=patternsearch1(cdatabase[2],database,o1,N)#右前
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==3:#右后
            tem=patternsearch1(cdatabase[3],database,o1,N)#右后
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==4:
            tem=patternsearch1(cdatabase[4],database,o1,N)
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
        if d2[n]==5:
            tem=patternsearch1(cdatabase[5],database,o1,N)
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==6:
            tem=patternsearch1(cdatabase[6],database,o1,N)
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==7:
            tem=patternsearch1(cdatabase[7],database,o1,N)
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前
            
    for n in range(gai,len(d1)):
        if d2[n]==8:#左后
            tem=patternsearch1(cdatabase[8],database,o1,N)#左后
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==9:#左前
            tem=patternsearch1(cdatabase[9],database,o1,N)#左前
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==10:#右前
            tem=patternsearch1(cdatabase[10],database,o1,N)#右前
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==11:#右后
            tem=patternsearch1(cdatabase[11],database,o1,N)#右后
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==12:
            tem=patternsearch1(cdatabase[12],database,o1,N)
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
        if d2[n]==13:
            tem=patternsearch1(cdatabase[13],database,o1,N)
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==14:
            tem=patternsearch1(cdatabase[14],database,o1,N)
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==15:
            tem=patternsearch1(cdatabase[15],database,o1,N)
            template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前      
    return m
'''

def Cdatabasebuile(database,lag,template_h,template_x,template_y,U):#构建分类数据库的整合函数 U为阈值
    #Num为分类数
    cdatabase=temdatabasec_5(database,lag,template_h,template_x,template_y)
    np.save('./database/cdatabase.npy',cdatabase)
    Cdatabase=initialpatterndatabaseSimple(cdatabase,U)
    return cdatabase,Cdatabase
    
def Cdatabasebuile2(database,lag,template_h,template_x,template_y,U):#构建分类数据库的整合函数2 U为阈值,9类别
    #Num为分类数
    cdatabase=temdatabasec_6(database,lag,template_h,template_x,template_y)
    np.save('./database/cdatabase.npy',cdatabase)
    Cdatabase=initialpatterndatabaseSimple2(cdatabase,U)
    return cdatabase,Cdatabase
def Cdatabasebuile3(database,lag,template_h,template_x,template_y,U):#构建分类数据库的整合函数2 U为阈值,9类别
    #Num为分类数
    cdatabase=temdatabasec_7(database,lag,template_h,template_x,template_y)
    np.save('./database/cdatabase.npy',cdatabase)
    Cdatabase=initialpatterndatabaseSimple3(cdatabase,U)
    return cdatabase,Cdatabase

def Cdatabasebuilezhu(database,lag,template_x,template_y,U):#构建柱状分类数据库的整合函数2 U为阈值,9类别
    #Num为分类数
    cdatabase=temdatabasec_8(database,lag,template_x,template_y)
    np.save('./database/cdatabase.npy',cdatabase)
    Cdatabase=initialpatterndatabaseSimplezhu(cdatabase,U)
    return cdatabase,Cdatabase

def initialGrid2(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):#非柱状初始模型构建,N为备选模式个数
    #直接用于分类好的数据库Cdatabase
    #N为备选模式个数
    #lag为重叠区宽度
    #flag为是否开启模式库分类
    d1=[]
    m,d1=lujing1(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    d2=[]
    d2=lujing1pluse(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    #print d1
    H=m.shape[0]
    hh=int((template_h-1)/2)
    s3=int(((H-1-(2*hh))/lag_h))+1
    gai=len(d2)/s3
    print( d1)
    for n in range(gai):
        #print d1[n][0],d1[n][1],d1[n][2]
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        #print o1
        #判断是哪个重叠区
        if d2[n]==0:#左后
            o1=layoutpick(o1,lag,0)
            tem=patternsearch2(Cdatabase[0],cdatabase[0],cdatabase[16],o1,N)#左后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==1:#左前
            o1=layoutpick(o1,lag,1)
            tem=patternsearch2(Cdatabase[1],cdatabase[1],cdatabase[16],o1,N)#左前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==2:#右前
            o1=layoutpick(o1,lag,2)
            tem=patternsearch2(Cdatabase[2],cdatabase[2],cdatabase[16],o1,N)#右前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==3:#右后
            o1=layoutpick(o1,lag,3)
            tem=patternsearch2(Cdatabase[3],cdatabase[3],cdatabase[16],o1,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==4:
            o1=layoutpick(o1,lag,4)
            tem=patternsearch2(Cdatabase[4],cdatabase[4],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
        if d2[n]==5:
            o1=layoutpick(o1,lag,5)
            tem=patternsearch2(Cdatabase[5],cdatabase[5],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==6:
            o1=layoutpick(o1,lag,6)
            tem=patternsearch2(Cdatabase[6],cdatabase[6],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==7:
            o1=layoutpick(o1,lag,7)
            tem=patternsearch2(Cdatabase[7],cdatabase[7],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前
    for n in range(gai,len(d1)):
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        if d2[n]==8:#左后
            o1=layoutpick(o1,lag,8)
            tem=patternsearch2(Cdatabase[8],cdatabase[8],cdatabase[16],o1,N)#左后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==9:#左前
            o1=layoutpick(o1,lag,9)
            tem=patternsearch2(Cdatabase[9],cdatabase[9],cdatabase[16],o1,N)#左前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==10:#右前
            o1=layoutpick(o1,lag,10)
            tem=patternsearch2(Cdatabase[10],cdatabase[10],cdatabase[16],o1,N)#右前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==11:#右后
            o1=layoutpick(o1,lag,11)
            tem=patternsearch2(Cdatabase[11],cdatabase[11],cdatabase[16],o1,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==12:
            o1=layoutpick(o1,lag,12)
            tem=patternsearch2(Cdatabase[12],cdatabase[12],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
        if d2[n]==13:
            o1=layoutpick(o1,lag,13)
            tem=patternsearch2(Cdatabase[13],cdatabase[13],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==14:
            o1=layoutpick(o1,lag,14)
            tem=patternsearch2(Cdatabase[14],cdatabase[14],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==15:
            o1=layoutpick(o1,lag,15)
            tem=patternsearch2(Cdatabase[15],cdatabase[15],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前    
    return m

def initialGrid3(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):#非柱状初始模型构建,N为备选模式个数
    #直接用于分类好的数据库Cdatabase  #search 简略版
    #N为备选模式个数
    #lag为重叠区宽度
    #flag为是否开启模式库分类
    d1=[]
    m,d1=lujing1(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    d2=[]
    d2=lujing1pluse(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    #print d1
    H=m.shape[0]
    hh=int((template_h-1)/2)
    s3=int(((H-1-(2*hh))/lag_h))+1
    gai=len(d2)/s3
    for n in range(gai):
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        #print o1
        #判断是哪个重叠区
        if d2[n]==0:#左后
            o1=layoutpick(o1,lag,0)
            tem=patternsearch3(Cdatabase[0],cdatabase[0],cdatabase[16],o1,N)#左后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==1:#左前
            o1=layoutpick(o1,lag,1)
            tem=patternsearch3(Cdatabase[1],cdatabase[1],cdatabase[16],o1,N)#左前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==2:#右前
            o1=layoutpick(o1,lag,2)
            tem=patternsearch3(Cdatabase[2],cdatabase[2],cdatabase[16],o1,N)#右前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==3:#右后
            o1=layoutpick(o1,lag,3)
            tem=patternsearch3(Cdatabase[3],cdatabase[3],cdatabase[16],o1,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==4:
            o1=layoutpick(o1,lag,4)
            tem=patternsearch3(Cdatabase[4],cdatabase[4],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
        if d2[n]==5:
            o1=layoutpick(o1,lag,5)
            tem=patternsearch3(Cdatabase[5],cdatabase[5],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==6:
            o1=layoutpick(o1,lag,6)
            tem=patternsearch3(Cdatabase[6],cdatabase[6],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==7:
            o1=layoutpick(o1,lag,7)
            tem=patternsearch3(Cdatabase[7],cdatabase[7],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前
            
    for n in range(gai,len(d1)):
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        if d2[n]==8:#左后
            o1=layoutpick(o1,lag,8)
            tem=patternsearch3(Cdatabase[8],cdatabase[8],cdatabase[16],o1,N)#左后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==9:#左前
            o1=layoutpick(o1,lag,9)
            tem=patternsearch3(Cdatabase[9],cdatabase[9],cdatabase[16],o1,N)#左前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==10:#右前
            o1=layoutpick(o1,lag,10)
            tem=patternsearch3(Cdatabase[10],cdatabase[10],cdatabase[16],o1,N)#右前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==11:#右后
            o1=layoutpick(o1,lag,11)
            tem=patternsearch3(Cdatabase[11],cdatabase[11],cdatabase[16],o1,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==12:
            o1=layoutpick(o1,lag,12)
            tem=patternsearch3(Cdatabase[12],cdatabase[12],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
        if d2[n]==13:
            o1=layoutpick(o1,lag,13)
            tem=patternsearch3(Cdatabase[13],cdatabase[13],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==14:
            o1=layoutpick(o1,lag,14)
            tem=patternsearch3(Cdatabase[14],cdatabase[14],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==15:
            o1=layoutpick(o1,lag,15)
            tem=patternsearch3(Cdatabase[15],cdatabase[15],cdatabase[16],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前      
    return m



def goolesuck(tem,database,a,b,N):#a b 都是list
    #print a,b
    ccc=list(set(a).intersection(set(b)))
    #print ccc
    if len(ccc)==0:
       ccc=list(set(a).union(set(b)))
    d=[]
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    for n in range(len(ccc)):
        tem2=database[ccc[n]]
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)
    r=random.randint(0,len(si)-1)
    t=ccc[r]
    return database[ccc[r]]

def goolesuck2(tem,database,a,b,c,N):#a b c都是list
    bbb=list(set(a).intersection(set(b)))
    if len(bbb)==0:
       bbb=list(set(a).union(set(b)))
    ccc=list(set(bbb).intersection(set(c)))
    if len(ccc)==0:
       ccc=list(set(bbb).union(set(c)))
    d=[]
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    for n in range(len(ccc)):
        tem2=database[ccc[n]]
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)
    r=random.randint(0,len(si)-1)
    t=ccc[r]
    return database[ccc[r]]
def goolesuck3(tem,database,a,b,c,d,N):#a b c d都是list
    aaa=list(set(a).intersection(set(b)))
    if len(aaa)==0:
       aaa=list(set(a).union(set(b)))
    bbb=list(set(aaa).intersection(set(c)))
    if len(bbb)==0:
       bbb=list(set(aaa).union(set(c)))
    ccc=list(set(bbb).intersection(set(d)))
    if len(ccc)==0:
       ccc=list(set(bbb).union(set(d)))
    fd=[]
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    for n in range(len(ccc)):
        tem2=database[ccc[n]]
        drill2=tem2.reshape(ss,1)
        fd.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(fd,N)
    r=random.randint(0,len(si)-1)
    t=ccc[r]
    return database[ccc[r]]
def initialGrid5(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):#非柱状初始模型构建,N为备选模式个数
    #对应改良版数据库2.0ver
    #直接用于分类好的数据库Cdatabase
    #N为备选模式个数
    #lag为重叠区宽度
    #flag为是否开启模式库分类
    d1=[]
    m,d1=lujing1(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    d2=[]
    d2=lujing1pluse(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    #print d1
    H=m.shape[0]
    hh=int((template_h-1)/2)
    s3=int(((H-1-(2*hh))/lag_h))+1
    gai=len(d2)/s3
    print (d1)
    for n in range(gai):
        #print d1[n][0],d1[n][1],d1[n][2]
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        #print o1
        #判断是哪个重叠区
        if d2[n]==0:#左后
            o1=layoutpick(o1,lag,0)
            tem=patternsearch2(Cdatabase[0],cdatabase[0],cdatabase[9],o1,N)#左后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==1:#左前
            o1=layoutpick(o1,lag,1)
            tem=patternsearch2(Cdatabase[1],cdatabase[1],cdatabase[9],o1,N)#左前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==2:#右前
            o1=layoutpick(o1,lag,2)
            tem=patternsearch2(Cdatabase[2],cdatabase[2],cdatabase[9],o1,N)#右前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==3:#右后
            o1=layoutpick(o1,lag,3)
            tem=patternsearch2(Cdatabase[3],cdatabase[3],cdatabase[9],o1,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==4:
            o1=layoutpick(o1,lag,4)
            tem=patternsearch2(Cdatabase[4],cdatabase[4],cdatabase[9],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
        if d2[n]==5:
            o1=layoutpick(o1,lag,5)
            tem=patternsearch2(Cdatabase[5],cdatabase[5],cdatabase[9],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==6:
            o1=layoutpick(o1,lag,6)
            tem=patternsearch2(Cdatabase[6],cdatabase[6],cdatabase[9],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==7:
            o1=layoutpick(o1,lag,7)
            tem=patternsearch2(Cdatabase[7],cdatabase[7],cdatabase[9],o1,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前
    for n in range(gai,len(d1)):
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        op=layoutpick(o1,lag,16)
        bianhao=patternsearchDi(Cdatabase[8],cdatabase[8],op)
        if d2[n]==8:#左后
            o2=layoutpick(o1,lag,0)
            bianhao2=patternsearchDi(Cdatabase[0],cdatabase[0],o2)
            o1=layoutpick(o1,lag,8)
            tem=goolesuck(o1,cdatabase[9],bianhao,bianhao2,N)#左后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==9:#左前
            o2=layoutpick(o1,lag,1)
            bianhao2=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o1=layoutpick(o1,lag,9)
            tem=goolesuck(o1,cdatabase[9],bianhao,bianhao2,N)#左前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==10:#右前
            o2=layoutpick(o1,lag,2)
            bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o1=layoutpick(o1,lag,10)
            tem=goolesuck(o1,cdatabase[9],bianhao,bianhao2,N)#右前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==11:#右后
            o2=layoutpick(o1,lag,3)
            bianhao2=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,11)
            tem=goolesuck(o1,cdatabase[9],bianhao,bianhao2,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==12:
            o2=layoutpick(o1,lag,4)
            bianhao2=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
            o1=layoutpick(o1,lag,12)
            tem=goolesuck(o1,cdatabase[9],bianhao,bianhao2,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
        if d2[n]==13:
            o2=layoutpick(o1,lag,5)
            bianhao2=patternsearchDi(Cdatabase[5],cdatabase[5],o2)
            o1=layoutpick(o1,lag,13)
            tem=goolesuck(o1,cdatabase[9],bianhao,bianhao2,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==14:
            o2=layoutpick(o1,lag,6)
            bianhao2=patternsearchDi(Cdatabase[6],cdatabase[6],o2)
            o1=layoutpick(o1,lag,14)
            tem=goolesuck(o1,cdatabase[9],bianhao,bianhao2,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==15:
            o2=layoutpick(o1,lag,7)
            bianhao2=patternsearchDi(Cdatabase[7],cdatabase[7],o2)
            o1=layoutpick(o1,lag,15)
            tem=goolesuck(o1,cdatabase[9],bianhao,bianhao2,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前    
    return m

def initialGrid6(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):#非柱状初始模型构建,N为备选模式个数
    #对应改良版数据库3.0ver
    #直接用于分类好的数据库Cdatabase
    #N为备选模式个数
    #lag为重叠区宽度
    #flag为是否开启模式库分类
    d1=[]
    temoo=np.zeros((template_h,template_x,template_y),int)
    m,d1=lujing1(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    d2=[]
    d2=lujing1pluse(m,d1,template_h,template_x,template_y,lag_h,lag_x,lag_y)
    #print d1
    H=m.shape[0]
    hh=int((template_h-1)/2)
    s3=int(((H-1-(2*hh))/lag_h))+1
    gai=len(d2)/s3
    #print d1
    for n in range(gai):
        #print d1[n][0],d1[n][1],d1[n][2]
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        #print o1
        #判断是哪个重叠区
        if d2[n]==0:#左后
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,19)
            bianhao2=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,0)
            tem=goolesuck(o1,cdatabase[5],bianhao1,bianhao2,N)#左后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==1:#左前
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,20)
            bianhao2=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
            o1=layoutpick(o1,lag,1)
            tem=goolesuck(o1,cdatabase[5],bianhao1,bianhao2,N)#左前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==2:#右前
            o2=layoutpick(o1,lag,18)
            bianhao1=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o2=layoutpick(o1,lag,20)
            bianhao2=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
            o1=layoutpick(o1,lag,2)
            tem=goolesuck(o1,cdatabase[5],bianhao1,bianhao2,N)#右前
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==3:#右后
            o2=layoutpick(o1,lag,18)
            bianhao1=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o2=layoutpick(o1,lag,19)
            bianhao2=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,3)
            tem=goolesuck(o1,cdatabase[5],bianhao1,bianhao2,N)#右后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==4:
            o2=layoutpick(o1,lag,18)
            bianhao1=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o2=layoutpick(o1,lag,20)
            bianhao2=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
            o2=layoutpick(o1,lag,19)
            bianhao3=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,4)
            tem=goolesuck2(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,N)#右前后
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        if d2[n]==5:
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,20)
            bianhao2=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
            o2=layoutpick(o1,lag,19)
            bianhao3=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,5)
            tem=goolesuck2(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
        if d2[n]==6:
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,18)
            bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o2=layoutpick(o1,lag,19)
            bianhao3=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,6)
            tem=goolesuck2(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
        if d2[n]==7:
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,18)
            bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o2=layoutpick(o1,lag,20)
            bianhao3=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
            o1=layoutpick(o1,lag,7)
            tem=goolesuck2(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,N)
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前
    for n in range(gai,len(d1)):
        o1=template1(m,template_h,template_x,template_y,d1[n][0],d1[n][1],d1[n][2])#待模拟区
        op=layoutpick(o1,lag,16)
        if op.all==0:
            tem=temoo
            m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
        else:
            bianhao1=patternsearchDi(Cdatabase[0],cdatabase[0],op)
            if d2[n]==8:#左后
            
                o2=layoutpick(o1,lag,17)
                bianhao2=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
                o2=layoutpick(o1,lag,19)
                bianhao3=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
                o1=layoutpick(o1,lag,8)
                tem=goolesuck2(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,N)
                m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
            if d2[n]==9:#左前
            
                o2=layoutpick(o1,lag,17)
                bianhao2=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
                o2=layoutpick(o1,lag,20)
                bianhao3=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
                o1=layoutpick(o1,lag,9)
                tem=goolesuck2(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,N)
                m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
            if d2[n]==10:#右前
            
                o2=layoutpick(o1,lag,18)
                bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
                o2=layoutpick(o1,lag,20)
                bianhao3=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
                o1=layoutpick(o1,lag,10)
                tem=goolesuck2(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,N)
                m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
            if d2[n]==11:#右后
            
                o2=layoutpick(o1,lag,18)
                bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
                o2=layoutpick(o1,lag,19)
                bianhao3=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
                o1=layoutpick(o1,lag,11)
                tem=goolesuck2(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,N)
                m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])
            if d2[n]==12:
            
                o2=layoutpick(o1,lag,18)
                bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
                o2=layoutpick(o1,lag,20)
                bianhao3=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
                o2=layoutpick(o1,lag,19)
                bianhao4=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
                o1=layoutpick(o1,lag,12)
                tem=goolesuck3(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,bianhao4,N)
                m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#右前后
            if d2[n]==13:
            
                o2=layoutpick(o1,lag,17)
                bianhao2=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
                o2=layoutpick(o1,lag,20)
                bianhao3=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
                o2=layoutpick(o1,lag,19)
                bianhao4=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
                o1=layoutpick(o1,lag,13)
                tem=goolesuck3(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,bianhao4,N)
                m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左前后
            if d2[n]==14:
            
                o2=layoutpick(o1,lag,18)
                bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
                o2=layoutpick(o1,lag,17)
                bianhao3=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
                o2=layoutpick(o1,lag,19)
                bianhao4=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
                o1=layoutpick(o1,lag,14)
                tem=goolesuck3(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,bianhao4,N)
                m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右后
            if d2[n]==15:
            
                o2=layoutpick(o1,lag,18)
                bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
                o2=layoutpick(o1,lag,20)
                bianhao3=patternsearchDi(Cdatabase[4],cdatabase[4],o2)
                o2=layoutpick(o1,lag,17)
                bianhao4=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
                o1=layoutpick(o1,lag,12)
                tem=goolesuck3(o1,cdatabase[5],bianhao1,bianhao2,bianhao3,bianhao4,N)
                m=template1R(m,tem,d1[n][0],d1[n][1],d1[n][2])#左右前    
    return m


def initialGridzhu1(m,template_x,template_y,lag,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N):#柱状初始模型构建,N为备选模式个数
    #对应柱状数据库

    #N为备选模式个数
    #lag为重叠区宽度

    d1=[]
    temoo=np.zeros((m.shape[0],template_x,template_y),int)
    m,d1=lujing2(m,d1,template_x,template_y,lag_x,lag_y)
    d2=[]
    d2=lujing2pluse(m,d1,template_x,template_y,lag_x,lag_y)
    H=m.shape[0]
    for n in range(len(d1)):
        #print d1[n][0],d1[n][1],d1[n][2]
        o1=template2(m,template_x,template_y,d1[n][0],d1[n][1])#待模拟区
        #print o1
        #判断是哪个重叠区
        if d2[n]==0:#左后
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[0],cdatabase[0],o2)
            o2=layoutpick(o1,lag,19)
            bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o1=layoutpick(o1,lag,0)
            tem=goolesuck(o1,cdatabase[4],bianhao1,bianhao2,N)#左后
            m=template2R(m,tem,d1[n][0],d1[n][1])
        if d2[n]==1:#左前
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[0],cdatabase[0],o2)
            o2=layoutpick(o1,lag,20)
            bianhao2=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,1)
            tem=goolesuck(o1,cdatabase[4],bianhao1,bianhao2,N)#左前
            m=template2R(m,tem,d1[n][0],d1[n][1])
        if d2[n]==2:#右前
            o2=layoutpick(o1,lag,18)
            bianhao1=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,20)
            bianhao2=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,2)
            tem=goolesuck(o1,cdatabase[4],bianhao1,bianhao2,N)#右前
            m=template2R(m,tem,d1[n][0],d1[n][1])
        if d2[n]==3:#右后
            o2=layoutpick(o1,lag,18)
            bianhao1=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,19)
            bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o1=layoutpick(o1,lag,3)
            tem=goolesuck(o1,cdatabase[4],bianhao1,bianhao2,N)#右后
            m=template2R(m,tem,d1[n][0],d1[n][1])
        if d2[n]==4:
            o2=layoutpick(o1,lag,18)
            bianhao1=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,20)
            bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o2=layoutpick(o1,lag,19)
            bianhao3=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,4)
            tem=goolesuck2(o1,cdatabase[4],bianhao1,bianhao2,bianhao3,N)#右前后
            m=template2R(m,tem,d1[n][0],d1[n][1])
        if d2[n]==5:
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[0],cdatabase[0],o2)
            o2=layoutpick(o1,lag,20)
            bianhao2=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o2=layoutpick(o1,lag,19)
            bianhao3=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,5)
            tem=goolesuck2(o1,cdatabase[4],bianhao1,bianhao2,bianhao3,N)
            m=template2R(m,tem,d1[n][0],d1[n][1])#左前后
        if d2[n]==6:
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[0],cdatabase[0],o2)
            o2=layoutpick(o1,lag,18)
            bianhao2=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,19)
            bianhao3=patternsearchDi(Cdatabase[2],cdatabase[2],o2)
            o1=layoutpick(o1,lag,6)
            tem=goolesuck2(o1,cdatabase[4],bianhao1,bianhao2,bianhao3,N)
            m=template2R(m,tem,d1[n][0],d1[n][1])#左右后
        if d2[n]==7:
            o2=layoutpick(o1,lag,17)
            bianhao1=patternsearchDi(Cdatabase[0],cdatabase[0],o2)
            o2=layoutpick(o1,lag,18)
            bianhao2=patternsearchDi(Cdatabase[1],cdatabase[1],o2)
            o2=layoutpick(o1,lag,20)
            bianhao3=patternsearchDi(Cdatabase[3],cdatabase[3],o2)
            o1=layoutpick(o1,lag,7)
            tem=goolesuck2(o1,cdatabase[4],bianhao1,bianhao2,bianhao3,N)
            m=template2R(m,tem,d1[n][0],d1[n][1])#左右前
      
    return m









