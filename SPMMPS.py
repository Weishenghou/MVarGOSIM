#!/usr/bin/env python
# coding: utf-8

# In[1]:


######################ver1.555
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
####################################################

def temdetectforwhat(tem,value):#检测是否包含待模拟点,自定义值ver
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]==value:
                    return False
    return True

def temdetectforwhat2(tem,value):#检测是否包含待模拟点,自定义值ver2,当该值小于3/4时返回错误
    count=0
    maxcount=(tem.shape[0])*(tem.shape[1])*(tem.shape[2])
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]==value:
                    count=count+1
    #print count
    if count<=0.8*maxcount:
       return False
    return True




#####################################################
#################################主程序########################################################
time_start1=time.time()#计时开始
time_start=time.time()#计时开始

######################################参数读取阶段################################################################



file1=open('./parameter.txt')
content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
Mh=int(''.join(string1))
#print Mh

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
Mx=int(''.join(string1))
#print Mx

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
My=int(''.join(string1))
#print My

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
lag=int(''.join(string1))
#print lag

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
lag_h=int(''.join(string1))
#print lag_h

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
lag_x=int(''.join(string1))
#print lag_x

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
lag_y=int(''.join(string1))
#print lag_y

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
patternSizeh=int(''.join(string1))
#print patternSizeh

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
patternSizex=int(''.join(string1))
#print patternSizex

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
patternSizey=int(''.join(string1))
#print patternSizey

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
U=int(''.join(string1))
#print U

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
N=int(''.join(string1))
#print N

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
size=int(''.join(string1))
#print size

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
itr=int(''.join(string1))
#print itr


content=file1.readline()
scale=[]
for i in content:
    if str.isdigit(i):
        scale.append(int(i))
#print scale

#一次模拟的个数/模拟并行核数
content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
Modelcount=int(''.join(string1))
#print Modelcount

##########################################################预处理阶段################################################################

yvalue=33#设置球粒陨石的值
jivalue=115#设置基质值
#构建初始网格
m=-np.ones((Mh,Mx,My),int)#默认空值为-1
GosimAIyunshi2(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,yvalue,jivalue,Modelcount)

