# coding: utf-8

# In[ ]:


import numpy as np
import cv2
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
from patterntool import*

def temdatabasec_3(database,lag,template_x,template_y):
    #柱状模板的模式库分类
    #lag为重叠区宽度
    cadatabase=[]
    le=len(database)
    i=0
    dis=[]#后左
    disx=[]#前
    disy=[]#右
    dis0x=[]#前后
    dis0y=[]#左右
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    dis0x=list(set(dis+disx))
    dis0y=list(set(dis+disy))
    #print dis,disx,disy,dis0x,dis0y
    ss1=(database[0].shape[0])*len(dis)*(database[0].shape[1])#竖面
    ss2=database[0].shape[0]*len(dis)*database[0].shape[2]#横面
    ss3=database[0].shape[0]*len(dis0y)*database[0].shape[1]#双竖面
    ss4=database[0].shape[0]*len(dis0x)*database[0].shape[2]#双横面
    #左后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库0
        shumian=np.resize(database[s][:,:,dis],(1,ss1))
        hengmian=np.resize(database[s][:,dis,:],(1,ss2))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)
    #左前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库1
        shumian=np.resize(database[s][:,:,dis],(1,ss1))
        hengmian=np.resize(database[s][:,disx,:],(1,ss2))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)
    #右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库2
        shumian=np.resize(database[s][:,:,disy],(1,ss1))
        hengmian=np.resize(database[s][:,disx,:],(1,ss2))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)
    #右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库3
        shumian=np.resize(database[s][:,:,disy],(1,ss1))
        hengmian=np.resize(database[s][:,dis,:],(1,ss2))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)    
    #右前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库4
        shumian=np.resize(database[s][:,:,disy],(1,ss1))
        hengmian=np.resize(database[s][:,dis0x,:],(1,ss4))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)  
    #左前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库5
        shumian=np.resize(database[s][:,:,dis],(1,ss1))
        hengmian=np.resize(database[s][:,dis0x,:],(1,ss4))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)  
    #左右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库6
        shumian=np.resize(database[s][:,:,dis0y],(1,ss3))
        hengmian=np.resize(database[s][:,dis,:],(1,ss2))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)
    #左右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库7
        shumian=np.resize(database[s][:,:,dis0y],(1,ss3))
        hengmian=np.resize(database[s][:,disx,:],(1,ss2))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)  
    
    print('done')
    return cadatabase


def temdatabasec_4(database,lag,template_x,template_y):
    #柱状模板的模式库分类改良version
    #lag为重叠区宽度
    cadatabase=[]
    le=len(database)
    i=0
    dis=[]#后左
    dis2y=[]#补左
    disx=[]#前
    disy=[]#右
    disyy=[]#补右
    dis0x=[]#前后
    dis0y=[]#左右
    disc=[]
    disk=[]
    for n in range(template_x):
        disc.append(n)
    for n in range(template_y):
        disk.append(n)
    for n in range(lag):
        dis.append(n)
    for n in range(lag,template_y):##
        dis2y.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_y-lag):##
        disyy.append(n)
    dis0x=list(set(dis+disx))
    dis0y=list(set(dis+disy))
    dis0xx=list(set(dis+disx)^set(disc))#补x中间
    dis0yy=list(set(dis+disy)^set(disk))#补y中间
    #print('dis,disx,disy,dis2y,dis0x,dis0y,disyy,dis0xx,dis0yy')
    #print dis,disx,disy,dis2y,dis0x,dis0y,disyy,dis0xx,dis0yy
    ss1=database[0].shape[0]*len(dis)*database[0].shape[1]#竖面
    ss2=database[0].shape[0]*len(dis)*database[0].shape[2]#横面
    ss3=database[0].shape[0]*len(dis0y)*database[0].shape[1]#双竖面
    ss4=database[0].shape[0]*len(dis0x)*database[0].shape[2]#双横面
    ss20=database[0].shape[0]*len(dis)*len(dis2y)
    ss21=database[0].shape[0]*len(disx)*len(dis2y)
    ss22=database[0].shape[0]*len(disx)*len(disyy)
    ss23=database[0].shape[0]*len(dis)*len(disyy)
    ss24=database[0].shape[0]*len(dis0x)*len(disyy)
    ss25=database[0].shape[0]*len(dis0x)*len(dis2y)
    ss26=database[0].shape[0]*len(dis)*len(dis0yy)
    ss27=database[0].shape[0]*len(disx)*len(dis0yy)
    #左后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库0
        shumian=np.resize(database[s][:,:,dis],(1,ss1))
        teoo=database[s][:,dis,:]
        teoo2=teoo[:,:,dis2y]
        hengmian=np.resize(teoo2,(1,ss20))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)
    #左前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库1
        shumian=np.resize(database[s][:,:,dis],(1,ss1))
        teoo=database[s][:,disx,:]
        teoo2=teoo[:,:,dis2y]
        hengmian=np.resize(teoo2,(1,ss21))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)
    #右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库2
        shumian=np.resize(database[s][:,:,disy],(1,ss1))
        teoo=database[s][:,disx,:]
        teoo2=teoo[:,:,disyy]
        hengmian=np.resize(teoo2,(1,ss22))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)
    #右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库3
        shumian=np.resize(database[s][:,:,disy],(1,ss1))
        teoo=database[s][:,dis,:]
        teoo2=teoo[:,:,disyy]
        hengmian=np.resize(teoo2,(1,ss23))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)    
    #右前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库4
        shumian=np.resize(database[s][:,:,disy],(1,ss1))
        teoo=database[s][:,dis0x,:]
        teoo2=teoo[:,:,disyy]
        hengmian=np.resize(teoo2,(1,ss24))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)  
    #左前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库5
        shumian=np.resize(database[s][:,:,dis],(1,ss1))
        teoo=database[s][:,dis0x,:]
        teoo2=teoo[:,:,dis2y]
        hengmian=np.resize(teoo2,(1,ss25))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)  
    #左右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库6
        shumian=np.resize(database[s][:,:,dis0y],(1,ss3))
        teoo=database[s][:,dis,:]
        teoo2=teoo[:,:,dis0yy]
        hengmian=np.resize(teoo2,(1,ss26))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)
    #左右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库7
        shumian=np.resize(database[s][:,:,dis0y],(1,ss3))
        teoo=database[s][:,disx,:]
        teoo2=teoo[:,:,dis0yy]
        hengmian=np.resize(teoo2	,(1,ss27))
        t=np.append(shumian,hengmian)
        d1.append(t)
    cadatabase.append(d1)  
    
    #数据库8原始数据库
    cadatabase.append(database)  
    
    print('done')
    return cadatabase





def temdatabasec_5(database,lag,template_h,template_x,template_y):
    #普通模板的模式库分类
    #lag为重叠区宽度
    cadatabase=[]
    le=len(database)
    i=0
    dis=[]#后左
    disx=[]#前
    disy=[]#右
    dis0x=[]#前后
    dis0y=[]#左右
    disc=[]
    disk=[]
    dish=[]#下
    for n in range(template_x):
        disc.append(n)
    for n in range(template_y):
        disk.append(n)
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
    dis0x=list(set(dis+disx))
    dis0y=list(set(dis+disy))
    #左后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库0
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库1
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库2
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库3
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)    
    #右前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库4
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,dis0x,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库5
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,dis0x,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库6
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis0y]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #左右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库7
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis0y]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    ######################################有下重叠区
    #左后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库8
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库9
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库10
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,disy]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库11
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,disy]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)    
    #右前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库12
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,disy]=1
        b[:,dis0x,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库13
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis]=1
        b[:,dis0x,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库14
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis0y]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #左右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库15
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis0y]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    cadatabase.append(database)#数据库16为本体
    print('done')
    return cadatabase

def temdatabasec_6(database,lag,template_h,template_x,template_y):
    #普通模板的模式库分类
    #lag为重叠区宽度
    cadatabase=[]
    le=len(database)
    #i=0
    dis=[]#后左
    disx=[]#前
    disy=[]#右
    dis0x=[]#前后
    dis0y=[]#左右
    disc=[]
    disk=[]
    dish=[]#下
    for n in range(template_x):
        disc.append(n)
    for n in range(template_y):
        disk.append(n)
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
    dis0x=list(set(dis+disx))
    dis0y=list(set(dis+disy))
    #左后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库0
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库1
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库2
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库3
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)    
    #右前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库4
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,dis0x,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左前后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库5
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,dis0x,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左右后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库6
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis0y]=1
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #左右前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库7
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis0y]=1
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    ######################################有下重叠区
    d1=[]
    for s in range(le):#遍历模式库
        #数据库8
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    
    cadatabase.append(database)#数据库9为本体
    print('done')
    return cadatabase


def temdatabasec_7(database,lag,template_h,template_x,template_y):
    #普通模板的模式库分类
    #仅仅提取五个面
    #lag为重叠区宽度
    cadatabase=[]
    le=len(database)
    dis=[]#后左
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

    #下
    d1=[]
    for s in range(le):#遍历模式库
        #数据库0
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1        
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
    #左
    d1=[]
    for s in range(le):#遍历模式库
        #数据库1
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #右
    d1=[]
    for s in range(le):#遍历模式库
        #数据库2
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库3
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)    
    #前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库4
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
  
    
    cadatabase.append(database)#数据库5为本体
    print('done')
    return cadatabase

def temdatabasec_8(database,lag,template_x,template_y):
    #柱状模板的模式库分类
    #仅仅提取四个面
    #lag为重叠区宽度
    cadatabase=[]
    le=len(database)
    template_h=database[0].shape[0]
    dis=[]#后左
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


    #左
    d1=[]
    for s in range(le):#遍历模式库
        #数据库0
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #右
    d1=[]
    for s in range(le):#遍历模式库
        #数据库1
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)
    #后
    d1=[]
    for s in range(le):#遍历模式库
        #数据库2
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,dis,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)    
    #前
    d1=[]
    for s in range(le):#遍历模式库
        #数据库3
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,disx,:]=1
        t=database[s]*b
        d1.append(t)
    cadatabase.append(d1)  
  
    
    cadatabase.append(database)#数据库4为本体
    print('done')
    return cadatabase

def layoutpick(tem,lag,flag):
    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    dis=[]#后左
    disx=[]#前
    disy=[]#右
    dis0x=[]#前后
    dis0y=[]#左右
    #disc=[]
    #disk=[]
    dish=[]
    '''
    for n in range(template_x):
        disc.append(n)
    for n in range(template_y):
        disk.append(n)
    '''
    for n in range(lag):
        dis.append(n)
    for n in range(template_x-lag,template_x):
        disx.append(n)
    for n in range(template_y-lag,template_y):
        disy.append(n)
    for n in range(template_h-lag,template_h):
        dish.append(n)
    dis0x=list(set(dis+disx))
    dis0y=list(set(dis+disy))
    if flag==0:
        #数据库0
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,dis,:]=1
        t=tem*b
    #左前  
    if flag==1:
        #数据库1
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,disx,:]=1
        t=tem*b
    #右前
    if flag==2:
        #数据库2
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,disx,:]=1
        t=tem*b
    #右后

    if flag==3:
        #数据库3
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,dis,:]=1
        t=tem*b
        
    #右前后
    if flag==4:
        #数据库4
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        b[:,dis0x,:]=1
        t=tem*b
         
    #左前后
    if flag==5:
        #数据库5
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        b[:,dis0x,:]=1
        t=tem*b
          
    #左右后
    if flag==6:
        #数据库6
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis0y]=1
        b[:,dis,:]=1
        t=tem*b
        
    #左右前
    if flag==7:
        #数据库7
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis0y]=1
        b[:,disx,:]=1
        t=tem*b
         
    ######################################有下重叠区
    if flag==8:
        #数据库8
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis]=1
        b[:,dis,:]=1
        t=tem*b
          
    #左前
    if flag==9:
        #数据库9
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis]=1
        b[:,disx,:]=1
        t=tem*b
        
    #右前
    if flag==10:
        #数据库10
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,disy]=1
        b[:,disx,:]=1
        t=tem*b
        
    #右后
    if flag==11:
        #数据库11
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,disy]=1
        b[:,dis,:]=1
        t=tem*b
          
    #右前后
    if flag==12:
        #数据库12
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,disy]=1
        b[:,dis0x,:]=1
        t=tem*b
         
    #左前后
    if flag==13:
        #数据库13
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis]=1
        b[:,dis0x,:]=1
        t=tem*b
         
    #左右后
    if flag==14:
        #数据库14
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis0y]=1
        b[:,dis,:]=1
        t=tem*b
    
    #左右前
    if flag==15:
        #数据库15
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        b[:,:,dis0y]=1
        b[:,disx,:]=1
        t=tem*b





    #下
    if flag==16:
        #只取最底层  0
        b=np.zeros((template_h,template_x,template_y),int)
        b[dish,:,:]=1
        t=tem*b

    if flag==17:
        #只取左  1
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,dis]=1
        t=tem*b
    if flag==18:
        #只取右  2
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,:,disy]=1
        t=tem*b
    if flag==19:
        #只取后  3
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,dis,:]=1
        t=tem*b
    if flag==20:
        #只取前  4
        b=np.zeros((template_h,template_x,template_y),int)
        b[:,disx,:]=1
        t=tem*b

    return t
