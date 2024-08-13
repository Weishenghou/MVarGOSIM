#!/usr/bin/env python
# coding: utf-8

# In[7]:

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

########################################工具函数集#############################################

def maxlist(listss):#列表频数最高返回
    temp = 0
    for i in listss:
        if listss.count(i) > temp:
            asa = i
            temp = listss.count(i)
    return asa

def sehaming(tem,flag):
    #将单个模板转换为单行矩阵,以便求解汉明距离
    th=tem.shape[0]
    l=tem.shape[1]
    tem2=np.zeros([1,th*l],np.uint8)
    if flag==0:#顺序
        n=0
        for n1 in range(l):
            for n2 in range(th):
                tem2[:,n]=tem[n2,n1]
                n=n+1
    if flag==1:#倒序
        n=0
        for n1 in range(-l+1,1):
            for n2 in range(th):
                tem2[:,n]=tem[n2,-n1]
                n=n+1
    return tem2

'''
def template1(Ti,template_h,template_x,template_y,h0,x0,y0):#非柱状模板
    tem=np.zeros((template_h,template_x,template_y),np.uint8)
    nn1=0
    nn2=0
    nn3=0
    hh=int((template_h-1)/2)
    xx=int((template_x-1)/2)
    yy=int((template_y-1)/2)
    for n1 in range(h0-hh,h0+hh+1):
        for n2 in range(x0-xx,x0+xx+1):
            for n3 in range(y0-yy,y0+yy+1):
                tem[nn1,nn2,nn3]=Ti[n1,n2,n3]
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
            #print(nn2)
        nn1=nn1+1
        nn2=0
        #print(nn1)
    return tem  #提取坐标h0,x0,y0处模板
'''
def template1(Ti,template_h,template_x,template_y,h0,x0,y0):#非柱状模板
    ph=template_h//2
    px=template_x//2
    py=template_y//2
    tem = Ti[h0-ph:h0+ph+1, x0-px:x0+px+1, y0-py:y0+py+1]
    return tem  #提取坐标h0,x0,y0处模板
def template1R(Ti,tem,h0,x0,y0):#非柱状模板返还
    ph=tem.shape[0]//2
    px=tem.shape[1]//2
    py=tem.shape[2]//2
    Ti[h0-ph:h0+ph+1, x0-px:x0+px+1, y0-py:y0+py+1]=tem 
    return Ti  
def template1RAI(Ti,tem,h0,x0,y0):#非柱状模板返还判断硬数据版
    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    nn1=0
    nn2=0
    nn3=0
    hh=int((template_h-1)/2)
    xx=int((template_x-1)/2)
    yy=int((template_y-1)/2)
    for n1 in range(h0-hh,h0+hh+1):
        for n2 in range(x0-xx,x0+xx+1):
            for n3 in range(y0-yy,y0+yy+1):
                if Ti[n1,n2,n3]==-1:
                    #if tem[nn1,nn2,nn3]!=30 and tem[nn1,nn2,nn3]!=0 :
                    Ti[n1,n2,n3]=tem[nn1,nn2,nn3]
                    #else:
                       #Ti[n1,n2,n3]=160
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
        nn1=nn1+1
        nn2=0

        #print(nn2)
    return Ti  #提取坐标x0,y0处模板 
def template2RAI(Ti,tem,x0,y0):#柱状模板返还
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    nn2=0
    nn3=0
    xx=int((template_x-1)/2)
    yy=int((template_y-1)/2)
    for n2 in range(x0-xx,x0+xx+1):
        for n3 in range(y0-yy,y0+yy+1):
            if Ti[:,n2,n3].all()==-1:
       
               Ti[:,n2,n3]=tem[:,nn2,nn3]
            nn3=nn3+1
            #print(nn3)
        nn2=nn2+1
        nn3=0
        #print(nn2)
    return Ti  #提取坐标x0,y0处模板
'''
def template1R(Ti,tem,h0,x0,y0):#非柱状模板返还
    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    nn1=0
    nn2=0
    nn3=0
    hh=int((template_h-1)/2)
    xx=int((template_x-1)/2)
    yy=int((template_y-1)/2)
    for n1 in range(h0-hh,h0+hh+1):
        for n2 in range(x0-xx,x0+xx+1):
            for n3 in range(y0-yy,y0+yy+1):
                Ti[n1,n2,n3]=tem[nn1,nn2,nn3]
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
            #print(nn2)
        nn1=nn1+1
        nn2=0
        #print(nn1)
    return Ti  #提取坐标h0,x0,y0处模板
'''
def template2(Ti,template_x,template_y,x0,y0):#柱状模板
    tem=np.zeros((Ti.shape[0],template_x,template_y),np.uint8)
    nn2=0
    nn3=0
    xx=int((template_x-1)/2)
    yy=int((template_y-1)/2)
    for n2 in range(x0-xx,x0+xx+1):
        for n3 in range(y0-yy,y0+yy+1):
            tem[:,nn2,nn3]=Ti[:,n2,n3]
            nn3=nn3+1
            #print(nn3)
        nn2=nn2+1
        nn3=0
        #print(nn2)
    return tem  #提取坐标x0,y0处模板

def template2R(Ti,tem,x0,y0):#柱状模板返还
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    nn2=0
    nn3=0
    xx=int((template_x-1)/2)
    yy=int((template_y-1)/2)
    for n2 in range(x0-xx,x0+xx+1):
        for n3 in range(y0-yy,y0+yy+1):
            Ti[:,n2,n3]=tem[:,nn2,nn3]
            nn3=nn3+1
            #print(nn3)
        nn2=nn2+1
        nn3=0
        #print(nn2)
    return Ti  #提取坐标x0,y0处模板

def templatedatabasec1(Ti,database,zuobiaolist,template_h,template_x,template_y):#对非柱模板 database为模板库，zuobiaolist为坐标库，都为list格式
    #此处的Ti未导入网格中，坐标为Ti本身坐标
    wid=int((max(template_h,template_x,template_y)-1)/2)
    Ti_pad=np.pad(Ti,wid,'edge')#直接选取最大的作为padding的宽度
    hh=Ti.shape[0]
    xx=Ti.shape[1]
    yy=Ti.shape[2]
    nn=0
    for n1 in range(wid,hh+wid):
        for n2 in range(wid,xx+wid):
            for n3 in range(wid,yy+wid):
                database.append(template1(Ti_pad,template_h,template_x,template_y,n1,n2,n3))
                zuobiaolist.append((n1-wid,n2-wid,n3-wid))
                nn=nn+1
                #print(nn)

                
def templatedatabasec2(Ti,database,zuobiaolist,template_h,template_x,template_y,m,flag):
    #对zuobiaolist进行转换,flag表示东西南北四个方向,1为竖（西，2为竖（东，3为横（北，4为横（南,m为模拟网格
    wid=int((max(template_h,template_x,template_y)-1)/2)
    Ti_pad=np.pad(Ti,wid,'edge')#直接选取最大的作为padding的宽度
    hh=Ti.shape[0]
    xx=Ti.shape[1]
    yy=Ti.shape[2]
    nn=0
    hh1=m.shape[0]
    xx1=m.shape[1]
    yy1=m.shape[2]
    if flag==1:#为西侧的剖面时
        for n1 in range(wid,hh+wid):
            for n2 in range(wid,xx+wid):
                for n3 in range(wid,yy+wid):
                    Tpl=template1(Ti_pad,template_h,template_x,template_y,n1,n2,n3)
                    database.append(Tpl)
                    zuobiaolist.append((n1-wid,n2-wid,n3-wid))
                    nn=nn+1
                    #print(nn)
    if flag==2:#为东侧的剖面时
        for n1 in range(wid,hh+wid):
            for n2 in range(wid,xx+wid):
                for n3 in range(wid,yy+wid):
                    Tpl=template1(Ti_pad,template_h,template_x,template_y,n1,n2,n3)
                    database.append(Tpl)
                    zuobiaolist.append((n1-wid,n2-wid+xx1-xx,n3-wid))
                    nn=nn+1
                    #print(nn)
    if flag==3:#为北侧的剖面时
        for n1 in range(wid,hh+wid):
            for n2 in range(wid,xx+wid):
                for n3 in range(wid,yy+wid):
                    database.append(template1(Ti_pad,template_h,template_x,template_y,n1,n2,n3))
                    zuobiaolist.append((n1-wid,n2-wid,n3-wid))
                    nn=nn+1
                    #print(nn)
    if flag==4:#为南侧的剖面时
        for n1 in range(wid,hh+wid):
            for n2 in range(wid,xx+wid):
                for n3 in range(wid,yy+wid):
                    database.append(template1(Ti_pad,template_h,template_x,template_y,n1,n2,n3))
                    zuobiaolist.append((n1-wid,n2-wid,n3-wid+yy1-yy))
                    nn=nn+1
                    #print(nn)
                    
                    
def templatedatabasec3(Ti,database,zuobiaolist,template_x,template_y,m,flag):
    #柱状模板的模式库建立
    #对zuobiaolist进行转换,flag表示东西南北四个方向,1为竖（西，2为竖（东，3为横（北，4为横（南,m为模拟网格
    wid=int((max(template_x,template_y)-1)/2)
    Ti_pad=np.pad(Ti,((0,0),(wid,wid),(wid,wid)),'edge')#直接选取最大的作为padding的宽度
    xx=Ti.shape[1]
    yy=Ti.shape[2]
    nn=0
    xx1=m.shape[1]
    yy1=m.shape[2]
    if flag==1:#为西侧的剖面时
        for n2 in range(wid,xx+wid):
            for n3 in range(wid,yy+wid):
                Tpl=template2(Ti_pad,template_x,template_y,n2,n3)
                database.append(Tpl)
                zuobiaolist.append((n2-wid,n3-wid))
                nn=nn+1
                #print(nn)
    if flag==2:#为东侧的剖面时
        for n2 in range(wid,xx+wid):
            for n3 in range(wid,yy+wid):
                Tpl=template2(Ti_pad,template_x,template_y,n2,n3)
                database.append(Tpl)
                zuobiaolist.append((n2-wid+xx1-xx,n3-wid))
                nn=nn+1
                #print(nn)
    if flag==3:#为北侧的剖面时
        for n2 in range(wid,xx+wid):
            for n3 in range(wid,yy+wid):
                database.append(template2(Ti_pad,template_x,template_y,n2,n3))
                zuobiaolist.append((n2-wid,n3-wid))
                nn=nn+1
                #print(nn)
    if flag==4:#为南侧的剖面时
        for n2 in range(wid,xx+wid):
            for n3 in range(wid,yy+wid):
                database.append(template2(Ti_pad,template_x,template_y,n2,n3))
                zuobiaolist.append((n2-wid,n3-wid+yy1-yy))
                nn=nn+1
                #print(nn)

def templatedatabasec4(Ti,database,zuobiaolist,template_h,template_x,template_y,m,flag):
    #对zuobiaolist进行转换,flag表示东西南北四个方向,1为竖（西，2为竖（东，3为横（北，4为横（南,m为模拟网格
    #无padding版
    wid=max(template_h,template_x,template_y)//2
    #Ti_pad=np.pad(Ti,wid,'edge')#直接选取最大的作为padding的宽度
    hh=Ti.shape[0]
    xx=Ti.shape[1]
    yy=Ti.shape[2]
    nn=0
    hh1=m.shape[0]
    xx1=m.shape[1]
    yy1=m.shape[2]
    if flag==1:#为西侧的剖面时
        for n1 in range(wid,hh-wid):
            for n2 in range(wid,xx-wid):
                #print Ti[n1-wid:n1+wid+1,n2-wid:n2-wid+1,:].shape[2],(n1,n2,0)
                database.append(Ti[n1-wid:n1+wid+1,n2-wid:n2+wid+1,:])
                zuobiaolist.append((n1,n2,0))
                nn=nn+1
                #print(nn)
    if flag==2:#为东侧的剖面时
        for n1 in range(wid,hh-wid):
            for n2 in range(wid,xx-wid):
                #print Ti[n1-wid:n1+wid+1,n2-wid:n2-wid+1,:].shape[2],(n1,n2,yy1-1)
                database.append(Ti[n1-wid:n1+wid+1,n2-wid:n2+wid+1,:])
                zuobiaolist.append((n1,n2,yy1-1,))
                nn=nn+1
                #print(nn)
    if flag==3:#为北侧的剖面时
        for n1 in range(wid,hh-wid):
            for n3 in range(wid,yy-wid):
                #print Ti[n1-wid:n1+wid+1,:,n3-wid:n3+wid+1].shape[2]
                database.append(Ti[n1-wid:n1+wid+1,:,n3-wid:n3+wid+1])
                zuobiaolist.append((n1,0,n3))
                nn=nn+1
                #print(nn)
    if flag==4:#为南侧的剖面时
        for n1 in range(wid,hh-wid):
            for n3 in range(wid,yy-wid):
                #print Ti[n1-wid:n1+wid+1,:,n3-wid:n3+wid+1]
                database.append(Ti[n1-wid:n1+wid+1,:,n3-wid:n3+wid+1])
                zuobiaolist.append((n1,xx1-1,n3))
                nn=nn+1
                #print(nn)
    
    return database,zuobiaolist


                
def databaseshow(database,zuobiaolist):#模式库显示工具
    for n in range(len(database)):
        print(database[n])
        print(zuobiaolist[n])
        
def databasesave(database,zuobiaolist):#模式库储存工具
    np.save('patterndatabase.npy',database)
    np.save('zuobiao.npy',zuobiaolist)
    
def maxlist(list):#列表频数最高返回
    temp = 0
    for i in list:
        if list.count(i) > temp:
            asa = i
            temp = list.count(i)
    return asa


########################Ti扩展##############################################
def extend2d(m,h1,x1,y1):#9格子内随机选取一个值 
    listcs=[]
    for ss2 in range(-1,2):
        for ss3 in range(-1,2):
            c=m[h1,x1+ss2,y1+ss3]
            if c!=-1:#默认空值为-1
                listcs.append(c)
    random.shuffle(listcs)
    if len(listcs)!=0:
       value=listcs[0]
    else:
       value=0
    return value

def extendTI1(m,template_x,template_y):#非柱状扩充TI一模板单位 
    lujinglist=[]
    mm=np.zeros((m.shape[0],m.shape[1],m.shape[2]),int)
    H=m.shape[0]
    lag_x=1
    lag_y=1
    L=m.shape[1]
    W=m.shape[2]
    s1=int((L-1-(2*1))/lag_x)
    s2=int((W-1-(2*1))/lag_y)
    #print s3
    s=max(template_x,template_y)-1


    for sh in range(H):
        x=1#初始点在一个模板厚度的ti缘
        y=1
        lagn=0
        for sc in range(s):
            for y1 in range(s2-lagn):
                lujinglist.append((sh,x,y))
                y=y+lag_y
            for x1 in range(s1-lagn):
                lujinglist.append((sh,x,y))
                x=x+lag_x
            for y2 in range(s2-lagn):
                lujinglist.append((sh,x,y))
                y=y-lag_y            
            for x12 in range(s1-lagn):
                lujinglist.append((sh,x,y))
                x=x-lag_x
            #完成一圈
            x=x+lag_x
            y=y+lag_y
            lagn=lagn+2
    #print lujinglist
    for n in range(len(lujinglist)):
        m[lujinglist[n][0],lujinglist[n][1],lujinglist[n][2]]=extend2d(m,lujinglist[n][0],lujinglist[n][1],lujinglist[n][2])
            
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    for n in range(template_x):
        list1.append(n)
    for n in range(L-template_x,L):
        list2.append(n)
    for n in range(template_y):
        list3.append(n)
    for n in range(W-template_y,W):
        list4.append(n)
    Ti1=m[:,list1,:]#后
    Ti2=m[:,list2,:]#前
    Ti3=m[:,:,list3]#左
    Ti4=m[:,:,list4]#右
    mm[:,list1,:]=Ti1
    mm[:,list2,:]=Ti2
    mm[:,:,list3]=Ti3
    mm[:,:,list4]=Ti4
    return mm,Ti1,Ti2,Ti3,Ti4#返模拟网格以及用于模式提取的Tis
