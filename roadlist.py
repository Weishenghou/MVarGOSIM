#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab
import time
from PIL import Image
from pathlib2 import Path #python3环境下
#from pathlib import Path  #python2环境下
import os
import random

def lujing1(m,lujinglist,template_h,template_x,template_y,lag_h,lag_x,lag_y):#非柱状圈回路径
    #lag为移动距离
    #template_x与template_y,lag_x与lag_y,默认统一奇偶，如果出问题原因就在此
    H=m.shape[0]
    hh=int((template_h-1)/2)
    L=m.shape[1]
    W=m.shape[2]
    s1=int((L-1-(2*template_x))/lag_x)
    s2=int((W-1-(2*template_y))/lag_y)
    s3=int(((H-1-(2*hh))/lag_h))+1
    #print hh,s3
    s=min((L-(2*template_x))/2,(W-(2*template_y))/2)
    h=H-hh-1
    if (L%2)==0:#当输入为偶数时
        for sh in range(s3):#h方向倒着取
            x=template_x#初始点在一个模板厚度的ti缘
            y=template_y
            lagn=0
            for sc in range(s):
                for y1 in range(s2-lagn):
                    if x<(L-template_x) and y<(W-template_y):
                       lujinglist.append((h,x,y))
                       y=y+lag_y
                for x1 in range(s1-lagn):
                    if x<(L-template_x) and y<(W-template_y):
                       lujinglist.append((h,x,y))
                       x=x+lag_x
                for y2 in range(s2-lagn):
                    if x<(L-template_x) and y<(W-template_y):
                       lujinglist.append((h,x,y))
                       y=y-lag_y            
                for x12 in range(s1-lagn):
                    if x<(L-template_x) and y<(W-template_y):
                       lujinglist.append((h,x,y))
                       x=x-lag_x
                #完成一圈
                x=x+lag_x
                y=y+lag_y
                lagn=lagn+2
            h=h-lag_h
    else:
        for sh in range(s3):
            x=template_x#初始点在一个模板厚度的ti缘
            y=template_y
            lagn=0
            for sc in range(s):
                for y1 in range(s2-lagn):
                    if x<(L-template_x) and y<(W-template_y):
                       lujinglist.append((h,x,y))
                       y=y+lag_y
                for x1 in range(s1-lagn):
                    if x<(L-template_x) and y<(W-template_y):
                       lujinglist.append((h,x,y))
                       x=x+lag_x
                for y2 in range(s2-lagn):
                    if x<(L-template_x) and y<(W-template_y):
                       lujinglist.append((h,x,y))
                       y=y-lag_y            
                for x12 in range(s1-lagn):
                    if x<(L-template_x) and y<(W-template_y):
                       lujinglist.append((h,x,y))
                       x=x-lag_x
                #完成一圈
                #print x,y
                x=x+lag_x
                y=y+lag_y
                lagn=lagn+2
            #x=L/2
            #y=W/2
            #lujinglist.append((h,x,y))
            h=h-lag_h
    list1=sorted(set(lujinglist),key=lujinglist.index) # sorted output
    lujinglist=list1
    return m,lujinglist




def lujing2(m,lujinglist,template_x,template_y,lag_x,lag_y):#柱状圈回路径
    #lag为移动距离
    #template_x与template_y默认统一奇偶
    L=m.shape[1]
    W=m.shape[2]
    x=template_x#初始点在一个模板厚度的ti缘
    y=template_y
    s1=int((L-1-(2*template_x))/lag_x)
    s2=int((W-1-(2*template_y))/lag_y)
    s=min((L-(2*template_x))/2,(W-(2*template_y))/2)
    print (s)
    lagn=0
    if (L%2)==0:#当输入为偶数时
        for sc in range(s):
            for y1 in range(s2-lagn):
                lujinglist.append((x,y))
                y=y+lag_y
            for x1 in range(s1-lagn):
                lujinglist.append((x,y))
                x=x+lag_x
            for y2 in range(s2-lagn):
                lujinglist.append((x,y))
                y=y-lag_y            
            for x12 in range(s1-lagn):
                lujinglist.append((x,y))
                x=x-lag_x
            #完成一圈
            x=x+lag_x
            y=y+lag_y
            lagn=lagn+2
    else:
        for sc in range(s):
            for y1 in range(s2-lagn):
                lujinglist.append((x,y))
                y=y+lag_y
            for x1 in range(s1-lagn):
                lujinglist.append((x,y))
                x=x+lag_x
            for y2 in range(s2-lagn):
                lujinglist.append((x,y))
                y=y-lag_y            
            for x12 in range(s1-lagn):
                lujinglist.append((x,y))
                x=x-lag_x
            #完成一圈
            x=x+lag_x
            y=y+lag_y
            lagn=lagn+2
        x=L/2
        y=W/2
        lujinglist.append((x,y))
    list1=sorted(set(lujinglist),key=lujinglist.index) # sorted output
    lujinglist=list1
    #print lujinglist
    return m,lujinglist


def lujing2pluse(m,lujinglist,template_x,template_y,lag_x,lag_y):#初始化的重叠区补充路径
    L=m.shape[1]
    W=m.shape[2]
    x=template_x#初始点在一个模板厚度的ti缘
    y=template_y
    s1=int((L-1-(2*template_x))/lag_x)
    s2=int((W-1-(2*template_y))/lag_y)
    s=min((L-(2*template_x))/2,(W-(2*template_y))/2)
    #print s

    
    Lh=L/2
    Wh=W/2
    d1=[]
    #print s3
    d1.append(0)#第一个点
    for n in range(1,len(lujinglist)-1):
        if (lujinglist[n][0]==lujinglist[n-1][0]) and (lujinglist[n][1]<lujinglist[n+1][1]):
            d1.append(0)
        elif (lujinglist[n][0]==lujinglist[n-1][0]) and (lujinglist[n][1]==lujinglist[n+1][1]):
            if (lujinglist[n][0]<=Lh) and (lujinglist[n][1]>=Wh):
                d1.append(6)#端点1
            else:
                d1.append(7)#端点3
        elif (lujinglist[n][0]>lujinglist[n-1][0]) and (lujinglist[n][1]==lujinglist[n+1][1]):
            d1.append(3) 
        elif (lujinglist[n][0]>lujinglist[n-1][0]) and (lujinglist[n][1]>lujinglist[n+1][1]):
            d1.append(4)#端点2
        elif (lujinglist[n][0]==lujinglist[n-1][0]) and (lujinglist[n][1]>lujinglist[n+1][1]):
            d1.append(2)
        elif (lujinglist[n][0]<lujinglist[n-1][0]) and (lujinglist[n][1]==lujinglist[n+1][1]):
            d1.append(1)
        else:
            d1.append(5)
    d1.append(0)
    return d1


def lujing3(m,lujinglist,template_h,template_x,template_y,lag_h,lag_x,lag_y):#非柱状随机路径
    #lag为移动距离
    #template_x与template_y,lag_x与lag_y,默认统一奇偶，如果出问题原因就在此
    H=m.shape[0]
    hh=int((template_h-1)/2)
    L=m.shape[1]
    W=m.shape[2]
    s1=int((L-1-(2*template_x))/lag_x)
    s2=int((W-1-(2*template_y))/lag_y)
    s3=int(((H-1-(2*hh))/lag_h))+1
    print(s3)
    s=min((L-(2*template_x))/2,(W-(2*template_y))/2)
    h=hh
    if (L%2)==0:#当输入为偶数时
        for sh in range(s3):
            x=template_x#初始点在一个模板厚度的ti缘
            y=template_y
            lagn=0
            for sc in range(s):
                for y1 in range(s2-lagn):
                    lujinglist.append((h,x,y))
                    y=y+lag_y
                for x1 in range(s1-lagn):
                    lujinglist.append((h,x,y))
                    x=x+lag_x
                for y2 in range(s2-lagn):
                    lujinglist.append((h,x,y))
                    y=y-lag_y            
                for x12 in range(s1-lagn):
                    lujinglist.append((h,x,y))
                    x=x-lag_x
                #完成一圈
                x=x+lag_x
                y=y+lag_y
                lagn=lagn+2
            h=h+lag_h
    else:
        for sh in range(s3):
            x=template_x#初始点在一个模板厚度的ti缘
            y=template_y
            lagn=0
            for sc in range(s):
                for y1 in range(s2-lagn):
                    lujinglist.append((h,x,y))
                    y=y+lag_y
                for x1 in range(s1-lagn):
                    lujinglist.append((h,x,y))
                    x=x+lag_x
                for y2 in range(s2-lagn):
                    lujinglist.append((h,x,y))
                    y=y-lag_y            
                for x12 in range(s1-lagn):
                    lujinglist.append((h,x,y))
                    x=x-lag_x
                #完成一圈
                x=x+lag_x
                y=y+lag_y
                lagn=lagn+2
            #x=L/2
            #y=W/2
            #lujinglist.append((h,x,y))
            h=h+lag_h
    random.shuffle(lujinglist)
    list1=sorted(set(lujinglist),key=lujinglist.index) # sorted output
    lujinglist=list1
    return m,lujinglist
def lujing4(m,lujinglist,template_x,template_y,lag_x,lag_y):#柱状随机路径
    #lag为移动距离
    #template_x与template_y默认统一奇偶
    L=m.shape[1]
    W=m.shape[2]
    x=template_x#初始点在一个模板厚度的ti缘
    y=template_y
    s1=int((L-1-(2*template_x))/lag_x)
    s2=int((W-1-(2*template_y))/lag_y)
    s=min((L-(2*template_x))/2,(W-(2*template_y))/2)
    print(s)
    lagn=0
    if (L%2)==0:#当输入为偶数时
        for sc in range(s):
            for y1 in range(s2-lagn):
                lujinglist.append((x,y))
                y=y+lag_y
            for x1 in range(s1-lagn):
                lujinglist.append((x,y))
                x=x+lag_x
            for y2 in range(s2-lagn):
                lujinglist.append((x,y))
                y=y-lag_y            
            for x12 in range(s1-lagn):
                lujinglist.append((x,y))
                x=x-lag_x
            #完成一圈
            x=x+lag_x
            y=y+lag_y
            lagn=lagn+2
    else:
        for sc in range(s):
            for y1 in range(s2-lagn):
                lujinglist.append((x,y))
                y=y+lag_y
            for x1 in range(s1-lagn):
                lujinglist.append((x,y))
                x=x+lag_x
            for y2 in range(s2-lagn):
                lujinglist.append((x,y))
                y=y-lag_y            
            for x12 in range(s1-lagn):
                lujinglist.append((x,y))
                x=x-lag_x
            #完成一圈
            x=x+lag_x
            y=y+lag_y
            lagn=lagn+2
        lujinglist.append((x,y))
    random.shuffle(lujinglist)
    list1=sorted(set(lujinglist),key=lujinglist.index) # sorted output
    lujinglist=list1
    return m,lujinglist


def lujing1pluse(m,lujinglist,template_h,template_x,template_y,lag_h,lag_x,lag_y):#初始化的重叠区补充路径
    H=m.shape[0]
    hh=int((template_h-1)/2)
    L=m.shape[1]
    W=m.shape[2]
    s3=int(((H-1-(2*hh))/lag_h))+1
    Lh=L/2
    Wh=W/2
    d1=[]
    #print s3
    d1.append(0)#第一个点
    for n in range(1,(len(lujinglist)/s3)):
        if (lujinglist[n][1]==lujinglist[n-1][1]) and (lujinglist[n][2]<lujinglist[n+1][2]):
            d1.append(0)
        elif (lujinglist[n][1]==lujinglist[n-1][1]) and (lujinglist[n][2]==lujinglist[n+1][2]):
            if (lujinglist[n][1]<=Lh) and (lujinglist[n][2]>=Wh):
                d1.append(6)#端点1
            else:
                d1.append(7)#端点3
        elif (lujinglist[n][1]>lujinglist[n-1][1]) and (lujinglist[n][2]==lujinglist[n+1][2]):
            d1.append(3) 
        elif (lujinglist[n][1]>lujinglist[n-1][1]) and (lujinglist[n][2]>lujinglist[n+1][2]):
            d1.append(4)#端点2
        elif (lujinglist[n][1]==lujinglist[n-1][1]) and (lujinglist[n][2]>lujinglist[n+1][2]):
            d1.append(2)
        elif (lujinglist[n][1]<lujinglist[n-1][1]) and (lujinglist[n][2]==lujinglist[n+1][2]):
            d1.append(1)
        else:
            d1.append(5)

    for n2 in range(s3-1):
        d1.append(8)#第一个点
        for n in range(1,(len(lujinglist)/s3)):
            if (lujinglist[n][1]==lujinglist[n-1][1]) and (lujinglist[n][2]<lujinglist[n+1][2]):
                d1.append(8)
            elif (lujinglist[n][1]==lujinglist[n-1][1]) and (lujinglist[n][2]==lujinglist[n+1][2]):
                if (lujinglist[n][1]<=Lh) and (lujinglist[n][2]>=Wh):
                    d1.append(14)#端点1
                else:
                    d1.append(15)#端点3
            elif (lujinglist[n][1]>lujinglist[n-1][1]) and (lujinglist[n][2]==lujinglist[n+1][2]):
                d1.append(11) 
            elif (lujinglist[n][1]>lujinglist[n-1][1]) and (lujinglist[n][2]>lujinglist[n+1][2]):
                d1.append(12)#端点2
            elif (lujinglist[n][1]==lujinglist[n-1][1]) and (lujinglist[n][2]>lujinglist[n+1][2]):
                d1.append(10)
            elif (lujinglist[n][1]<lujinglist[n-1][1]) and (lujinglist[n][2]==lujinglist[n+1][2]):
                d1.append(9)
            else:
                d1.append(13)
        
    return d1


