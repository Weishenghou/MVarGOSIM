#!/usr/bin/env python
# coding: utf-8

# In[ ]:


######################ver2.0
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pylab

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
import difflib
import itertools as it
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
from Fault import* 
from Fault2 import*  
from NewEM import*
from fenji import* 
#################################子程序########################
def doyoulikewhatyousee(Ti):#提取TI地层层序 unfinish
    code343=[]    
    for x in range(Ti.shape[1]):
        for h in range(Ti.shape[0]):
            code343.append(Ti[h,x])

def zhiyutrans(dg):#将空白值255转换为-1
    for x in range(dg.shape[0]):
        for y in range(dg.shape[1]):
            if dg[x,y]==255:
                dg[x,y]=-1
    return dg

def zhiyutiqu(dg):#提取所有等高线值
    zhiyutrans(dg)
    lista=[]
    for x in range(dg.shape[0]):
        for y in range(dg.shape[1]):
            if (dg[x,y]!=-1) and (dg[x,y] not in lista):
                lista.append(dg[x,y])
    lista.sort()#排序
    #b=lista[::-1] 
    
    #lista=lista+b
    return lista

def xun(tem,hvalue):#寻找模版大小内最近的,lvalue为当前值，hvalue为下一个等高线的高程值
    lsit=[]#记录距离
    listx=[]#记录x坐标
    listy=[]#记录y坐标
    for x1 in range(tem.shape[0]):
        for y1 in range(tem.shape[1]):
            if tem[x1,y1]==hvalue:
                lsit.append((x1-(tem.shape[0])/2)**2+(y1-(tem.shape[1]/2))**2)
                listx.append(x1)
                listy.append(y1)
    a=lsit.index(min(lsit))
    return listx[a],listy[a]
    
    
def zhixianjisuan(tem,xx,yy,lvalue,hvalue):#直线间插值,lvalue为当前值，hvalue为下一个等高线的高程值
    x0=tem.shape[0]//2
    y0=tem.shape[1]//2
    s=max(abs(xx-x0),abs(yy-y0))
    values=(hvalue-lvalue)/s
    for nn in range(s):
        x1=x0+(nn*(xx-x0)//s)
        y1=y0+(nn*(yy-y0)//s)
        if tem[x1,y1]==-1:
            tem[x1,y1]=int(lvalue+(nn*values)+0.5)
    return tem

def extend2dAIfor2d(m,x1,y1):#9格子内随机选取一个值 for 2d version 
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
def extendTimodelfor2d(m,template_x,template_y):#全自动拓展插入硬数据的待模拟网格for 2d version
    lag=max(template_x,template_y)//2
    m2=np.pad(m,lag,'edge')
    d=[]
    for x in range(lag,m2.shape[0]-lag):
        for y in range(lag,m2.shape[1]-lag):
             d.append((x,y))
    
    for cc in range(lag+3):
        #random.shuffle(d)
        flag=0
        for n in range(len(d)):
            x=d[n][0]
            y=d[n][1]
            if m2[x,y]==-1:
                value=extend2dAIfor2d(m2,x,y)
                flag=-1
                if value!=-1:
                    #print value
                    m[x-lag,y-lag]=value
                '''
                else:
                    if cc==lag-1:
                        m[h-lag,x-lag,y-lag]=value'''
        if flag==0:
            break
        m2=np.pad(m,lag,'edge')
        #填充为1的    
    return m


def template1RAIfor2d(Ti,tem,x0,y0):#非柱状模板返还判断硬数据版
    template_x=tem.shape[0]
    template_y=tem.shape[1]

    nn2=0
    nn3=0
    xx=int((template_x-1)/2)
    yy=int((template_y-1)/2)

    for n2 in range(x0-xx,x0+xx+1):
        for n3 in range(y0-yy,y0+yy+1):
            if Ti[n2,n3]==-1:
                Ti[n2,n3]=tem[nn2,nn3]
            nn3=nn3+1
            #print(nn3)
        nn2=nn2+1
        nn3=0
    nn2=0

        #print(nn2)
    return Ti  #提取坐标x0,y0处模板 

def denggaoxiao2dem(dg):#dg为二维等高线图
    lista=zhiyutiqu(dg)
    print('包含等高线的值',lista)
    l=max(dg.shape[0],dg.shape[1])
    dg2=np.pad(dg,((l,l),(l,l)),'constant',constant_values=-10)
    #print(dg2)
    for n in range(len(lista)):
        for nn in range(len(lista)):
            for x in range(l,dg.shape[0]+l):
                for y in range(l,dg.shape[1]+l):
                    #print(x-l,y-l)
                    if dg2[x,y]==lista[n]:
                        xx,yy=xun(dg2[x-l:x+l+1,y-l:y+l+1],lista[nn])
                        tem0=zhixianjisuan(dg2[x-l:x+l+1,y-l:y+l+1],xx,yy,lista[n],lista[nn])
                        #计算直线
                        #返回tem
                        #print(tem0)
                        dg2=template1RAIfor2d(dg2,tem0,x,y)
    dgs=dg2[l:dg.shape[0]+l,l:l+dg.shape[1]]
    dgs=extendTimodelfor2d(dgs,l,l)
    return dgs
                    
def two2dem(m,dg):#通过俯视图定义模拟范围
    dg=sectionex(dg,m.shape[1],m.shape[2])
    #首先将俯视图拉伸至同等大小
    dgs=denggaoxiao2dem(dg)
    for x in range(m.shape[1]):
        for y in range(m.shape[2]):
            for h in range(dgs[x,y]):
                m[h,x,y]=-10
    return m


def sectionloadG(m,section,hz,hz2,xz,yz,xz2,yz2):#相对坐标的剖面导入,section为数组,xz,yz分别为剖面两端点的相对坐标，加入高程坐标ver
#斜剖面导入后需要扩充才正确
    #print(hz,hz2)
    ns=section.shape[1]
    hc=float(hz2-hz)+1
    xc=float(xz2-xz)
    yc=float(yz2-yz)
    if xc<0:
        xc1=xc-1
    else:
        xc1=xc+1
    if yc<0:
        yc1=yc-1
    else:
        yc1=yc+1
    #计量后加一为长度
    lv=int(max(abs(xc1),abs(yc1)))#比较长度绝对值大小，得到即为斜剖面需要填网格总数，所以需要加一
    xlv=xc/(lv-1)
    ylv=yc/(lv-1)
    x1=xz
    y1=yz
    #对section的处理
    #h=m.shape[0]
    section=sectionex(section,int(hc),lv)
    #print h
    #print section.shape[0],section.shape[1],lv,m.shape[0],m.shape[1],m.shape[2]
    for n in range(lv):
        m[hz:hz2+1,x1,y1]=section[:,n]      
        #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
        x1=int(xz+(n+1)*xlv+0.5)#四舍五入
        y1=int(yz+(n+1)*ylv+0.5)
    return m

def sectionreadG(m,hz,hz2,xz,yz,xz2,yz2):#相对坐标的剖面读取,section为数组,xz,yz分别为剖面两端点的相对坐标，加入高程坐标ver

    if (abs(xz-xz2)==m.shape[1])&(yz==yz2):#当剖面为竖时 
        section=m[hz:hz2+1,:,yz]
    elif (abs(yz-yz2)==m.shape[2])&(xz==xz2):#当剖面为横时
        section=m[hz:hz2+1,xz,:]
    else: #斜剖面
        xc=float(xz2-xz)
        yc=float(yz2-yz)
        if xc<0:
            xc1=xc-1
        else:
            xc1=xc+1
        if yc<0:
            yc1=yc-1
        else:
            yc1=yc+1
        #计量后加一为长度
        lv=int(max(abs(xc1),abs(yc1)))#比较长度绝对值大小，得到即为斜剖面需要填网格总数，所以需要加一
        xlv=xc/(lv-1)
        ylv=yc/(lv-1)
        x1=xz
        y1=yz
        section=np.zeros((m.shape[0],lv),int)

        for n in range(lv):
            section[:,n]=m[:,x1,y1]      
            #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
            x1=int(xz+(n+1)*xlv+0.5)#四舍五入
            y1=int(yz+(n+1)*ylv+0.5)
    return section

def RecodeTIextendforEMG(section,m,template_x,template_y,h1,h2,x1,y1,x2,y2):#EM迭代用剖面提取，(x,y）为剖面定位的一组坐标
    #m为已经完成扩展的模拟网格
    dx=[]
    dy=[]
    dh=[]
    lag=max(template_x,template_y)//2
    ms=-np.ones((m.shape[0],m.shape[1],m.shape[2]),int)
    sectionloadG(ms,section,h1,h2,x1,y1,x2,y2)#独立载入防止造成多剖面混乱
    Tizuobiaox=-np.ones((m.shape[0],m.shape[1],m.shape[2]), int)
    Tizuobiaoy=-np.ones((m.shape[0],m.shape[1],m.shape[2]), int)
    Tizuobiaoh=-np.ones((m.shape[0],m.shape[1],m.shape[2]), int)
    for h in range(Tizuobiaoh.shape[0]):        
        for x in range(Tizuobiaoh.shape[1]):
            for y in range(Tizuobiaoh.shape[2]):
                Tizuobiaoh[h,x,y]=h
    if abs(h1-h2)>=lag:
        for n1 in range(min(h1,h2),max(h1,h2)+1):
            dh.append(n1)
    else:
        for n1 in range(max(0,min(h1,h2)-lag),min(max(h1,h2)+lag,m.shape[1]-1)+1):
            dh.append(n1)    

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
    fowt=temp[:,:,dy]
    fow=fowt[dh,:,:]
    Tizuobiaoxt=Tizuobiaox[:,dx,:]
    Tizuobiaoxx=Tizuobiaoxt[:,:,dy]
    Tizuobiaox=Tizuobiaoxx[dh,:,:]
    Tizuobiaoyt=Tizuobiaoy[:,dx,:]
    Tizuobiaoyy=Tizuobiaoyt[:,:,dy]
    Tizuobiaoy=Tizuobiaoyy[dh,:,:]
    Tizuobiaoht=Tizuobiaoh[:,dx,:]
    Tizuobiaohh=Tizuobiaoht[:,:,dy]
    Tizuobiaoh=Tizuobiaohh[dh,:,:]
    c=max(fow.shape[1],fow.shape[2])
    #Tizuobiaoh=-np.ones((fow.shape[0],fow.shape[1],fow.shape[2]), int)

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

def sectionexyunshi(section,height,length,jivalue):#剖面的缩放
    ns=section.shape[1]
    ns2=section.shape[0]
    lv=length
    lv2=height
    if ns2!=lv2:
        if ns2>lv2:
            #缩小至lv2高度
            kksk=float(lv2)/ns2
            section_new=np.zeros((height,section.shape[1]),int)
            for n in range(ns2):
                for kkk in range(section_new.shape[1]):
                    if section_new[int(n*kksk),kkk]!=jivalue:
                       section_new[int(n*kksk),kkk]=section[n,kkk]
                #print float(n*kksk)
            section=section_new
        elif ns2<lv:
            #扩大至lv长度
            kksk=ns2/float(lv2)
            section_new=np.zeros((height,section.shape[1]),int)
            for n in range(lv2):
                section_new[n,:]=section[int(n*kksk),:]
            section=section_new
    if ns!=lv:
        if ns>lv:
            #缩小至lv长度
            kksk=float(lv)/ns
            section_new2=np.zeros((section.shape[0],lv),int)
            for n in range(ns):
                for kkk in range(section_new2.shape[0]):
                    if section_new2[kkk,int(n*kksk)]!=jivalue:
                       section_new2[kkk,int(n*kksk)]=section[kkk,n]
                #print float(n*kksk)
            section=section_new2
        elif ns<lv:
            #扩大至lv长度
            kksk=ns/float(lv)
            section_new2=np.zeros((section.shape[0],lv),int)
            for n in range(lv):
                section_new2[:,n]=section[:,int(n*kksk)]
            section=section_new2

    return section

def sectionload_xG(m,section,hz,hz2,xz,yz,xz2,yz2,jivalue):#相对坐标的剖面导入,section为数组,xz,yz分别为剖面两端点的相对坐标
#斜剖面导入后需要扩充才正确 jivalue为基质值
 

    #斜剖面导入后需要扩充才正确
    #print(hz,hz2)
    ns=section.shape[1]
    hc=float(hz2-hz)+1
    xc=float(xz2-xz)
    yc=float(yz2-yz)
    if xc<0:
        xc1=xc-1
    else:
        xc1=xc+1
    if yc<0:
        yc1=yc-1
    else:
        yc1=yc+1
    #计量后加一为长度
    lv=int(max(abs(xc1),abs(yc1)))#比较长度绝对值大小，得到即为斜剖面需要填网格总数，所以需要加一
    xlv=xc/(lv-1)
    ylv=yc/(lv-1)
    x1=xz
    y1=yz
    #对section的处理
    #h=m.shape[0]
    section=sectionexyunshi(section,int(hc),lv,jivalue)
    #print h
    #print section.shape[0],section.shape[1],lv,m.shape[0],m.shape[1],m.shape[2]
    for n in range(lv):
        m[hz:hz2+1,x1,y1]=section[:,n]      
        #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
        x1=int(xz+(n+1)*xlv+0.5)#四舍五入
        y1=int(yz+(n+1)*ylv+0.5)
    return m

def sectionload_xG2(m,section,hz,hz2,xz,yz,xz2,yz2,jivalue):#相对坐标的剖面导入,section为数组,xz,yz分别为剖面两端点的相对坐标,增加剖面上方赋值空值的功能
#斜剖面导入后需要扩充才正确 jivalue为基质值
 

    #斜剖面导入后需要扩充才正确
    #print(hz,hz2)
    ns=section.shape[1]
    hc=float(hz2-hz)+1
    xc=float(xz2-xz)
    yc=float(yz2-yz)
    if xc<0:
        xc1=xc-1
    else:
        xc1=xc+1
    if yc<0:
        yc1=yc-1
    else:
        yc1=yc+1
    #计量后加一为长度
    lv=int(max(abs(xc1),abs(yc1)))#比较长度绝对值大小，得到即为斜剖面需要填网格总数，所以需要加一
    xlv=xc/(lv-1)
    ylv=yc/(lv-1)
    x1=xz
    y1=yz
    #对section的处理
    #h=m.shape[0]
    section=sectionexyunshi(section,int(hc),lv,jivalue)
    #print h
    #print section.shape[0],section.shape[1],lv,m.shape[0],m.shape[1],m.shape[2]
    for n in range(lv):
        m[hz:hz2+1,x1,y1]=section[:,n]      
        #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
        x1=int(xz+(n+1)*xlv+0.5)#四舍五入
        y1=int(yz+(n+1)*ylv+0.5)

    x1=xz
    y1=yz
    for n in range(lv):
        if section[0,n]==0:
           m[0:hz+1,x1,y1]=0      
        #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
        x1=int(xz+(n+1)*xlv+0.5)#四舍五入
        y1=int(yz+(n+1)*ylv+0.5)

        
    return m


def sectionloadandextendG(m,template_x,template_y,flag,scale,jvalue):#flag==1为patchmatch步骤，0为initial步骤 比起上面的版本增加了剖面上方归为零值的默认操作、防止小个体jvalue被缩放忽略的操作
    #对剖面进行导入和Ti提取的函数  #scale为当前倍率
    #输出的坐标列表为RecodePatchmatch需要的格式
    #插入地层层序判断的版本 加入了h方向的坐标
    Tilist=[]
    Tizuobiaolist=[]
    codelist=[]#地层层序列表
    file1=open('./Ti/Tiparameter.txt')
    content=file1.readline()
    string1=[i for i in content if str.isdigit(i)]
    num=int(''.join(string1))
    print('剖面数目：')
    print(num)
    for n in range(num):
        guding=[]
        for aa in range(6):
            content=file1.readline()
            string1=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string1))
            guding.append(xx)
        path='./Ti/'+str(n+1)+'.bmp'
        section=cv2.imread(path,0)
        #print(guding)
        codelist=doyoulikewhatyousee1(section,codelist)

        #print guding[0],guding[1],guding[2],guding[3]
        m=sectionload_xG2(m,section,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale,guding[4]*scale,guding[5]*scale,jvalue)#载入剖面
        if flag==1:
            Ti,Tizuobiao=RecodeTIextendforEMG(section,m,template_x,template_y,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale,guding[4]*scale,guding[5]*scale)
            Tilist.append(Ti)
            Tizuobiaolist.append(Tizuobiao)

    return m,Tilist,Tizuobiaolist,codelist
########################################################导入剖面（包含高程）以及二维转三维高程约束程序






########################################################




def replacepi(re,hardlist):#re为带替换的值，hardlist为不替换的坐标
    for n in range(re.shape[0]):
        if re[n] not in hardlist:
            re[n]=-1
    return re

def initialroadlistAIR(m,template_h,template_x,template_y,lag):#改进版
    #lag为重叠区
    #自动初始化网格系统
    lujing=[]
    Roadlist=[]#最终路径名单
    lujing=lujinglistAI2(m,template_h,template_x,template_y,lag-1)
    random.shuffle(lujing)
    #print len(lujing)
    
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
 
    DevilTrigger=False
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    Banlujing=[]
    b=np.zeros((template_h,template_x,template_y),int)
    n=0
    
    while n<len(lujing):
        if m2[lujing[n][0]+lag,lujing[n][1]+lag,lujing[n][2]+lag]==-1:
            #if lujing[n] not in Banlujing:
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            
            if temdetectD1(o1[0:lag,:,:]): 
                #上
                k=k+1
            
            if temdetectD1(o1[template_h-lag:template_h,:,:]):
                #下
                k=k+1
            if temdetectD1(o1[:,0:lag,:]):
                #后
                k=k+1
            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前 
                k=k+1
            if temdetectD1(o1[:,:,0:lag]):
                #左
                k=k+1
            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                k=k+1
            if (h1>template_h-lag) and (k>=2):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            elif (h1<=template_h-lag) and (k!=0):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            else:
                lujing.append(lujing[n])
        print(len(Roadlist),len(lujing)-n)
        n=n+1
        #print len(Roadlist)
    print('roadlist initial done')
    return Roadlist

def initialPythia(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N,codelist,hardlist):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIR(m,template_h,template_x,template_y,lag)
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
    reb=-np.ones((m2.shape[0]),int)
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
    sqflag=0
    while sqflag==0:
        sqflag=0
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
            
            if temdetectD1(o1[0:lag,:,:]):
                #上
                b=np.zeros((template_h,template_x,template_y),int)
                b[dis,:,:]=1
                c[dis,:,:]=1
                temo=o1*b
                canpatternlist0=patternsearchDi(Cdatabase[5],cdatabase[5],temo)
                   

            #if o1[template_h-1,template_x//2,template_y//2]!=-1:
            if temdetectD1(o1[template_h-lag:template_h,:,:]):
                #下
                b=np.zeros((template_h,template_x,template_y),int)
                b[dish,:,:]=1
                c[dish,:,:]=1
                temo=o1*b
                canpatternlist1=patternsearchDi(Cdatabase[0],cdatabase[0],temo)
                if temdetect0d(o1[template_h-1,:,:]):
                    flag=1

            if temdetectD1(o1[:,0:lag,:]):
                #后
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,dis,:]=1
                c[:,dis,:]=1
                temo=o1*b
                canpatternlist2=patternsearchDi(Cdatabase[3],cdatabase[3],temo)
                if temdetect0d(o1[:,0,:]):
                    flag=1

            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,disx,:]=1
                c[:,disx,:]=1
                temo=o1*b
                canpatternlist3=patternsearchDi(Cdatabase[4],cdatabase[4],temo)
                if temdetect0d(o1[:,template_x-1,:]):
                    flag=1
 
            if temdetectD1(o1[:,:,0:lag]):
                #左
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,dis]=1
                c[:,:,dis]=1
                temo=o1*b
                canpatternlist4=patternsearchDi(Cdatabase[1],cdatabase[1],temo)
                if temdetect0d(o1[:,:,0]):
                    flag=1

            if temdetectD1(o1[:,:,template_y-lag:template_y]):
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
            print(n)
            if flag!=0:
                tem=np.zeros((template_h,template_x,template_y),int)
            else:
                #print("have")
                temo=o1*c
                tem=patternsearchAI2(temo,c,cdatabase[6],canpatternlist,N)
            m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
        
        
        relist=[]
        
        for x2 in range(lag,X+lag):  
            for y2 in range(lag,Y+lag):
                
                code897=doyoulikewhatyousee3(m2[lag:H+lag,x2,y2])
                if codecheckZ(code897,codelist):
                   
                   if (x2,y2) not in relist:
                        #print((x2,y2))
                        #print(code897)
                        relist.append((x2,y2))
        print(len(relist))
        #print(code)
        for n in range(len(relist)):
            wsyw=m2[:,relist[n][0],relist[n][1]]
            wsyw=replacepi(wsyw,hardlist)
            m2[:,relist[n][0],relist[n][1]]=wsyw
           

        print('output')
        
        m2=extendTimodel(m2,5,5,5)
        lujing=[]
        disss=[]    
        ms,disss=checkunreal2(m2,lag)
        if len(disss)==0:
            sqflag=1
        else:
            
            lujing=subroadlistinitialfornew(m2,disss,template_h,template_x,template_y,lag)
            
        
    m=cut(m2,lag)
    return m
def initialPythiasub(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N,codelist,hardlist):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIR(m,template_h,template_x,template_y,lag)
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
    reb=-np.ones((m2.shape[0]),int)
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
    sqflag=0
    while sqflag==0:
        sqflag=0
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
            
            if temdetectD1(o1[0:lag,:,:]):
                #上
                b=np.zeros((template_h,template_x,template_y),int)
                b[dis,:,:]=1
                c[dis,:,:]=1
                temo=o1*b
                canpatternlist0=patternsearchDi(Cdatabase[5],cdatabase[5],temo)
                   

            #if o1[template_h-1,template_x//2,template_y//2]!=-1:
            if temdetectD1(o1[template_h-lag:template_h,:,:]):
                #下
                b=np.zeros((template_h,template_x,template_y),int)
                b[dish,:,:]=1
                c[dish,:,:]=1
                temo=o1*b
                canpatternlist1=patternsearchDi(Cdatabase[0],cdatabase[0],temo)
                if temdetect0d(o1[template_h-1,:,:]):
                    flag=1

            if temdetectD1(o1[:,0:lag,:]):
                #后
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,dis,:]=1
                c[:,dis,:]=1
                temo=o1*b
                canpatternlist2=patternsearchDi(Cdatabase[3],cdatabase[3],temo)
                if temdetect0d(o1[:,0,:]):
                    flag=1

            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,disx,:]=1
                c[:,disx,:]=1
                temo=o1*b
                canpatternlist3=patternsearchDi(Cdatabase[4],cdatabase[4],temo)
                if temdetect0d(o1[:,template_x-1,:]):
                    flag=1
 
            if temdetectD1(o1[:,:,0:lag]):
                #左
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,dis]=1
                c[:,:,dis]=1
                temo=o1*b
                canpatternlist4=patternsearchDi(Cdatabase[1],cdatabase[1],temo)
                if temdetect0d(o1[:,:,0]):
                    flag=1

            if temdetectD1(o1[:,:,template_y-lag:template_y]):
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
            print(n)
            if flag!=0:
                tem=np.zeros((template_h,template_x,template_y),int)
            else:
                #print("have")
                temo=o1*c
                tem=patternsearchAI2(temo,c,cdatabase[6],canpatternlist,N)
            m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
        
        
        relist=[]
        
        for x2 in range(lag,X+lag):  
            for y2 in range(lag,Y+lag):
                
                code897=doyoulikewhatyousee3(m2[lag:H+lag,x2,y2])
                if codecheckZ(code897,codelist):
                   
                   if (x2,y2) not in relist:
                        #print((x2,y2))
                        #print(code897)
                        relist.append((x2,y2))
        print(len(relist))
        #print(code)
        for n in range(len(relist)):
            wsyw=m2[:,relist[n][0],relist[n][1]]
            wsyw=replacepi(wsyw,hardlist)
            m2[:,relist[n][0],relist[n][1]]=wsyw
           
        data=m2.transpose(-1,-2,0)#转置坐标系

        print('output')
        
        m2=extendTimodel(m2,5,5,5)
        lujing=[]
        disss=[]    
        ms,disss=checkunreal2(m2,lag)
        if len(disss)==0:
            sqflag=1
        else:
            
            lujing=subroadlistinitialfornew(m2,disss,template_h,template_x,template_y,lag)
            
        
    m=cut(m2,lag)
    return m
def initialAIforPythia(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,hardlist,code,valueliata):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格
    #m为已经导入了Ti的模拟网格
    time_start1=time.time()
    m=extendTimodel(m,template_h,template_x,template_y)#拓展模拟网格


    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial.vtk') 
    np.save('./output/outputinitial.npy',m)
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
    print('数据库构建时间损耗:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    #########################机器学习部分
    m=buildfaultkerasM3(m,valueliata,flaglist,epoch)
    np.save('./output/outputinitialFault.npy',m)


    #########################
    
    
    
    
    m=initialPythiasub(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N,code,hardlist)
    
    time_end=time.time()
    
    np.save('./output/outputinitial3.npy',m)
    
    print('initial done')
    print('初始化建模时间损耗:')
    print(time_end-time_start)
    #初始化
    return m


def Pythia(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,hardlist,code,valueliata,scale,epoch,flaglist,jvalue):
    #皮提亚主程序
    #code 需要事先定义好TI中不包含的
    
    #m,Tilist,Tizuobiaolist=sectionloadandextend(m,patternSizex,patternSizey,0,1)
    m,Tilist,Tizuobiaolist,codelist=sectionloadandextendG(m,template_x,template_y,flag,1,jvalue)
    #待插入二维高程建模区域约束
    codelist.append(code)
    
    print('当前合理地层层序列表：',codelist)
    print('当前模拟地层顺序：',valueliata)
    m=initialAIforPythia(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,hardlist,codelist,valueliata)
    #初始化部分结束
    
    #em迭代部分
    
    sancheck=1#sectionloadandextend倍率机制
    np.save('./output/initial.npy',m)
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)

        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist,codelist=sectionloadandextendG(mm,patternSizex,patternSizey,1,sancheck,jvalue)


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
        #########m= patchmatchmultiTiB(m,Tilist,size,itr,1)
        CTI=[]#检测使用率剖面
        #m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,4,0)#并行进程的数目
        #计算量加大过多
        #m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path="./output/reconstruction.npy"
        np.save(path,m)
        time_end=time.time()
        #size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end-time_start)

        data=m.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/output.vtk') 
    return m

    
################################计时程序########################################################
time_start1=time.time()#计时开始


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

#一次模拟的个数
content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
Modelcount=int(''.join(string1))
#print Modelcount
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
#####################预处理阶段################################################################
epoch=10000

valueliata=[]
valueliata=Tilistvalueextract()
valueliata.sort(reverse=True)
flaglist=[0,0,0]#用来判断地层为局部或者全局
print (valueliata)
valueliata=[40,21,50]
print('请自行进行排序:',valueliata)
code=[40,21,50]
hardlist=[]
jvalue=30
m=-np.ones((Mh,Mx,My),int)#默认空值为-1

Pythia(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,hardlist,code,valueliata,scale,epoch,flaglist,jvalue)

time_end1=time.time()#计时结束
print('总耗时：',time_end1-time_start1)

