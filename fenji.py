#!/usr/bin/env python
# coding: utf-8

# In[9]:


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
from PatchmatchZ import*
from entropyweight import*
from AIinitial import*
from NewEM import*

####################################################
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

def sectionload_xyunshi(m,section,xz,yz,xz2,yz2,jivalue):#相对坐标的剖面导入,section为数组,xz,yz分别为剖面两端点的相对坐标
#斜剖面导入后需要扩充才正确 jivalue为基质值
    ns=section.shape[1]
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
    h=m.shape[0]
    section=sectionexyunshi(section,h,lv,jivalue)
    #print h
    #print section.shape[0],section.shape[1],lv,m.shape[0],m.shape[1],m.shape[2]
    for n in range(lv):
       m[:,x1,y1]=section[:,n]      
       #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
       x1=int(xz+(n+1)*xlv+0.5)#四舍五入
       y1=int(yz+(n+1)*ylv+0.5)
    return m

def doyoulikewhatyousee1(Ti,code):#提取TI地层层序 


    for x in range(Ti.shape[1]):
        code343=[]
        code343.append(Ti[0,x])
        for h in range(1,Ti.shape[0]):
            if Ti[h,x]!=Ti[h-1,x]:
               if Ti[h,x]!=-1:
                  code343.append(Ti[h,x])
        if code343 not in code:
           code.append(code343)
    return code

'''
def doyoulikewhatyousee2(Ti):#提取网格地层层序 
    code=[]    

    for x in range(Ti.shape[1]):
        for y in range(Ti.shape[2]): 
            code343=[]

               code343.append(Ti[0,x,y])
            
            for h in range(Ti.shape[0]):
                
                if Ti[h,x,y]!=Ti[h-1,x,y]:
                   if Ti[h,x,y]!=-1:
                      if len(code343)==0:
                         code343.append(Ti[h,x,y])
                      elif Ti[h,x,y]!=code343[len(code343)-1]:
                         code343.append(Ti[h,x,y])
            if code343 not in code:
                code.append(code343)
    return code
'''
def doyoulikewhatyousee2(Ti):#提取网格地层层序 
    code=[]    

    for x in range(Ti.shape[1]):
        for y in range(Ti.shape[2]): 
            code343=[]

            code343.append(Ti[0,x,y])
            
            for h in range(Ti.shape[0]):                
                if Ti[h,x,y]!=Ti[h-1,x,y]:
                    code343.append(Ti[h,x,y])
            for cik in range(len(code343)):
                if -1 in code343:
                       
                   code343.remove(-1)
                else:
                   break        
            code343 = [k for k, g in it.groupby(code343)]      
            #去掉-1，去掉相邻重复
            if code343 not in code:
                code.append(code343)
    return code
def doyoulikewhatyousee3(Ti):#提取网格地层层序 
  


    code343=[]

    code343.append(Ti[0])
            
    for h in range(Ti.shape[0]):                
        if Ti[h]!=Ti[h-1]:
           code343.append(Ti[h])
    for cik in range(len(code343)):
        if -1 in code343:
                       
           code343.remove(-1)
        if -2 in code343:
                       
           code343.remove(-2)
        else:
           break        
    code343 = [k for k, g in it.groupby(code343)]      
    #去掉-1，去掉相邻重复

    return code343


def codecheck(list1,codelist):#地层检测机制
    if len(list1)==1:
        return True
    flag=0#判断是否全部都非列表子集
    for neck in range(len(codelist)):
        if set(list1) <=set(codelist[neck]):
            flag=1
            b = [val for val in codelist[neck] if val in list1]
            for n in range(1,len(list1)):
                if list1.index(b[n-1])>list1.index(b[n]):
                    return False
    if flag==1:
        return True
    return False



def Nocodecheck(list1,codelist): #地层检测机,若存在问题返回True
    if len(list1)==1:
       return False
    flag=0
    for neck in range(len(codelist)):
        if set(list1) <=set(codelist[neck]):
            flag=1
            b = [val for val in codelist[neck] if val in list1]
            for n in range(1,len(list1)):
                if list1.index(b[n-1])>list1.index(b[n]):
                    return True
    if flag==1:        
       return False
    return True


def ctrlf(m,m2,valuelist,value2):#valuelist为待选中值列表，value2为替换值
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                for n in range(len(valuelist)):
                    if m[h,x,y]==valuelist[n]:
                        m2[h,x,y]=value2 
    return m2



def pictureclass(m,fenjilist,fenjineironglist):#fenjilist存要分级的几个值，fenjineironglist储存每个级对应包含哪些值
    m2= -np.ones_like(m)
    for n in range(len(fenjilist)):
        m2=ctrlf(m,m2,fenjineironglist[n],fenjilist[n])
    
    return m2
 
def ctrlf(m,m2,valuelist,value2):#valuelist为待选中值列表，value2为替换值
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                for n in range(len(valuelist)):
                    if m[h,x,y]==valuelist[n]:
                        m2[h,x,y]=value2 
    return m2

def pictureclass(m,fenjilist,fenjineironglist):#fenjilist存要分级的几个值，fenjineironglist储存每个级对应包含哪些值
    m2= -np.ones_like(m)
    for n in range(len(fenjilist)):
        m2=ctrlf(m,m2,fenjineironglist[n],fenjilist[n])
    
    return m2 
    
def sectionloadandextendFenji(m,template_x,template_y,flag,scale):#flag==1为patchmatch步骤，0为initial步骤
    #对剖面进行导入和Ti提取的函数  #scale为当前倍率
    Tilist=[]
    Tizuobiaolist=[]
    file1=open('./Ti/Tiparameter.txt')
    content=file1.readline()
    string1=[i for i in content if str.isdigit(i)]
    num=int(''.join(string1))
    print('剖面数目：')
    print(num)
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
    #软数据
    Tilist2=[]
    file2=open('./Ti/Tiparameter.txt')
    content=file2.readline()
    string2=[i for i in content if str.isdigit(i)]
        
    m2= -np.ones_like(m)
    for n in range(num):
        guding=[]
        for aa in range(4):
            content=file2.readline()
            string2=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string2))
            guding.append(xx)
        path='./Ti/softdata/'+str(n+1)+'.bmp'
        section=cv2.imread(path,0)
        #print guding[0],guding[1],guding[2],guding[3]
        m2=sectionload_x(m2,section,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)#载入剖面
    return m,m2,Tilist,Tizuobiaolist

def sectionloadandextendFenji2(m,template_x,template_y,flag,scale,jvalue):#flag==1为patchmatch步骤，0为initial步骤
    #对剖面进行导入和Ti提取的函数  #scale为当前倍率
    #输出的坐标列表为RecodePatchmatch需要的格式
    Tilist=[]
    Tizuobiaolist=[]
    file1=open('./Ti/Tiparameter.txt')
    content=file1.readline()
    string1=[i for i in content if str.isdigit(i)]
    num=int(''.join(string1))
    print('剖面数目：')
    print(num)
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





def sectionloadandextendFenji3(m,template_x,template_y,flag,scale,jvalue):#flag==1为patchmatch步骤，0为initial步骤
    #对剖面进行导入和Ti提取的函数  #scale为当前倍率
    #输出的坐标列表为RecodePatchmatch需要的格式
    #插入地层层序判断的版本
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
        for aa in range(4):
            content=file1.readline()
            string1=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string1))
            guding.append(xx)
        path='./Ti/'+str(n+1)+'.bmp'
        section=cv2.imread(path,0)

        codelist=doyoulikewhatyousee1(section,codelist)

        #print guding[0],guding[1],guding[2],guding[3]
        sectionload_xyunshi(m,section,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale,jvalue)
        #sectionload_x(m,section,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)#载入剖面
        if flag==1:
            Ti,Tizuobiao=RecodeTIextendforEM(section,m,template_x,template_y,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)
            Tilist.append(Ti)
            Tizuobiaolist.append(Tizuobiao)
    #都执行完后可进行gosiminitialAI

    return m,Tilist,Tizuobiaolist,codelist







def databasebuildAIFenji(Exm,template_h,template_x,template_y,Fenjilist,Fenjineironglist):#智能构建模式库
    #Exm为已经完成了拓展的模拟网格
    lag=max(template_h,template_x,template_y)
    Exm2=np.pad(Exm,lag,'edge')#拓展
    Fdatabase=[]
    Fzuobiaolist=[]
    database=[]
    zuobiaolist=[]
    for soul in range(len(Fenjilist)):
        for h in range(Exm.shape[0]):
            for x in range(Exm.shape[1]):
                for y in range(Exm.shape[2]):
                    if Exm[h,x,y]!=-1:
                        h0=h+lag
                        x0=x+lag
                        y0=y+lag
                        tem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
                        if temdetect(tem):#如果不包含待模拟点则为模板
                            if Exm[h,x,y] in Fenjineironglist[soul]:
                                database.append(tem)
                                zuobiaolist.append((h,x,y))
        Fdatabase.append(database)#标号与分级列表标号一致
        Fzuobiaolist.append(zuobiaolist)
    return Fdatabase,Fzuobiaolist

def template1RAIFenji(Ti,tem,h0,x0,y0,Tisoft,value):#非柱状模板返还判断硬数据版
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
                if (Ti[n1,n2,n3]==-1) and (Tisoft[n1,n2,n3]==value):
                    Ti[n1,n2,n3]=tem[nn1,nn2,nn3]
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
        nn1=nn1+1
        nn2=0

        #print(nn2)
    return Ti  #提取坐标x0,y0处模板 

def initialgridAI2zuobiaoverFenji(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,zuobiaolist,N,Fenjilist):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAI(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#拓展
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
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
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
        if o1[0,template_x//2,template_y//2]!=-1:
            #上
            b=np.zeros((template_h,template_x,template_y),int)
            b[dis,:,:]=1
            c[dis,:,:]=1
            temo=o1*b
            canpatternlist0=patternsearchDi(CFdatabase[dark][5],cFdatabase[dark][5],temo)
            if temdetect0d(o1[0,:,:]):
                flag=1
        '''        

        if o1[template_h-1,template_x//2,template_y//2]!=-1:
            #下
            b=np.zeros((template_h,template_x,template_y),int)
            b[dish,:,:]=1
            c[dish,:,:]=1
            temo=o1*b
            canpatternlist1=patternsearchDi(CFdatabase[dark][0],cFdatabase[dark][0],temo)
            if temdetect0d(o1[template_h-1,:,:]):
                flag=1

        if o1[template_h//2,0,template_y//2]!=-1:
            #后
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,dis,:]=1
            c[:,dis,:]=1
            temo=o1*b
            canpatternlist2=patternsearchDi(CFdatabase[dark][3],cFdatabase[dark][3],temo)
            if temdetect0d(o1[:,0,:]):
                flag=1

        if o1[template_h//2,template_x-1,template_y//2]!=-1:
            #前
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,disx,:]=1
            c[:,disx,:]=1
            temo=o1*b
            canpatternlist3=patternsearchDi(CFdatabase[dark][4],cFdatabase[dark][4],temo)
            if temdetect0d(o1[:,template_x-1,:]):
                flag=1
 
        if o1[template_h//2,template_x//2,0]!=-1:
            #左
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,dis]=1
            c[:,:,dis]=1
            temo=o1*b
            canpatternlist4=patternsearchDi(CFdatabase[dark][1],cFdatabase[dark][1],temo)
            if temdetect0d(o1[:,:,0]):
                flag=1

        if o1[template_h//2,template_x//2,template_y-1]!=-1:
            #右
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,disy]=1
            c[:,:,disy]=1
            temo=o1*b
            canpatternlist5=patternsearchDi(CFdatabase[dark][2],cFdatabase[dark][2],temo)
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
            tem=patternsearchAI3(temo,lujing[n][0],lujing[n][1],lujing[n][2],c,cFdatabase[dark][6],canpatternlist,zuobiaolist,N)#判断坐标远近返回最近的候选模板
        ms=template1RAI(ms,tem,h1,x1,y1,ms2,Fenjilist[dark])
            
        
    m=cut(ms,lag)
    return m

def initialgridAI2Fenji(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,zuobiaolist,N,Fenjilist):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAI(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#拓展
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
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
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
        if o1[0,template_x//2,template_y//2]!=-1:
            #上
            b=np.zeros((template_h,template_x,template_y),int)
            b[dis,:,:]=1
            c[dis,:,:]=1
            temo=o1*b
            canpatternlist0=patternsearchDi(CFdatabase[dark][5],cFdatabase[dark][5],temo)
            if temdetect0d(o1[0,:,:]):
                flag=1
        '''       

        if o1[template_h-1,template_x//2,template_y//2]!=-1:
            #下
            b=np.zeros((template_h,template_x,template_y),int)
            b[dish,:,:]=1
            c[dish,:,:]=1
            temo=o1*b
            canpatternlist1=patternsearchDi(CFdatabase[dark][0],cFdatabase[dark][0],temo)
            if temdetect0d(o1[template_h-1,:,:]):
                flag=1

        if o1[template_h//2,0,template_y//2]!=-1:
            #后
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,dis,:]=1
            c[:,dis,:]=1
            temo=o1*b
            canpatternlist2=patternsearchDi(CFdatabase[dark][3],cFdatabase[dark][3],temo)
            if temdetect0d(o1[:,0,:]):
                flag=1

        if o1[template_h//2,template_x-1,template_y//2]!=-1:
            #前
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,disx,:]=1
            c[:,disx,:]=1
            temo=o1*b
            canpatternlist3=patternsearchDi(CFdatabase[dark][4],cFdatabase[dark][4],temo)
            if temdetect0d(o1[:,template_x-1,:]):
                flag=1
 
        if o1[template_h//2,template_x//2,0]!=-1:
            #左
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,dis]=1
            c[:,:,dis]=1
            temo=o1*b
            canpatternlist4=patternsearchDi(CFdatabase[dark][1],cFdatabase[dark][1],temo)
            if temdetect0d(o1[:,:,0]):
                flag=1

        if o1[template_h//2,template_x//2,template_y-1]!=-1:
            #右
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,disy]=1
            c[:,:,disy]=1
            temo=o1*b
            canpatternlist5=patternsearchDi(CFdatabase[dark][2],cFdatabase[dark][2],temo)
            if temdetect0d(o1[:,:,template_y-1]):
                flag=1

                
                
                

        canpatternlist=[]
        canpatternlist=list(set(canpatternlist0).union(set(canpatternlist1)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist4)))
        canpatternlist=list(set(canpatternlist).union(set(canpatternlist5)))
        
        #print(len(canpatternlist0),len(canpatternlist1),len(canpatternlist2),len(canpatternlist3),len(canpatternlist4),len(canpatternlist5))
        #print(canpatternlist0,canpatternlist1,canpatternlist2,canpatternlist3,canpatternlist4,canpatternlist5)
        canpatternlist=list(set(canpatternlist))
        #print len(canpatternlist),canpatternlist
        if flag!=0:
            tem=np.zeros((template_h,template_x,template_y),int)
        else:
            #print("have")
            #print(len(canpatternlist))
            #print(o1)
            temo=o1*c
            tem=patternsearchAI2(temo,c,cFdatabase[dark][6],canpatternlist,N)
        ms=template1RAI(ms,tem,h1,x1,y1)
            
        
    m=cut(ms,lag)
    return m


def gosiminitialAIzuobiaoverFenji(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,Fenjineironglist):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
    #m为已经导入了Ti的模拟网格
    time_start1=time.time()
    m=extendTimodel(m,template_h,template_x,template_y)#拓展模拟网格
    m2=extendTimodel(m2,template_h,template_x,template_y)#拓展模拟网格
    #np.save('./output/wtf.npy',m)

    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial1.vtk') 

    print('extend done')

 
    my_file = Path("./database/Softdatabase.npy")
    if my_file.exists():
        Softdatabase=np.load('./database/Softdatabase.npy')
        Softzuobiaolist=np.load('./database/Softzuobiaolist.npy')
        Scdatabase=np.load('./database/Scdatabase.npy')
        SCdatabase=np.load('./database/SCdatabase.npy')
        Fdatabaset=np.load('./database/Fdatabase.npy')
        Fzuobiaolist=np.load('./database/Fzuobiaolist.npy')
        CFdatabase=np.load('./database/CFdatabase.npy')
        cFdatabase=np.load('./database/cFdatabase.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Softdatabase,Softzuobiaolist=databasebuildAI(m2,template_h,template_x,template_y)#软数据数据库构建
        Fdatabase,Fzuobiaolist=databasebuildAIFenji(m,template_h,template_x,template_y,Fenjilist,Fenjineironglist)
        #以及分级数据库构建
        Scdatabase=databasecataAI(Softdatabase,lag)
        SCdatabase=databaseclusterAI(Scdatabase,U)
        np.save('./database/cdatabase.npy',Scdatabase)
        np.save('./database/Cdatabase.npy',SCdatabase)
        np.save('./database/Softdatabase.npy',Softdatabase)
        #np.save('./database/Softzuobiaolist.npy',Softzuobiaolist)
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        cFdatabase=[]#分级分类数据库
        CFdatabase=[]#分级分类聚类数据库
        for thesake in range(len(Fenjilist)):
            print(these)
            cdatabase=databasecataAI(Fdatabase[thesake],lag)
            cFdatabase.append(cdatabase)        
            Cdatabase=databaseclusterAI(cdatabase,U)
            CFdatabase.append(Cdatabase)        
        np.save('./database/cFdatabase.npy',cFdatabase)
        np.save('./database/Cdatabase.npy',CFdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    m2=initialgridAI2zuobiaover(m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,SCdatabase,Scdatabase,Softzuobiaolist,N)
    m=initialgridAI2zuobiaoverFenji(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,Fzuobiaolist,N,Fenjilist)
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
    
def GosimAIFenji(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,Fenjineironglist):
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    #m,m2,Tilist,Tilist2,Tizuobiaolist=sectionloadandextendFenji(m,patternSizex,patternSizey,0,1)
    m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIzuobiaoverFenji(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,Fenjineironglist)
    
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
        m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path="./output/reconstruction.npy"
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


def DelphiinitialAIzuobiaoverFenji(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,Fenjineironglist):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
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

 
    my_file = Path("./database/CCFdatabase.npy")
    if my_file.exists():
        Fdatabaset=np.load('./database/Fdatabase.npy')
        Fzuobiaolist=np.load('./database/Fzuobiaolist.npy')
        CFdatabase=np.load('./database/CCFdatabase.npy')
        cFdatabase=np.load('./database/cFdatabase.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Fdatabase,Fzuobiaolist=databasebuildAIFenji(m,template_h,template_x,template_y,Fenjilist,Fenjineironglist)
        #以及分级数据库构建
        
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        cFdatabase=[]#分级分类数据库
        CFdatabase=[]#分级分类聚类数据库
        for thesake in range(len(Fenjilist)):
            print(thesake)
            cdatabase=databasecataAI(Fdatabase[thesake],lag)
            cFdatabase.append(cdatabase)        
            Cdatabase=databaseclusterAI(cdatabase,U)
            CFdatabase.append(Cdatabase)        
        np.save('./database/cFdatabase.npy',cFdatabase)
        np.save('./database/CCFdatabase.npy',CFdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()

    return m,cFdatabase,CFdatabase,Fzuobiaolist


def DelphiFenji(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,Fenjineironglist):
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    #m,m2,Tilist,Tilist2,Tizuobiaolist=sectionloadandextendFenji(m,patternSizex,patternSizey,0,1)
    m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1,30)
    m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m,cFdatabase,CFdatabase,Fzuobiaolist=DelphiinitialAIzuobiaoverFenji(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,Fenjineironglist)
    
    print('initial done')
    return m,cFdatabase,CFdatabase,Fzuobiaolist

def DelphiFenji2(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,Fenjineironglist):
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    #m,m2,Tilist,Tilist2,Tizuobiaolist=sectionloadandextendFenji(m,patternSizex,patternSizey,0,1)
    m,Tilist,Tizuobiaolist,codelist=sectionloadandextendFenji3(m,patternSizex,patternSizey,0,1,30)
    m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m,cFdatabase,CFdatabase,Fzuobiaolist=DelphiinitialAIzuobiaoverFenji(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,Fenjineironglist)
    
    print('initial done')
    return m,cFdatabase,CFdatabase,Fzuobiaolist,codelist

###################序贯模拟重制版
def temcheck(tem):#检测该节点模版大小内是否有待模拟点
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]==-1:
                    return True
    return False
def lujinglistAI2(m,template_h,template_x,template_y,lag):#将所有待模拟网格中空值1的待模拟点加入模拟路径,lag为重叠区大小
    roadlist=[]
    #lagh=template_h//2+1
    #lagx=template_x//2+1
    #lagy=template_y//2+1

    for h in range(0,m.shape[0],lag):
            for x in range(0,m.shape[1],lag):
                for y in range(0,m.shape[2],lag):
                    if temcheck(m[h-lag:h+lag+1,x-lag:x+lag+1,y-lag:y+lag+1]):
                        roadlist.append((h,x,y))
    return roadlist

def temdetectD1(tem):#检测是否包含待模拟点是否大于阈值
    count=0
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]==-1:
                    count=count+1
    if count>=0.5*(tem.shape[0])*(tem.shape[1])*(tem.shape[2]):
        return False
    return True
def temdetectD2(tem):#检测是否包含待模拟点是否大于阈值
    count=0
    for h in range(tem.shape[0]):
        for x in range(tem.shape[1]):
            for y in range(tem.shape[2]):
                if tem[h,x,y]==-1:
                    count=count+1
    if count>=0.8*(tem.shape[0])*(tem.shape[1])*(tem.shape[2]):
        return False
    return True
def initialroadlistAIFenji(m,template_h,template_x,template_y,lag):#改进版
    #lag为重叠区
    #自动初始化网格系统
    lujing=[]
    Roadlist=[]#最终路径名单
    lujing=lujinglistAI2(m,template_h,template_x,template_y,lag-1)
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
            '''
            if temdetectD1(o1[0:lag,:,:]): 
                #上
                k=k+1
            '''
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
                '''
                for hb in range(h1-lag-lag,h1+1):
                    for xb in range(x1-lag-lag,x1+1):
                        for yb in range(y1-lag-lag,y1+1):
                            Banlujing.append((hb,xb,yb))#将已经模拟的区域坐标加入黑名单
                '''
            else:
                lujing.append(lujing[n])
        #print(len(Roadlist),len(lujing)-n)
        n=n+1
        #print len(Roadlist)
    #print('roadlist initial done')
    return Roadlist





def patternsearchFenji(o1,c,database,canpatternlist,N):#根据重叠区,在候选列表中选取备选模式 #直接返回候选模板 ver2.0 
    


    #N为备选模板个数
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    template_h=database[0].shape[0]
    template_x=database[0].shape[1]
    template_y=database[0].shape[2]
    d=[]#备选列表    
    drill1=o1.reshape(ss,1)
    #print len(canpatternlist)
    if len(canpatternlist)<=3000:
       A=len(canpatternlist)
    else:
       A=3000
       shuffle(canpatternlist)
 

    for n in range(A):#取出相似度高的
        ctem=database[canpatternlist[n]]*c
        drill2=ctem.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    si=getListMinNumIndex(d,N)








    dq1=patternSelhi2(o1)

    q=[]#备选列表元素匹配程度
    for n in range(A):#取出元素匹配程度高的
        dq2=patternSelhi2(database[canpatternlist[n]]*c)

        q.append(difflib.SequenceMatcher(None,dq1,dq2).ratio())

    si2=getListMaxNumIndex(q,N)
    tmp = [i for i in si if i in si2]
    if len(tmp)==0:
       tmp=list(set(si).union(set(si2)))
    r=random.randint(0,len(tmp)-1)
    t=tmp[r]

    return database[canpatternlist[t]]

def patternsearchFenji2(o1,c,database,canpatternlist,N,msn,h1,code):#根据重叠区,在候选列表中选取备选模式 #直接返回候选模板 ver2.0 
    #ms为Template_x*template_y*h的待模拟柱
    flag=0
    while flag==0:
        #N为备选模板个数
        ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
        template_h=database[0].shape[0]
        template_x=database[0].shape[1]
        template_y=database[0].shape[2]
        d=[]#备选列表    
        drill1=o1.reshape(ss,1)
        #print len(canpatternlist)
        if len(canpatternlist)<=3000:
            A=len(canpatternlist)
        else:
            A=3000
            shuffle(canpatternlist)
        for n in range(A):#取出相似度高的
            ctem=database[canpatternlist[n]]*c
            drill2=ctem.reshape(ss,1)
            d.append(hamming_distance(drill1,drill2))
        si=getListMinNumIndex(d,N)


        dq1=patternSelhi2(o1)
        q=[]#备选列表元素匹配程度
        for n in range(A):#取出元素匹配程度高的
            dq2=patternSelhi2(database[canpatternlist[n]]*c)

            q.append(difflib.SequenceMatcher(None,dq1,dq2).ratio())

            si2=getListMaxNumIndex(q,N)
        tmp = [i for i in si if i in si2]
        if len(tmp)==0:
            tmp=list(set(si).union(set(si2)))
        r=random.randint(0,len(tmp)-1)
        t=tmp[r]
        tem=database[canpatternlist[t]]
        msn=template1RAI(msn,tem,h1,template_x//2,template_y//2) 
        code897=doyoulikewhatyousee2(msn)
        for ccc in range(len(code897)):
            if code897.all() not in code:
               flag=0
               
            else:
               flag=1
               break
    return tem


def template1RAIFenji(Ti,Ti2,dark,tem,h0,x0,y0):#非柱状模板返还判断硬数据版
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
                if Ti[n1,n2,n3]==-1 and Ti2[n1,n2,n3]==dark:
                    Ti[n1,n2,n3]=tem[nn1,nn2,nn3]
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
        nn1=nn1+1
        nn2=0

        #print(nn2)
    return Ti  #提取坐标x0,y0处模板 





def initialroadlistAIFenjiZ(m,ms,template_h,template_x,template_y,lag,Fenjilist):#改进版,适配只进行区域内模拟的版本
    #lag为重叠区
    #自动初始化网格系统
    lujing=[]
    Roadlist=[]#最终路径名单
    lujing=lujinglistAI2(m,template_h,template_x,template_y,lag-1)
    random.shuffle(lujing)
    #print len(lujing)
    
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(ms,lag,'edge')#拓展
    Fin=m.shape[0]*m.shape[1]*m.shape[2]
    Fin=Fin*1000#最大循环次数
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
            dark=Fenjilist.index(ms2[h1,x1,y1])
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            '''
            if temdetectD1(o1[0:lag,:,:]): 
                #上
                k=k+1
            '''
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
            if (h1>template_h-lag) and (k>=2) and (len(Roadlist)>=10):
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            if (h1>template_h-lag) and (k==1) and (len(Roadlist)<10):
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            elif (h1<=template_h-lag) and (k!=0):
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                '''
                for hb in range(h1-lag-lag,h1+1):
                    for xb in range(x1-lag-lag,x1+1):
                        for yb in range(y1-lag-lag,y1+1):
                            Banlujing.append((hb,xb,yb))#将已经模拟的区域坐标加入黑名单
                '''
            else:
                lujing.append(lujing[n])
        print(len(Roadlist),len(lujing)-n)
        n=n+1
        #print len(Roadlist)
    #print('roadlist initial done')
    return Roadlist




def initialgridAI2Fenji2(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,zuobiaolist,N,Fenjilist):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#拓展
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
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
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
        if temdetectD1(o1[0:lag,:,:]):
            #上
            b=np.zeros((template_h,template_x,template_y),int)
            b[dis,:,:]=1
            c[dis,:,:]=1
            temo=o1*b
            canpatternlist0=patternsearchDi(CFdatabase[dark][5],cFdatabase[dark][5],temo)
            if temdetect0d(o1[0,:,:]):
                flag=1
        '''     

        if temdetectD1(o1[template_h-lag:template_h,:,:]):
            #下
            b=np.zeros((template_h,template_x,template_y),int)
            b[dish,:,:]=1
            c[dish,:,:]=1
            temo=o1*b
            canpatternlist1=patternsearchDi(CFdatabase[dark][0],cFdatabase[dark][0],temo)
            if temdetect0d(o1[template_h-1,:,:]):
                flag=1

        if temdetectD1(o1[:,0:lag,:]):
            #后
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,dis,:]=1
            c[:,dis,:]=1
            temo=o1*b
            canpatternlist2=patternsearchDi(CFdatabase[dark][3],cFdatabase[dark][3],temo)
            if temdetect0d(o1[:,0,:]):
                flag=1

        if temdetectD1(o1[:,template_x-lag:template_x,:]):
            #前
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,disx,:]=1
            c[:,disx,:]=1
            temo=o1*b
            canpatternlist3=patternsearchDi(CFdatabase[dark][4],cFdatabase[dark][4],temo)
            if temdetect0d(o1[:,template_x-1,:]):
                flag=1
 
        if temdetectD1(o1[:,:,0:lag]):
            #左
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,dis]=1
            c[:,:,dis]=1
            temo=o1*b
            canpatternlist4=patternsearchDi(CFdatabase[dark][1],cFdatabase[dark][1],temo)
            if temdetect0d(o1[:,:,0]):
                flag=1

        if temdetectD1(o1[:,:,template_y-lag:template_y]):
            #右
            b=np.zeros((template_h,template_x,template_y),int)
            b[:,:,disy]=1
            c[:,:,disy]=1
            temo=o1*b
            canpatternlist5=patternsearchDi(CFdatabase[dark][2],cFdatabase[dark][2],temo)
            if temdetect0d(o1[:,:,template_y-1]):
                flag=1

                
                
                

        canpatternlist=[]

        Tcanpatternlist=list(set(canpatternlist0).intersection(set(canpatternlist1)))
        if len(Tcanpatternlist)==0:
            canpatternlist=list(set(canpatternlist0).union(set(canpatternlist1)))
        else:
            canpatternlist=Tcanpatternlist

        Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist2)))
        if len(Tcanpatternlist)==0:
            canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
        else:
            canpatternlist=Tcanpatternlist

        Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist3)))
        if len(Tcanpatternlist)==0:
            canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))
        else:
            canpatternlist=Tcanpatternlist

        Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist4)))
        if len(Tcanpatternlist)==0:
            canpatternlist=list(set(canpatternlist).union(set(canpatternlist4)))
        else:
            canpatternlist=Tcanpatternlist

        Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist5)))
        if len(Tcanpatternlist)==0:
            canpatternlist=list(set(canpatternlist).union(set(canpatternlist5)))
        else:
            canpatternlist=Tcanpatternlist


        print(len(canpatternlist))




        #print(len(canpatternlist0),len(canpatternlist1),len(canpatternlist2),len(canpatternlist3),len(canpatternlist4),len(canpatternlist5))
        #print(canpatternlist0,canpatternlist1,canpatternlist2,canpatternlist3,canpatternlist4,canpatternlist5)
        canpatternlist=list(set(canpatternlist))
        #print len(canpatternlist),canpatternlist
        if flag!=0:
            tem=np.zeros((template_h,template_x,template_y),int)

        else:
            #print("have")
            #print(len(canpatternlist))
            #print(o1)
            temo=o1*c
            tem=patternsearchFenji(temo,c,cFdatabase[dark][6],canpatternlist,N)
            #ms=template1RAI(ms,tem,h1,x1,y1)
            #tem= patternsearchFenji(o1,c,database,canpatternlist,N)

        ms=template1RAIFenji(ms,ms2,dark,tem,h1,x1,y1)    
    m=cut(ms,lag)
    return m




def patternsearchFenji3(o1,c,database,canpatternlist,N,msn,h1,code):#根据重叠区,在候选列表中选取备选模式 #直接返回候选模板 ver3.0 
    #ms为Template_x*template_y*h的待模拟柱
    flag=0
    count=0
    while flag==0:
        #N为备选模板个数
        ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
        template_h=database[0].shape[0]
        template_x=database[0].shape[1]
        template_y=database[0].shape[2]
        d=[]#备选列表    
        drill1=o1.reshape(ss,1)
        #print len(canpatternlist)
        if len(canpatternlist)<=3000:
            A=len(canpatternlist)
        else:
            A=30
            shuffle(canpatternlist)
        for n in range(A):#取出相似度高的
            ctem=database[canpatternlist[n]]*c
            drill2=ctem.reshape(ss,1)
            d.append(hamming_distance(drill1,drill2))
        si=getListMinNumIndex(d,N)


        dq1=patternSelhi2(o1)
        q=[]#备选列表元素匹配程度
        for n in range(A):#取出元素匹配程度高的
            dq2=patternSelhi2(database[canpatternlist[n]]*c)

            q.append(difflib.SequenceMatcher(None,dq1,dq2).ratio())

            si2=getListMaxNumIndex(q,N)
        tmp = [i for i in si if i in si2]
        if len(tmp)==0:
            tmp=list(set(si).union(set(si2)))
        r=random.randint(0,len(tmp)-1)
        t=tmp[r]
        tem=database[canpatternlist[t]]
        msn=template1RAI(msn,tem,h1,template_x//2,template_y//2) 
        code897=doyoulikewhatyousee2(msn)
        for ccc in range(len(code897)):
            flag=1
            if  Nocodecheck(code897[ccc],code):
                flag=0
                count=count+1
                break

                
        if count==A:
            return np.zeros((template_h,template_x,template_y),int)
    return tem





def initialgridAI2Fenji3(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,zuobiaolist,N,Fenjilist,code):
    #自动初始化网格系统，分类加速ver3 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIFenji(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#拓展
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
        sqflag=0
        while sqflag==0:
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
            o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
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
            if temdetectD(o1[0:lag,:,:]):
                #上
                b=np.zeros((template_h,template_x,template_y),int)
                b[dis,:,:]=1
                c[dis,:,:]=1
                temo=o1*b
                canpatternlist0=patternsearchDi(CFdatabase[dark][5],cFdatabase[dark][5],temo)
                if temdetect0d(o1[0,:,:]):
                    flag=1
           '''     

            if temdetectD1(o1[template_h-lag:template_h,:,:]):
                #下
                b=np.zeros((template_h,template_x,template_y),int)
                b[dish,:,:]=1
                c[dish,:,:]=1
                temo=o1*b
                canpatternlist1=patternsearchDi(CFdatabase[dark][0],cFdatabase[dark][0],temo)
                if temdetect0d(o1[template_h-1,:,:]):
                    flag=1

            if temdetectD1(o1[:,0:lag,:]):
                #后
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,dis,:]=1
                c[:,dis,:]=1
                temo=o1*b
                canpatternlist2=patternsearchDi(CFdatabase[dark][3],cFdatabase[dark][3],temo)
                if temdetect0d(o1[:,0,:]):
                    flag=1

            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,disx,:]=1
                c[:,disx,:]=1
                temo=o1*b
                canpatternlist3=patternsearchDi(CFdatabase[dark][4],cFdatabase[dark][4],temo)
                if temdetect0d(o1[:,template_x-1,:]):
                    flag=1
 
            if temdetectD1(o1[:,:,0:lag]):
                #左
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,dis]=1
                c[:,:,dis]=1
                temo=o1*b
                canpatternlist4=patternsearchDi(CFdatabase[dark][1],cFdatabase[dark][1],temo)
                if temdetect0d(o1[:,:,0]):
                    flag=1

            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,disy]=1
                c[:,:,disy]=1
                temo=o1*b
                canpatternlist5=patternsearchDi(CFdatabase[dark][2],cFdatabase[dark][2],temo)
                if temdetect0d(o1[:,:,template_y-1]):
                    flag=1

                
                
                

            canpatternlist=[]

            Tcanpatternlist=list(set(canpatternlist0).intersection(set(canpatternlist1)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist0).union(set(canpatternlist1)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist2)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist3)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist4)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist4)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist5)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist5)))
            else:
                canpatternlist=Tcanpatternlist


            print(len(canpatternlist))




            #print(len(canpatternlist0),len(canpatternlist1),len(canpatternlist2),len(canpatternlist3),len(canpatternlist4),len(canpatternlist5))
            #print(canpatternlist0,canpatternlist1,canpatternlist2,canpatternlist3,canpatternlist4,canpatternlist5)
            canpatternlist=list(set(canpatternlist))
            #print len(canpatternlist),canpatternlist
            if flag!=0:
                tem=np.zeros((template_h,template_x,template_y),int)

            else:
                #print("have")
                #print(len(canpatternlist))
                #print(o1)
                temo=o1*c
                #tem=patternsearchFenji(temo,c,cFdatabase[dark][6],canpatternlist,N)
                #ms=template1RAI(ms,tem,h1,x1,y1)
                #tem= patternsearchFenji(o1,c,database,canpatternlist,N)
                tem= patternsearchFenji3(o1,c,cFdatabase[dark][6],canpatternlist,N,ms[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1],h1,code)
            msn=ms[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1]
            msn=template1RAI(msn,tem,h1,template_x//2,template_y//2) 
            code897=doyoulikewhatyousee2(msn)
            for ccc in range(len(code897)):
                sqflag=1
                if Nocodecheck(code897[ccc],code):
                    sqflag=0
                    break

            print(code897,code)
        ms=template1RAI(ms,tem,h1,x1,y1)    
    m=cut(ms,lag)
    return m























###############################改版
'''
#因为无法在有地层穿插情况下使用而废弃
def codecheckredo(list1,codelist):#地层检测机制
    list2=[]
    if len(list1)==1:
        return list2
    for neck in range(len(codelist)):
        if set(list1) <=set(codelist[neck]):
            b = [val for val in codelist[neck] if val in list1]
            for n in range(1,len(list1)):
                if list1.index(b[n-1])>list1.index(b[n]):
                    list2.append(b[n-1])
    return list2
def coderedo(tem,list2):
    for n in range(len(list2)):
        for h in range(tem.shape[0]):
            for x in range(tem.shape[1]):
                for y in range(tem.shape[2]):
                    if tem[h,x,y]==list2[n]:
                        tem[h,x,y]=-1#将有问题的点重置为-1
    return tem

def codechekeredonow(list1,tem,codelist):#检测地层层序错误并替换
    list2=codecheckredo(list1,codelist)
    if len(list2)!=0:
        coderedo(tem,list2)
    return tem

'''

def codecheckZ(list1,codelist):#地层检测机制，如果存在错误则返回真
    list2=[]
    sub=[]#子集
    a=len(list1)
    if a==1:
        return True
    for neck in range(len(codelist)):
        #sub.extend(sum([list(map(list, set(it.combinations(codelist[neck], a)))) for i in range(len(codelist[neck]) + 1)], []))
        sub1=(sorted(set(it.combinations(codelist[neck],r=a))))
        sub2=[list(m) for m in sub1]
        sub=sub+sub2
    #print(sub)
    if list1 not in sub:
        return True
    return False



def checkunreal(m,lag):#检测待模拟点然后返回待模拟点列表
    d=[]
    for h in range(lag,m.shape[0]-lag):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if m[h,x,y]==-1:
                    d.append((h,x,y))
    return d  

def checkunreal2(m,lag):#检测待模拟点然后返回待模拟点列表
    d=[]
    for h in range(lag,m.shape[0]-lag):
        for x in range(lag,m.shape[1]-lag):
            for y in range(lag,m.shape[2]-lag):
                if m[h,x,y]==-2:
                   m[h,x,y]==-1
                if m[h,x,y]==-1:
                    d.append((h,x,y))

    return m,d  
def Templatefenji(Ti,Ti2,dark,tem,h0,x0,y0,hardvaluelist):#非柱状模板返还判断硬数据版
    #hardvaluelist为硬约束的数据

    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    nn1=0
    nn2=0
    nn3=0
    hh=int((template_h)/2)
    xx=int((template_x)/2)
    yy=int((template_y)/2)
    for n1 in range(h0-hh,h0+hh+1):
        for n2 in range(x0-xx,x0+xx+1):
            for n3 in range(y0-yy,y0+yy+1):
                if Ti[n1,n2,n3]==-1 and Ti2[n1,n2,n3]==dark:
                    Ti[n1,n2,n3]=tem[nn1,nn2,nn3]
                    for ro in range(len(hardvaluelist)):
                        if (tem[nn1,nn2,nn3]==hardvaluelist[ro]):
                           Ti[n1,n2,n3]=-2#将该值暂时替换为-2值
                           break
                
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
        nn1=nn1+1
        nn2=0

        #print(nn2)
    return Ti  #提取坐标x0,y0处模板 
def Templatefenji2(Ti,Ti2,dark,tem,h0,x0,y0,hardvaluelist,Fenjineironglist):#非柱状模板返还判断硬数据版
    #hardvaluelist为硬约束的数据

    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    nn1=0
    nn2=0
    nn3=0
    hh=int((template_h)/2)
    xx=int((template_x)/2)
    yy=int((template_y)/2)
    for n1 in range(h0-hh,h0+hh+1):
        for n2 in range(x0-xx,x0+xx+1):
            for n3 in range(y0-yy,y0+yy+1):
                if Ti[n1,n2,n3]==-1 and Ti2[n1,n2,n3]==dark:
                    Ti[n1,n2,n3]=tem[nn1,nn2,nn3]
                    for ro in range(len(hardvaluelist)):
                        if (tem[nn1,nn2,nn3]==hardvaluelist[ro]) and (tem[nn1,nn2,nn3] not in Fenjineironglist[dark]):
                           Ti[n1,n2,n3]=-2#将该值暂时替换为-2值
                           break
                
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
        nn1=nn1+1
        nn2=0

        #print(nn2)
    return Ti  #提取坐标x0,y0处模板 


def subroadlistinitialagain(m,ms2,lujing,template_h,template_x,template_y,lag,Fenjilist):#改进版
    #lag为重叠区
    #自动初始化网格系统

    Roadlist=[]#最终路径名单

    random.shuffle(lujing)
    #print len(lujing)
    
    #print len(lujing)
    m2=m.copy()

    DevilTrigger=False

    Banlujing=[]
    b=np.zeros((template_h,template_x,template_y),int)
    n=0
    
    while n<len(lujing):
        if m2[lujing[n][0],lujing[n][1],lujing[n][2]]==-1:
            #if lujing[n] not in Banlujing:
            h1=lujing[n][0]
            x1=lujing[n][1]
            y1=lujing[n][2]
            dark=Fenjilist.index(ms2[h1,x1,y1])
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            '''
            if temdetectD1(o1[0:lag,:,:]): 
                #上
                k=k+1
            '''
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
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            elif (h1<=template_h-lag) and (k!=0):
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            else:
                lujing.append(lujing[n])
        #print(len(Roadlist),len(lujing)-n)
        n=n+1
        #print len(Roadlist)
    #print('roadlist initial done')
    return Roadlist




def oldStep2initial(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,zuobiaolist,N,Fenjilist,code,hvaluelist):
    #自动初始化网格系统，分类加速ver3 直接载入路径版
    #code为地层层序数据库
    #hvaluelist为模拟返回值
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#拓展
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
    reb=-np.ones((ms.shape[0]),int)
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
        
    sqflag=0    
    #############################################
    while sqflag==0:
        sqflag=0
        #relist=[]
        #print(len(lujing))
        for n in range(len(lujing)):
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
            o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            flag=0
            canpatternlist0=[]
            canpatternlist1=[]
            canpatternlist2=[]
            canpatternlist3=[]
            canpatternlist4=[]
            canpatternlist5=[]
            c=np.zeros((template_h,template_x,template_y),int)
            

            if temdetectD1(o1[template_h-lag:template_h,:,:]):
                #下
                b=np.zeros((template_h,template_x,template_y),int)
                b[dish,:,:]=1
                c[dish,:,:]=1
                temo=o1*b
                canpatternlist1=patternsearchDi(CFdatabase[dark][0],cFdatabase[dark][0],temo)
                if temdetect0d(o1[template_h-1,:,:]):
                    flag=1

            if temdetectD1(o1[:,0:lag,:]):
                #后
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,dis,:]=1
                c[:,dis,:]=1
                temo=o1*b
                canpatternlist2=patternsearchDi(CFdatabase[dark][3],cFdatabase[dark][3],temo)
                if temdetect0d(o1[:,0,:]):
                    flag=1

            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,disx,:]=1
                c[:,disx,:]=1
                temo=o1*b
                canpatternlist3=patternsearchDi(CFdatabase[dark][4],cFdatabase[dark][4],temo)
                if temdetect0d(o1[:,template_x-1,:]):
                    flag=1
 
            if temdetectD1(o1[:,:,0:lag]):
                #左
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,dis]=1
                c[:,:,dis]=1
                temo=o1*b
                canpatternlist4=patternsearchDi(CFdatabase[dark][1],cFdatabase[dark][1],temo)
                if temdetect0d(o1[:,:,0]):
                    flag=1

            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,disy]=1
                c[:,:,disy]=1
                temo=o1*b
                canpatternlist5=patternsearchDi(CFdatabase[dark][2],cFdatabase[dark][2],temo)
                if temdetect0d(o1[:,:,template_y-1]):
                    flag=1

                
                
                

            canpatternlist=[]

            Tcanpatternlist=list(set(canpatternlist0).intersection(set(canpatternlist1)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist0).union(set(canpatternlist1)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist2)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist3)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist4)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist4)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist5)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist5)))
            else:
                canpatternlist=Tcanpatternlist


            #print(len(canpatternlist))




            #print(len(canpatternlist0),len(canpatternlist1),len(canpatternlist2),len(canpatternlist3),len(canpatternlist4),len(canpatternlist5))
            #print(canpatternlist0,canpatternlist1,canpatternlist2,canpatternlist3,canpatternlist4,canpatternlist5)
            canpatternlist=list(set(canpatternlist))
            #print len(canpatternlist),canpatternlist
            if flag!=0:
                tem=np.zeros((template_h,template_x,template_y),int)
                #tem=patternsearchFenji(o1,c,cFdatabase[dark][6],canpatternlist,N)
                #print('thatisthepoint')
            else:
                #print("have")
                #print(len(canpatternlist))
                #print(o1)
                temo=o1*c
                tem=patternsearchFenji(o1,c,cFdatabase[dark][6],canpatternlist,N)
                #ms=template1RAI(ms,tem,h1,x1,y1)
                #tem= patternsearchFenji(o1,c,database,canpatternlist,N)
                #tem= patternsearchFenji3(o1,c,cFdatabase[dark][6],canpatternlist,N,ms[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1],h1,code)
            #ms=template1RAI(ms,tem,h1,x1,y1)
            ms=Templatefenji(ms,ms2,dark,tem,h1,x1,y1,hvaluelist) 
            '''
            msn=ms[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1]
            msn2=ms2[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1]
            msn=Templatefenji(msn,msn2,dark,tem,h1,template_x//2,template_y//2,hvaluelist) 
            code897=doyoulikewhatyousee2(msn)
    
            ms[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1]=msn
            for six in range(len(code897)):
                #print(code897[six],code)
                if codecheckZ(code897[six],code):
                   if (x1,y1) not in relist:
           
                      relist.append((x1,y1))
            '''
        relist=[]
        '''
        data=ms.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),  dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial3.vtk') 
        print('beeeeeforeoutput')    
        '''
        for x2 in range(lag,X+lag):  
            for y2 in range(lag,Y+lag):
                
                code897=doyoulikewhatyousee3(ms[lag:H+lag,x2,y2])
                if codecheckZ(code897,code):
                   
                   if (x2,y2) not in relist:
                      print((x2,y2))
                      print(code897)
                      relist.append((x2,y2))
        print(len(relist))
        #print(code)
        for n in range(len(relist)):
            ms[:,relist[n][0],relist[n][1]]=reb
        '''    
        data=ms.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),  dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial4.vtk') 
        print('output')
        '''
        lujing=[]
        disss=[]    
        disss=checkunreal(ms,lag)
        if len(disss)==0:
            sqflag=1
        else:
            
            lujing=subroadlistinitialagain(ms,ms2,disss,template_h,template_x,template_y,lag,Fenjilist)
    m=cut(ms,lag)
    return m




def Step2initial(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,zuobiaolist,N,Fenjilist,Fenjineironglist,code,hvaluelist):
    #自动初始化网格系统，分类加速ver3 直接载入路径版
    #code为地层层序数据库
    #hvaluelist为模拟返回值
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#拓展
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
    reb=-np.ones((ms.shape[0]),int)
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
        
    sqflag=0    
    #############################################
    while sqflag==0:
        sqflag=0
        #relist=[]
        #print(len(lujing))
        for n in range(len(lujing)):
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号    
 
            #print(dark)
            o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            flag=0
            canpatternlist0=[]
            canpatternlist1=[]
            canpatternlist2=[]
            canpatternlist3=[]
            canpatternlist4=[]
            canpatternlist5=[]
            c=np.zeros((template_h,template_x,template_y),int)
            

            if temdetectD1(o1[template_h-lag:template_h,:,:]):
                #下
                b=np.zeros((template_h,template_x,template_y),int)
                b[dish,:,:]=1
                c[dish,:,:]=1
                temo=o1*b
                canpatternlist1=patternsearchDi(CFdatabase[dark][0],cFdatabase[dark][0],temo)
                if temdetect0d(o1[template_h-1,:,:]):
                    flag=1

            if temdetectD1(o1[:,0:lag,:]):
                #后
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,dis,:]=1
                c[:,dis,:]=1
                temo=o1*b
                canpatternlist2=patternsearchDi(CFdatabase[dark][3],cFdatabase[dark][3],temo)
                if temdetect0d(o1[:,0,:]):
                    flag=1

            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,disx,:]=1
                c[:,disx,:]=1
                temo=o1*b
                canpatternlist3=patternsearchDi(CFdatabase[dark][4],cFdatabase[dark][4],temo)
                if temdetect0d(o1[:,template_x-1,:]):
                    flag=1
 
            if temdetectD1(o1[:,:,0:lag]):
                #左
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,dis]=1
                c[:,:,dis]=1
                temo=o1*b
                canpatternlist4=patternsearchDi(CFdatabase[dark][1],cFdatabase[dark][1],temo)
                if temdetect0d(o1[:,:,0]):
                    flag=1

            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                b=np.zeros((template_h,template_x,template_y),int)
                b[:,:,disy]=1
                c[:,:,disy]=1
                temo=o1*b
                canpatternlist5=patternsearchDi(CFdatabase[dark][2],cFdatabase[dark][2],temo)
                if temdetect0d(o1[:,:,template_y-1]):
                    flag=1

                
                
                

            canpatternlist=[]

            Tcanpatternlist=list(set(canpatternlist0).intersection(set(canpatternlist1)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist0).union(set(canpatternlist1)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist2)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist2)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist3)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist3)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist4)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist4)))
            else:
                canpatternlist=Tcanpatternlist

            Tcanpatternlist=list(set(canpatternlist).intersection(set(canpatternlist5)))
            if len(Tcanpatternlist)==0:
                canpatternlist=list(set(canpatternlist).union(set(canpatternlist5)))
            else:
                canpatternlist=Tcanpatternlist


            #print(len(canpatternlist))




            #print(len(canpatternlist0),len(canpatternlist1),len(canpatternlist2),len(canpatternlist3),len(canpatternlist4),len(canpatternlist5))
            #print(canpatternlist0,canpatternlist1,canpatternlist2,canpatternlist3,canpatternlist4,canpatternlist5)
            canpatternlist=list(set(canpatternlist))
            #print(len(canpatternlist),canpatternlist)
            if flag!=0:
                #######tem=np.zeros((template_h,template_x,template_y),int)
                tem=patternsearchFenji(o1,c,cFdatabase[dark][6],canpatternlist,N)
                #print('thatisthepoint')
            else:
                #print("have")
                #print(len(canpatternlist))
                #print(o1)
                temo=o1*c
                tem=patternsearchFenji(o1,c,cFdatabase[dark][6],canpatternlist,N)
                #ms=template1RAI(ms,tem,h1,x1,y1)
                #tem= patternsearchFenji(o1,c,database,canpatternlist,N)
                #tem= patternsearchFenji3(o1,c,cFdatabase[dark][6],canpatternlist,N,ms[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1],h1,code)
            #ms=template1RAI(ms,tem,h1,x1,y1)
            ms=Templatefenji2(ms,ms2,dark,tem,h1,x1,y1,hvaluelist,Fenjineironglist) 
            '''
            msn=ms[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1]
            msn2=ms2[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1]
            msn=Templatefenji(msn,msn2,dark,tem,h1,template_x//2,template_y//2,hvaluelist) 
            code897=doyoulikewhatyousee2(msn)
    
            ms[:,x1-lag:x1+lag+1,y1-lag:y1+lag+1]=msn
            for six in range(len(code897)):
                #print(code897[six],code)
                if codecheckZ(code897[six],code):
                   if (x1,y1) not in relist:
           
                      relist.append((x1,y1))
            '''
        relist=[]
        '''
        data=ms.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),  dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial3.vtk') 
        print('beeeeeforeoutput')    
        '''
        for x2 in range(lag,X+lag):  
            for y2 in range(lag,Y+lag):
                
                code897=doyoulikewhatyousee3(ms[lag:H+lag,x2,y2])
                if codecheckZ(code897,code):
                   
                   if (x2,y2) not in relist:
                      #print((x2,y2))
                      #print(code897)
                      relist.append((x2,y2))
        print(len(relist))
        #print(code)
        for n in range(len(relist)):
            ms[:,relist[n][0],relist[n][1]]=reb
        '''    
        data=ms.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),  dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial4.vtk') 
        print('output')
        '''
        lujing=[]
        disss=[]    
        ms,disss=checkunreal2(ms,lag)
        if len(disss)==0:
            sqflag=1
        else:
            
            lujing=subroadlistinitialagain(ms,ms2,disss,template_h,template_x,template_y,lag,Fenjilist)
    m=cut(ms,lag)
    return m



######################################普通非分级模型初始化构建
def TemplateHard(Ti,tem,h0,x0,y0,hardvaluelist):#非柱状模板返还判断硬数据版
    #hardvaluelist为硬约束的数据

    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    nn1=0
    nn2=0
    nn3=0
    hh=int((template_h)//2)
    xx=int((template_x)//2)
    yy=int((template_y)//2)
    for n1 in range(h0-hh,h0+hh+1):
        for n2 in range(x0-xx,x0+xx+1):
            for n3 in range(y0-yy,y0+yy+1):
                if Ti[n1,n2,n3]==-1:
                    Ti[n1,n2,n3]=tem[nn1,nn2,nn3]
                    for ro in range(len(hardvaluelist)):
                        if (tem[nn1,nn2,nn3]==hardvaluelist[ro]):
                            Ti[n1,n2,n3]=-2
                            break
                
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
        nn1=nn1+1
        nn2=0
        #print(nn2)
    return Ti  #提取坐标x0,y0处模板 



def subroadlistinitialfornew(m2,lujing,template_h,template_x,template_y,lag):#改进版
    #lag为重叠区
    #自动初始化网格系统

    Roadlist=[]#最终路径名单

    random.shuffle(lujing)
    #print len(lujing)
    
    #print len(lujing)
    m3=m2.copy()

    DevilTrigger=False

    Banlujing=[]
    b=np.zeros((template_h,template_x,template_y),int)
    n=0
    
    while n<len(lujing):
        if m2[lujing[n][0],lujing[n][1],lujing[n][2]]==-1:
            #if lujing[n] not in Banlujing:
            h1=lujing[n][0]
            x1=lujing[n][1]
            y1=lujing[n][2]

            o1=template1(m3,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            '''
            if temdetectD1(o1[0:lag,:,:]): 
                #上
                k=k+1
            '''
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
                m2=template1RAI(m3,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            elif (h1<=template_h-lag) and (k!=0):
                m2=template1RAI(m3,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
            else:
                lujing.append(lujing[n])
        #print(len(Roadlist),len(lujing)-n)
        n=n+1
        #print len(Roadlist)
    #print('roadlist initial done')
    return Roadlist


def initialgridAIfornew(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N,codelist,hardlist):
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
            m2[:,relist[n][0],relist[n][1]]=reb
        '''    
        data=ms.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),  dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial4.vtk') 
        print('output')
        '''
        lujing=[]
        disss=[]    
        ms,disss=checkunreal2(m2,lag)
        if len(disss)==0:
            sqflag=1
        else:
            
            lujing=subroadlistinitialfornew(m2,disss,template_h,template_x,template_y,lag)

            
        
    m=cut(m2,lag)
    return m

def gosiminitialAIfornew(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,hardlist,code):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格
    #m为已经导入了Ti的模拟网格
    time_start1=time.time()
    #m=extendTimodel(m,template_h,template_x,template_y)#拓展模拟网格


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
    m=initialgridAIfornew(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Cdatabase,cdatabase,zuobiaolist,N,code,hardlist)
    
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







#####################################################16种模版的数据库
def databaseclusterAI2(cdatabase,U):#模式分类
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

    p8= multiprocessing.Process(target=Simplecluster, args=(cdatabase[7],U,7)) 
    print('process start') 
    p8.start()

    p9= multiprocessing.Process(target=Simplecluster, args=(cdatabase[8],U,8)) 
    print('process start') 
    p9.start()

    p10= multiprocessing.Process(target=Simplecluster, args=(cdatabase[9],U,9)) 
    print('process start') 
    p10.start()

    p11= multiprocessing.Process(target=Simplecluster, args=(cdatabase[10],U,10)) 
    print('process start') 
    p11.start()

    p12= multiprocessing.Process(target=Simplecluster, args=(cdatabase[11],U,11)) 
    print('process start') 
    p12.start()

    p13= multiprocessing.Process(target=Simplecluster, args=(cdatabase[12],U,12)) 
    print('process start') 
    p13.start()

    p14= multiprocessing.Process(target=Simplecluster, args=(cdatabase[13],U,13)) 
    print('process start') 
    p14.start()

    p15= multiprocessing.Process(target=Simplecluster, args=(cdatabase[14],U,14)) 
    print('process start') 
    p15.start()

    p16= multiprocessing.Process(target=Simplecluster, args=(cdatabase[15],U,15)) 
    print('process start') 
    p16.start()
    

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    p15.join()
    p16.join()

    print('process end')

    Cdatabase=[]
    cc1=np.load('./database/clusters0.npy')
    cc2=np.load('./database/clusters1.npy')
    cc3=np.load('./database/clusters2.npy')
    cc4=np.load('./database/clusters3.npy')
    cc5=np.load('./database/clusters4.npy')
    cc6=np.load('./database/clusters5.npy')
    cc7=np.load('./database/clusters6.npy')
    cc8=np.load('./database/clusters7.npy')
    cc9=np.load('./database/clusters8.npy')
    cc10=np.load('./database/clusters9.npy')
    cc11=np.load('./database/clusters10.npy')
    cc12=np.load('./database/clusters11.npy')
    cc13=np.load('./database/clusters12.npy')
    cc14=np.load('./database/clusters13.npy')
    cc15=np.load('./database/clusters14.npy')
    cc16=np.load('./database/clusters15.npy')

    Cdatabase.append(cc1)
    Cdatabase.append(cc2)
    Cdatabase.append(cc3)
    Cdatabase.append(cc4)
    Cdatabase.append(cc5)
    Cdatabase.append(cc6)
    Cdatabase.append(cc7)
    Cdatabase.append(cc8)
    Cdatabase.append(cc9)
    Cdatabase.append(cc10)
    Cdatabase.append(cc11)
    Cdatabase.append(cc12)
    Cdatabase.append(cc13)
    Cdatabase.append(cc14)
    Cdatabase.append(cc15)
    Cdatabase.append(cc16)
    os.remove('./database/clusters0.npy')
    os.remove('./database/clusters1.npy')
    os.remove('./database/clusters2.npy')
    os.remove('./database/clusters3.npy')
    os.remove('./database/clusters4.npy')
    os.remove('./database/clusters5.npy')
    os.remove('./database/clusters6.npy')
    os.remove('./database/clusters7.npy')
    os.remove('./database/clusters8.npy')
    os.remove('./database/clusters9.npy')
    os.remove('./database/clusters10.npy')
    os.remove('./database/clusters11.npy')
    os.remove('./database/clusters12.npy')
    os.remove('./database/clusters13.npy')
    os.remove('./database/clusters14.npy')
    os.remove('./database/clusters15.npy')
    
    #np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase

def initialpatternbaseFENJI(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,Fenjineironglist):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
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

 
    my_file = Path("./database/CCFdatabase.npy")
    if my_file.exists():
        Fdatabaset=np.load('./database/Fdatabase.npy')
        Fzuobiaolist=np.load('./database/Fzuobiaolist.npy')
        CFdatabase=np.load('./database/CCFdatabase.npy')
        cFdatabase=np.load('./database/cFdatabase.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Fdatabase,Fzuobiaolist=databasebuildAIFenji(m,template_h,template_x,template_y,Fenjilist,Fenjineironglist)
        #以及分级数据库构建
        
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        cFdatabase=[]#分级分类数据库
        CFdatabase=[]#分级分类聚类数据库
        for thesake in range(len(Fenjilist)):
            print(thesake)
            cdatabase=temdatabasec_5(Fdatabase[thesake],lag,template_h,template_x,template_y)
            cFdatabase.append(cdatabase)        
            Cdatabase=databaseclusterAI2(cdatabase,U)
            CFdatabase.append(Cdatabase)        
        np.save('./database/cFdatabase.npy',cFdatabase)
        np.save('./database/CCFdatabase.npy',CFdatabase)
        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()

    return m,cFdatabase,CFdatabase,Fzuobiaolist

def flagpanbieqi(down, back, front,left,right):#判别类别
    if down==0:
        flag=7
        if (back==1) and (front==0) and (left==1) and (right==0):
            flag=0
        if (back==0) and (front==1) and (left==1) and (right==0):
            flag=1
        if (back==0) and (front==1) and (left==0) and (right==1):
            flag=2
        if (back==1) and (front==0) and (left==0) and (right==1):
            flag=3
        if (back==1) and (front==1) and (left==0) and (right==1):
            flag=4
        if (back==1) and (front==1) and (left==1) and (right==0):
            flag=5
        if (back==1) and (front==0) and (left==1) and (right==1):
            flag=6
        if (back==0) and (front==1) and (left==1) and (right==1):
            flag=7
        
    elif down==1: 
        flag=15
        if (back==1) and (front==0) and (left==1) and (right==0):
            flag=8
        if (back==0) and (front==1) and (left==1) and (right==0):
            flag=9
        if (back==0) and (front==1) and (left==0) and (right==1):
            flag=10
        if (back==1) and (front==0) and (left==0) and (right==1):
            flag=11
        if (back==1) and (front==1) and (left==0) and (right==1):
            flag=12
        if (back==1) and (front==1) and (left==1) and (right==0):
            flag=13
        if (back==1) and (front==0) and (left==1) and (right==1):
            flag=14
        if (back==0) and (front==1) and (left==1) and (right==1):
            flag=15

    return flag
    
def initialroadlistAIFenji2(m,template_h,template_x,template_y,lag):#改进版
    #lag为重叠区
    #自动初始化网格系统
    lujing=[]
    Roadlist=[]#最终路径名单
    roadlistflag=[]#节点模版类别
    lujing=lujinglistAI2(m,template_h,template_x,template_y,lag-1)
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

            down=0
            back=0
            front=0
            left=0
            right=0



            '''
            if temdetectD(o1[0:lag,:,:]): 
                #上
                k=k+1
            '''
            if temdetectD1(o1[template_h-lag:template_h,:,:]):
                #下
                k=k+1
                down=1
            if temdetectD1(o1[:,0:lag,:]):
                #后
                k=k+1
                back=1
            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前 
                k=k+1
                front=1
            if temdetectD1(o1[:,:,0:lag]):
                #左
                k=k+1
                left=1
            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                k=k+1
                right=1

            if (h1>template_h-lag) and (k>=2):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                roadlistflag.append(flagpanbieqi(down, back, front,left,right))
            elif (h1<=template_h-lag) and (k!=0):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                roadlistflag.append(flagpanbieqi(down, back, front,left,right))
                '''
                for hb in range(h1-lag-lag,h1+1):
                    for xb in range(x1-lag-lag,x1+1):
                        for yb in range(y1-lag-lag,y1+1):
                            Banlujing.append((hb,xb,yb))#将已经模拟的区域坐标加入黑名单
                '''
            else:
                lujing.append(lujing[n])
        #print(len(Roadlist),len(lujing)-n)
        n=n+1
        #print len(Roadlist)
    #print('roadlist initial done')
    return Roadlist,roadlistflag

def patternsearch16(Cdatabase,cdatabase,tem):#初始模型构建中依据重叠区用编辑距离搜索最适合模板的代号,Cdatabase为分类好的数据库
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

def initialgridAI2Fenji2333(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,zuobiaolist,N,Fenjilist):
    #自动初始化网格系统，分类加速ver2 直接载入路径版
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing,lujingflag=initialroadlistAIFenji2(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#拓展
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
    c=np.ones((template_h,template_x,template_y),int)
  
    for n in range(len(lujing)):
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
        k=0#重叠区计数器
        flag=lujingflag[n]

        canpatternlist=patternsearch16(CFdatabase[dark][flag],cFdatabase[dark][flag],o1)    #待修改为其他计算相似度公式


        print(len(canpatternlist))
        tem=patternsearchAI2(o1,c,cFdatabase[dark][16],canpatternlist,N)


        ms=template1RAI(ms,tem,h1,x1,y1)
            
        
    m=cut(ms,lag)
    return m


def DelphiFenji222(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,Fenjineironglist):
    time_starts=time.time()#计时开始
    #初始模型构建 m为已经构建好的模拟网格
    #m,m2,Tilist,Tilist2,Tizuobiaolist=sectionloadandextendFenji(m,patternSizex,patternSizey,0,1)
    m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m,cFdatabase,CFdatabase,Fzuobiaolist=initialpatternbaseFENJI(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,Fenjineironglist)
    
    print('initial done')
    return m,cFdatabase,CFdatabase,Fzuobiaolist

#####################################################
