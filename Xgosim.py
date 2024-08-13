#!/usr/bin/env python
# coding: utf-8

# In[2]:


######################ver0.1
import numpy as np
from numpy import*
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

def databasebuildAIClass(Exm,Exmf,template_h,template_x,template_y,Fenjilist):#智能构建模式库
    #Exm为未完成了拓展的模拟网格
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
                            if Exmf[h,x,y] == Fenjilist[soul]:
                                database.append(tem)
                                zuobiaolist.append((h,x,y))
        Fdatabase.append(database)#标号与分级列表标号一致
        Fzuobiaolist.append(zuobiaolist)
    return Fdatabase,Fzuobiaolist


def initialgridAI2Class(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,CFdatabase,cFdatabase,zuobiaolist,N,Fenjilist):
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

def gosiminitialAIClass(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist):
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
        Fdatabase,Fzuobiaolist=databasebuildAIClass(m,m2,template_h,template_x,template_y,Fenjilist)
        #以及分级数据库构建
        Scdatabase=databasecataAI(Softdatabase,lag)
        SCdatabase=databaseclusterAI(Scdatabase,U)
        np.save('./database/Scdatabase.npy',Scdatabase)
        np.save('./database/SCdatabase.npy',SCdatabase)
        np.save('./database/Softdatabase.npy',Softdatabase)
        np.save('./database/Softzuobiaolist.npy',Softzuobiaolist)
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        cFdatabase=[]#分级分类数据库
        CFdatabase=[]#分级分类聚类数据库
        for thesake in range(len(Fenjilist)):
            #print(these)
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

def GosimAIClass(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist):
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    m,m2,Tilist,Tizuobiaolist=sectionloadandextendFenji(m,patternSizex,patternSizey,0,1)
    #m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    #m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m2,patternSizex,patternSizey,0,1)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIClass(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist)
    
    print('initial done')


    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)

        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=Recodesectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)


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
        m= patchmatchmultiTiB(m,Tilist,size,itr,1)
        CTI=[]#检测使用率剖面
        #m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,8,0)#并行进程的数目
        #计算量加大过多
        #m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
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


    
    
    
    
    
    
    
##################################优化版#######################################################
def databasebuildAIY(Exm,template_h,template_x,template_y,lag):#智能构建模式库优化版
    #Exm为已经完成了拓展的模拟网格

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
    return Exm2,zuobiaolist




def databasebuildAIClassY(Exm,Exmf,template_h,template_x,template_y,lag,Fenjilist):#智能构建模式库
    #Exm为未完成了拓展的模拟网格

    Exm2=np.pad(Exm,lag,'edge')#拓展

    Fzuobiaolist=[]

    for soul in range(len(Fenjilist)):
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
                            if Exmf[h,x,y] == Fenjilist[soul]:
                                zuobiaolist.append((h,x,y))
        #标号与分级列表标号一致
        print(soul,len(zuobiaolist))
        Fzuobiaolist.append(zuobiaolist)
    return Exm2,Fzuobiaolist

def SimpleclusterY(database,zuobiaolist,template_h,template_x,template_y,lag,U,name):#简单聚类方法
    #U为阈值
    Cdatabase=[]
    d=[]
    c=[]
    for n in range(len(zuobiaolist)):
        if n not in c:
            d=[]
            h0=zuobiaolist[n][0]+lag
            x0=zuobiaolist[n][1]+lag
            y0=zuobiaolist[n][2]+lag
            d.append(n)
            c.append(n)
            for m in range(n+1,len(zuobiaolist)):
                h1=zuobiaolist[m][0]+lag
                x1=zuobiaolist[m][1]+lag
                y1=zuobiaolist[m][2]+lag
                if cluster_distance(template1(database,template_h,template_x,template_y,h0,x0,y0), template1(database,template_h,template_x,template_y,h1,x1,y1))<=U:
                    d.append(m)
                    c.append(m)
            Cdatabase.append(d)
    #print len(c)
    np.save('./database/clusters'+str(name)+'.npy',Cdatabase)
    return Cdatabase
    #结果只保存编号

def databaseclusterAIY(Exm1,Exm2,Softzuobiaolist,Fzuobiaolist,template_h,template_x,template_y,lag,U):#模式分类 Exm1为分类总类别拓展模式库，Exm2为分类拓展模式库
    #优化版
    print('start')


    p1= multiprocessing.Process(target=SimpleclusterY, args=(Exm1,Softzuobiaolist,template_h,template_x,template_y,lag,U,'S')) 
    print('softdata process start') 
    p1.start()
    processes=list()
    for n in range(len(Fzuobiaolist)):
        print(n,'process start')
        p=multiprocessing.Process(target=SimpleclusterY,args=(Exm2,Fzuobiaolist[n],template_h,template_x,template_y,lag,U,n))
        p.start()
        processes.append(p)
    p1.join()
    for p in processes:
        p.join()


    

    
    time.sleep(5)
    print('process end')
    Sclass=np.load('./database/clustersS.npy')
    Fclass=[]
    for n in range(len(Fzuobiaolist)):
        path='./database/clusters'+str(n)+'.npy'
        cc=np.load(path)
        Fclass.append(cc)


    
    #np.save('./database/Cdatabase.npy',Cdatabase)
    return Sclass,Fclass
def patternsearchNY(tem,Exm,Sclass,zuobiaolist,lag,N):
    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    c1=99999
    cc=0
    
    for n in range(len(Sclass)):#选取距离最近的类别
        #print(len(Sclass[n]))
        rrr=random.randint(0,len(Sclass[n])-1)
        h0=zuobiaolist[Sclass[n][rrr]][0]+lag
        x0=zuobiaolist[Sclass[n][rrr]][1]+lag
        y0=zuobiaolist[Sclass[n][rrr]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
            c1=fun
            cc=n
    d=[]
    p=[]
    '''
    for n in range(len(Sclass[cc])):

        h0=zuobiaolist[Sclass[cc][n]][0]+lag
        x0=zuobiaolist[Sclass[cc][n]][1]+lag
        y0=zuobiaolist[Sclass[cc][n]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    '''
    for n in range(min(len(Sclass[cc]),1000)):
        rrr=random.randint(0,len(Sclass[cc])-1)
        h0=zuobiaolist[Sclass[cc][rrr]][0]+lag
        x0=zuobiaolist[Sclass[cc][rrr]][1]+lag
        y0=zuobiaolist[Sclass[cc][rrr]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
        p.append(rrr)
    #N选1
    si=getListMinNumIndex(d,N)
    #print(len(si),si)
    r=random.randint(0,len(si)-1)
    assassin=si[r]
    t=p[assassin]
    h0=zuobiaolist[Sclass[cc][t]][0]+lag
    x0=zuobiaolist[Sclass[cc][t]][1]+lag
    y0=zuobiaolist[Sclass[cc][t]][2]+lag
    resulttem=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
    return resulttem

def initialgridAIY(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm,Sclass,zuobiaolist,N,hardlist):
    #优化版，Exm为拓展数据库
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAI(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    m2=Exm

    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]

        
        
    #############################################
    sqflag=0
    while sqflag==0:
        sqflag=0
        for n in range(len(lujing)):
            print(n,len(lujing))
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            flag=0
            tem=patternsearchNY(o1,Exm,Sclass,zuobiaolist,lag,N)
            
            
          

            #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
            m2=template1R(m2,tem,h1,x1,y1)
        sqflag=1
        '''
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

        lujing=[]
        disss=[]    
        ms,disss=checkunreal2(m2,lag)
        if len(disss)==0:
            sqflag=1
        else:
            
            lujing=subroadlistinitialfornew(m2,disss,template_h,template_x,template_y,lag)
        '''
            
        
    m=cut(m2,lag)
    return m

def patternsearchClassNY(o1,Exm2,Fclass,Fzuobiaolist,dark,lag,N):
    template_h=o1.shape[0]
    template_x=o1.shape[1]
    template_y=o1.shape[2]
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    drill1=o1.reshape(ss,1)
    c1=99999
    cc=0
    for n in range(len(Fclass[dark])):#选取距离最近的类别
        rrr=random.randint(0,len(Fclass[dark][n])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
            c1=fun
            cc=n
    d=[]
    p=[]
    for n in range(min(len(Fclass[dark][cc]),1000)):
        rrr=random.randint(0,len(Fclass[dark][cc])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
        p.append(rrr)
    
    #N选1
    si=getListMinNumIndex(d,N)
    r=random.randint(0,len(si)-1)
    t=si[r]
    t=p[t]
    h0=Fzuobiaolist[dark][Fclass[dark][cc][t]][0]+lag
    x0=Fzuobiaolist[dark][Fclass[dark][cc][t]][1]+lag
    y0=Fzuobiaolist[dark][Fclass[dark][cc][t]][2]+lag

    resulttem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
    return resulttem

def initialgridAIClassY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm1,Exm2,Fzuobiaolist,Fclass,N,Fenjilist,hardlist):
    #优化版处理分级数据
    #此处m2为软数据
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAI(m,template_h,template_x,template_y,lag)
    #lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=Exm2#拓展
    ms2=Exm1#软数据拓展
    
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
  
    
  
    for n in range(len(lujing)):
        print(n,len(lujing))
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
        k=0#重叠区计数器
        flag=0
        tem=patternsearchClassNY(o1,ms,Fclass,Fzuobiaolist,dark,lag,N)
            
            
       

        #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
        ms=template1R(ms,tem,h1,x1,y1)   
        
    #m=cut(ms,lag)
    m=ms[lag:m.shape[0]+lag,lag:m.shape[1]+lag,lag:m.shape[2]+lag]
    return m

def gosiminitialAIClassY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
    #m为已经导入了Ti的模拟网格
    #优化版
    hardlist=[]
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

 
    my_file = Path("./database/Sclass.npy")
    if my_file.exists():
        Softdatabase=np.load('./database/Sdatabase.npy')
        Softzuobiaolist=np.load('./database/Softzuobiaolist.npy')
        Fdatabase=np.load('./database/Fdatabase.npy')
        Fzuobiaolist=np.load('./database/Fzuobiaolist.npy')
        Sclass=np.load('./database/Sclass.npy')
        Fclass=np.load('./database/Fclass.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Softdatabase,Softzuobiaolist=databasebuildAIY(m2,template_h,template_x,template_y,lag)#软数据数据库构建
        np.save('./database/Sdatabase.npy',Softdatabase)#softdata为扩充后的网格
        np.save('./database/Softzuobiaolist.npy',Softzuobiaolist)
        Fdatabase,Fzuobiaolist=databasebuildAIClassY(m,m2,template_h,template_x,template_y,lag,Fenjilist)
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        
        #以及分级数据库构建
        Sclass,Fclass=databaseclusterAIY(Softdatabase,Fdatabase,Softzuobiaolist,Fzuobiaolist,template_h,template_x,template_y,lag,U)
        
        print(len(Sclass))
        for n in range(len(Fclass)):
            print(len(Fclass[n]))
        np.save('./database/Sclass.npy',Sclass)
        np.save('./database/Fclass.npy',Fclass)


        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    data=Softdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/softdatabas.vtk') 
    data=Fdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/Fdatabase.vtk') 

    m2=initialgridAIY(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Sclass,Softzuobiaolist,N,hardlist)
    data=m2.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial2.vtk') 
    m=initialgridAIClassY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Fdatabase,Fzuobiaolist,Fclass,N,Fenjilist,hardlist)
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

def GosimAIClassY(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist):
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    m,m2,Tilist,Tizuobiaolist=sectionloadandextendFenji(m,patternSizex,patternSizey,0,1)
    #m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    #m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m2,patternSizex,patternSizey,0,1)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIClassY(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist)
    
    print('initial done')

    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial3.vtk') 
    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)

        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=Recodesectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)


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
        m= patchmatchmultiTiB(m,Tilist,size,itr,1)
        CTI=[]#检测使用率剖面
        #m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        #m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,8,0)#并行进程的数目
        #计算量加大过多
        #m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path="./output/reconstruction.npy"
        data=m.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial2.vtk') 
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









'''
##################################################直方图匹配版###########################################


def zhifangtutiqu(M,valist):#M为待测模版，valist为整个模型中涉及的值
    zhifangtu=[]
    for n in range(len(valist)):
        zhifangtu.append(np.sum(M == valist[n]))    
    return np.array(zhifangtu).astype(np.int)

def jisuan(Az,Bz):#Az和Bz分别为三维模版A和B的直方图
    a=0
    for n in range(len(Az)):
        a=a+abs(Az[n]-Bz[n])
    return a

def ZFTcount(A,B,valist):#输入AB计算两者直方图相似性得分
    return jisuan(zhifangtutiqu(A,valist),zhifangtutiqu(B,valist))

def zhifangtumatch(Alist,Blist,valist,N):#A为邻近领域
    s=np.zeros(len(valist))
    for n in range(len(Alist)):
        print(zhifangtutiqu(Alist[n],valist))
        s=s+zhifangtutiqu(Alist[n],valist)
    S=s/len(Alist)
    sc=[]
    #print(S.shape,len(Alist),len(Blist))
    for n in range(len(Blist)):
        sc.append(jisuan(S,zhifangtutiqu(Blist[n],valist)))
    D=getListMinNumIndex(sc,N)
    r=random.randint(0,len(D)-1)
    t=D[r]    
    return Blist[t]
    #返还最合适的
def zhifangtumatchonly(A,Blist,valist,N):#A为周围
    
    sc=[]
    for n in range(len(Blist)):
        sc.append(jisuan(A,Blist[n]))
    sc=getListMinNumIndex(d,N)
    r=random.randint(0,len(sc)-1)
    t=sc[r]    
    return Blist[t]
    #返还最合适的
    
def Alistget(Exm,h0,x0,y0,patternsize): #提取待模拟区域周围已经赋值的模版,h0,x0,y0为待模拟点的坐标,Exm2为扩展后的模拟网格
    ll=patternsize//2+1
    Alist=[]
    for x in range(-ll,ll+1,1):
        for y in range(-ll,ll+1,1):
            h1=h0
            x1=x0+x
            y1=y0+y
            tem=template1(Exm,patternsize,patternsize,patternsize,h1,x1,y1)
            if temdetect(tem):#不包含待模拟点
                Alist.append(tem)
    return Alist
    
def patternsearchClassNHY(o1,Exm2,Fclass,Fzuobiaolist,dark,lag,N,Alist,valist):
    template_h=o1.shape[0]
    template_x=o1.shape[1]
    template_y=o1.shape[2]
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    drill1=o1.reshape(ss,1)
    c1=99999
    cc=0
    for n in range(len(Fclass[dark])):#选取距离最近的类别
        rrr=random.randint(0,len(Fclass[dark][n])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
            c1=fun
            cc=n
    d=[]
    p=[]
    for n in range(min(len(Fclass[dark][cc]),1000)):
        rrr=random.randint(0,len(Fclass[dark][cc])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
        p.append(rrr)
    
    #N选1
    
    
    
    
    
    
    si=getListMinNumIndex(d,N)
    Blist=[]
    for n in range(len(si)):
        t=si[n]
        t=p[t]
        h0=Fzuobiaolist[dark][Fclass[dark][cc][t]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][t]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][t]][2]+lag

        resulttem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        
        Blist.append(resulttem)
    resulttem=zhifangtumatch(Alist,Blist,valist,1)
    
    return resulttem

def initialroadlistAIY(m,template_h,template_x,template_y,lag):#改进版
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
    count=0
    while n<len(lujing):
        flag=0
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
                flag=1
            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前 
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,0:lag]):
                #左
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                k=k+1
                flag=1
           
            if (flag!=0) and (k!=0) and (count<=40):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
            elif (flag!=0) and (k>=2) and (count>40):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
                
            else:
                lujing.append(lujing[n])
        #print(len(Roadlist),len(lujing)-n)
        n=n+1
        #print len(Roadlist)
    #print('roadlist initial done')
    return Roadlist


def initialgridAIClassHY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm1,Exm2,Fzuobiaolist,Fclass,N,Fenjilist,hardlist,valist):
    #优化版处理分级数据
    #valist为该模型中所有值类型列表
    #此处m2为软数据
    #添加了直方图匹配
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIY(m,template_h,template_x,template_y,lag)
    #lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=Exm2#拓展
    ms2=Exm1#软数据拓展
    
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
  
    
  
    for n in range(len(lujing)):
        print(n,len(lujing))
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
        Alist=Alistget(ms,h1,x1,y1,template_h)#获取待模拟区域周遭模版
        k=0#重叠区计数器
        flag=0
        tem=patternsearchClassNHY(o1,ms,Fclass,Fzuobiaolist,dark,lag,N,Alist,valist)
            
            
       

        #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
        m2=template1R(ms,tem,h1,x1,y1)   
        
    m=cut(ms,lag)
    return m

    
def gosiminitialAIClassHY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,valist):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
    #m为已经导入了Ti的模拟网格
    #优化版
    hardlist=[]
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

 
    my_file = Path("./database/Sclass.npy")
    if my_file.exists():
        Softdatabase=np.load('./database/Sdatabase.npy')
        Softzuobiaolist=np.load('./database/Softzuobiaolist.npy')
        Fdatabase=np.load('./database/Fdatabase.npy')
        Fzuobiaolist=np.load('./database/Fzuobiaolist.npy')
        Sclass=np.load('./database/Sclass.npy')
        Fclass=np.load('./database/Fclass.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Softdatabase,Softzuobiaolist=databasebuildAIY(m2,template_h,template_x,template_y,lag)#软数据数据库构建
        np.save('./database/Sdatabase.npy',Softdatabase)#softdata为扩充后的网格
        np.save('./database/Softzuobiaolist.npy',Softzuobiaolist)
        Fdatabase,Fzuobiaolist=databasebuildAIClassY(m,m2,template_h,template_x,template_y,lag,Fenjilist)
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        
        #以及分级数据库构建
        Sclass,Fclass=databaseclusterAIY(Softdatabase,Fdatabase,Softzuobiaolist,Fzuobiaolist,template_h,template_x,template_y,lag,U)
        
        print(len(Sclass))
        for n in range(len(Fclass)):
            print(len(Fclass[n]))
        np.save('./database/Sclass.npy',Sclass)
        np.save('./database/Fclass.npy',Fclass)


        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    data=Softdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/softdatabas.vtk') 
    data=Fdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/Fdatabase.vtk') 

    m2=initialgridAIY(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Sclass,Softzuobiaolist,N,hardlist)
    data=m2.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial2soft.vtk') 
    
    
    
    
    
    m=initialgridAIClassHY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Fdatabase,Fzuobiaolist,Fclass,N,Fenjilist,hardlist,valist)
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


def GosimAIClassHY(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,valist):
    #带直方图匹配版本
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    m,m2,Tilist,Tizuobiaolist=sectionloadandextendFenji(m,patternSizex,patternSizey,0,1)
    #m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    #m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m2,patternSizex,patternSizey,0,1)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIClassHY(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,valist)
    
    print('initial done')

    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial3.vtk') 
    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)

        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=Recodesectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)


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
        m= patchmatchmultiTiB(m,Tilist,size,itr,1)
        CTI=[]#检测使用率剖面
        #m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        #m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,8,0)#并行进程的数目
        #计算量加大过多
        #m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path="./output/reconstruction.npy"
        data=m.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial2.vtk') 
        np.save(path,m)
        time_end=time.time()
        #size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end-time_start)
        print('对比1：',TIvsM(mm,m,patternSizeh,lag,1000))
        print('对比2：',TIvsM2(mm,m,patternSizeh,lag,1000))
 
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
    
    
    
    return m
    
def TIvsM(Exm,m,patternsize,lag,count):#对训练图像以及结果进行直方图对比 ，count为对比个数,随机选取
    zb=[]
    pa=[]
    for h in range(m.shape[0]):    
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                tem=template1(Exm,patternsize,patternsize,patternsize,h+lag,x+lag,y+lag)
                if temdetect(tem):#不包含待模拟点
                    pa.append(tem)
                    zb.append([h,x,y])
    zbM=[]
    paM=[]
    for h in range(lag,m.shape[0]-lag):    
        for x in range(lag,m.shape[1]-lag):
            for y in range(lag,m.shape[2]-lag):
                tem=template1(m,patternsize,patternsize,patternsize,h+lag,x+lag,y+lag)
                if temdetect(tem):#不包含待模拟点
                    paM.append(tem)
                    zbM.append([h,x,y])  
    malajisi=0#计量列表                
    for n in range(count):
        r1=random.randint(0,len(zb)-1)
        r2=random.randint(0,len(zbM)-1)
        A=pa[r1]
        B=paM[r2]
        malajisi=malajisi+ZFTcount(A,B,valist)
        
    return float(malajisi)/count


def TIvsM2(Exm,m,patternsize,lag,count):#对训练图像以及结果进行直方图对比 ，count为对比个数,等高度随机选取
    zb=[]
    pa=[]
    for h in range(m.shape[0]):    
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                tem=template1(Exm,patternsize,patternsize,patternsize,h+lag,x+lag,y+lag)
                if temdetect(tem):#不包含待模拟点
                    pa.append(tem)
                    zb.append([h,x,y])


    malajisi=0#计量列表                
    for n in range(count):
        r1=random.randint(0,len(zb)-1)
        x0=random.randint(lag,m.shape[1]-lag)
        y0=random.randint(lag,m.shape[2]-lag)
        B=template1(Exm,patternsize,patternsize,patternsize,zb[r1][0],x0,y0)
        A=pa[r1]

        malajisi=malajisi+ZFTcount(A,B,valist)
    return float(malajisi)/count



'''


##############################################################ANOTHERVERSION#########################################################################
import scipy.stats
'''
def thesame(m1,m2):
    for h in range(m1.shape[0]):
        for x in range(m1.shape[1]):
            for y in range(m1.shape[2]):
                if m1[h,x,y]!=m2[h,x,y]:
                    return False
    return True #如果两个pattern完全一致则返回真
'''
def thesame(m1,m2):
   return (m1==m2).all()            
'''
def zhifangtu(Exm,lag,template_h,template_x,template_y,listpattern):
    listpattern=[]
    listvalue=[]
    for h in range(lag,Exm.shape[0]-lag):
        for x in range(lag,Exm.shape[1]-lag):
            for y in range(lag,Exm.shape[2]-lag):
                m1=template1(Exm,template_h,template_x,template_y,h,x,y)
                for n in range(len(listpattern)):
                    m2=template1(Exm,template_h,template_x,template_y,listpattern[n][0],listpattern[n][1],listpattern[n][2])
                    if thesame(m1,m2):
                        listvalue[n]=listvalue[n]+1
                        break
                    else:
                        listpattern.append([h,x,y])
                        
    return listvalue,listpattern
'''    


def zhifangtupattern(Exm,lag,template_h,template_x,template_y):#仅仅提取pattern并且构建列表
    listpattern=[]
    listvalue=[]
    listpattern.append([lag,lag,lag])
    for h in range(lag,Exm.shape[0]-lag):
        for x in range(lag,Exm.shape[1]-lag):
            for y in range(lag,Exm.shape[2]-lag):
                flag=0
                m1=template1(Exm,template_h,template_x,template_y,h,x,y)
                for n in range(len(listpattern)):
                    #print(n)
                    m2=template1(Exm,template_h,template_x,template_y,listpattern[n][0],listpattern[n][1],listpattern[n][2])
                    if thesame(m1,m2):
                        flag=0
                        break
                    else:
                        flag=1
                if flag==1: 
                    listpattern.append([h,x,y])
    listvalue=np.zeros(len(listpattern),int) 
    #print(len(listpattern))
    return listvalue,listpattern

def zhifangtuvaluelist(Exm,lag,template_h,template_x,template_y,listpattern,listvalue):#仅仅提取pattern并且构建列表

    for h in range(lag,Exm.shape[0]-lag):
        for x in range(lag,Exm.shape[1]-lag):
            for y in range(lag,Exm.shape[2]-lag):
                m1=template1(Exm,template_h,template_x,template_y,h,x,y)
                for n in range(len(listpattern)):
                    m2=template1(Exm,template_h,template_x,template_y,listpattern[n][0],listpattern[n][1],listpattern[n][2])
                    if thesame(m1,m2):
                        listvalue[n]=listvalue[n]+1
                        break
    #print(listvalue)
    return listvalue



def compareJS(lista,listb):
    lista=np.asarray(lista)
    listb=np.asarray(listb)
    M= (lista+listb)/2
    return 0.5*scipy.stats.entropy(lista, M)+0.5*scipy.stats.entropy(listb, M)
    return v
    
def TIandModelcompare(Exm,m,lag,template_h,template_x,template_y):
    listpattern=[]
    Exm2=np.pad(m,lag,'edge')
    listvalue,listpattern=zhifangtupattern(Exm,lag,template_h,template_x,template_y)
    listvalue2=listvalue.copy()
    listvalue= zhifangtuvaluelist(Exm,lag,template_h,template_x,template_y,listpattern,listvalue)
    listvalue2= zhifangtuvaluelist(Exm2,lag,template_h,template_x,template_y,listpattern,listvalue2)
    #print(listvalue.type)

    return compareJS(listvalue,listvalue2)





##################################################直方图匹配版2#####################################################################################
def cal_distanceforz(a, b, A_padding, B, p_size,valuelist):
    #print(b)
    p = p_size // 2
    patch_a = A_padding[a[0]-p:a[0]+p+1, a[1]-p:a[1]+p+1, a[2]-p:a[2]+p+1]
    #print(a[0]-p,a[0]+p+1,patch_a.shape[0])
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, b[2]-p:b[2]+p+1]
    #print(patch_a.shape[2])

    lista=zhifangtutiqu(patch_a,valist)
    listb=zhifangtutiqu(patch_b,valist)
    
    
    dist=compareJS(lista,listb)
    return dist

def initializationforz(A, B, p_size,valuelist):#初始化，构建映射
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    p = p_size // 2
    B_padding = np.ones([B_h+p*2, B_l+p*2,B_w+p*2]) * np.nan
    B_padding[p:B_h+p, p:B_l+p,p:B_w+p] = B
    B=B_padding
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    random_B_r = np.random.randint(p, B_h-p, [A_h, A_l,A_w])#分别储存对应B的坐标
    #print random_B_r
    random_B_c = np.random.randint(p, B_l-p, [A_h, A_l,A_w])
    random_B_v = np.random.randint(p, B_w-p, [A_h, A_l,A_w])
    #A_padding = np.ones([A_h+p*2, A_l+p*2, 1]) * np.nan
    #A_padding[p:A_h+p, p:A_l+p, :] = A
    A_padding = np.ones([A_h+p*2, A_l+p*2,A_w+p*2]) * np.nan
    A_padding[p:A_h+p, p:A_l+p,p:A_w+p] = A

    f = np.zeros([A_h, A_l,A_w], dtype=object)
    dist = np.zeros([A_h, A_l,A_w],float)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                a = np.array([i+p, j+p,k+p])
                b = np.array([random_B_r[i, j,k], random_B_c[i, j,k],random_B_v[i, j,k]], dtype=np.int32)
                #print b
                f[i, j,k] = b
                dist[i, j,k] = cal_distanceforz(a, b, A_padding, B, p_size,valuelist)
    #print f
    return f, dist, A_padding,B
def propagationforz(f, a, dist, A_padding, B, p_size, is_odd,valuelist):#传播
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
        idx = np.argmin(np.array([d_current, d_left, d_up,d_forward]))# 可以多加几个传播值
        if idx == 1:
            f[x, y,z] = f[max(x - 1, 0), y,z]
            dist[x, y,z] = cal_distanceforz(a, f[x, y,z], A_padding, B, p_size,valuelist)
        if idx == 2:
            f[x, y,z] = f[x, max(y - 1, 0),z]
            dist[x, y,z] = cal_distanceforz(a, f[x, y,z], A_padding, B, p_size,valuelist)
        if idx == 3:
            f[x, y,z] = f[x, y,max(z - 1, 0)]
            dist[x, y,z] = cal_distanceforz(a, f[x, y,z], A_padding, B, p_size,valuelist)
    else:
        d_right = dist[min(x + 1, A_h-1), y,z]
        d_down = dist[x, min(y + 1, A_l-1),z]
        d_current = dist[x, y,z]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y,z] = f[min(x + 1, A_h-1), y,z]
            dist[x, y,z] = cal_distanceforz(a, f[x, y,z], A_padding, B, p_size,valuelist)
        if idx == 2:
            f[x, y,z] = f[x, min(y + 1, A_l-1),z]
            dist[x, y,z] = cal_distanceforz(a, f[x, y,z], A_padding, B, p_size,valuelist)
        if idx == 3:
            f[x, y,z] = f[x,y, min(z + 1, A_w-1)]
            dist[x, y,z] = cal_distanceforz(a, f[x, y,z], A_padding, B, p_size,valuellist)




def NNSBforz(img, ref, p_size, itr,name,idcard,valuelist):#寻找最近零并行版
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding ,Bref= initializationforz(img, ref, p_size,valuelist)
    p=p_size//2
    print("initialization done")
    for itr in range(1, itr+1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_l - 1, -1, -1):
                    for k in range(A_w - 1, -1, -1):
                        a = np.array([i+p, j+p,k+p])
                        propagationforz(f, a, dist, img_padding, Bref, p_size, False,valuelist)
                        random_search(f, a, dist, img_padding, Bref, p_size)
        else:
            for i in range(A_h):
                for j in range(A_l):
                    for k in range(A_w):
                        a = np.array([i+p, j+p,k+p])
                        propagationforz(f, a, dist, img_padding, Bref, p_size, True,valuelist)
                        random_search(f, a, dist, img_padding, Bref, p_size)
        print("iteration: %d"%(itr))
    path1='./database/patchmatch('+str(name)+')('+str(idcard)+')f.npy'
    path2='./database/patchmatch('+str(name)+')('+str(idcard)+')dist.npy'
    path3='./database/patchmatch('+str(name)+')('+str(idcard)+')Bref.npy'
    np.save(path1,f)
    np.save(path2,dist)
    np.save(path3,Bref)
    return f,dist,Bref

def patchmatchmultiTiBforz(m,Tilist,size,itr,name,valuelist):#patchmatch做优化 并行版
    #size为模板大小，itr为迭代次数
    Fore = np.zeros([m.shape[0], m.shape[1],m.shape[2]])
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    F=[]
    DIST=[]
    BTilist=[]
    processes=list()
    for n in range(len(Tilist)):
        print('porcess:')
        print(name,n)
        s=multiprocessing.Process(target=NNSBforz, args=(m,Tilist[n],size,itr,name,n,valuelist))
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    for n in range(len(Tilist)):
        path1='./database/patchmatch('+str(name)+')('+str(n)+')f.npy'
        path2='./database/patchmatch('+str(name)+')('+str(n)+')dist.npy'
        path3='./database/patchmatch('+str(name)+')('+str(n)+')Bref.npy'
        f=np.load(path1)
        dist=np.load(path2)
        Bref=np.load(path3)
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)
    '''
    for n in range(len(Tilist)):
        p=multiprocessing.Process(target=NNS, args=(m,Tilist[n],size,itr))
        print('process start!')
    '''
    print("Searching done!")
    #print F[0],F[1],F[2],F[3]
    Fore=projectTTT(m,F,DIST)
    #print Fore
    print("Pick done!")
    Re=reconstructionTTT(m,F,Fore,BTilist)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re

def zhifangtutiqu(M,valist):#M为待测模版，valist为整个模型中涉及的值
    zhifangtu=[]
    for n in range(len(valist)):
        zhifangtu.append(np.sum(M == valist[n]))    
    return np.array(zhifangtu).astype(np.int)

def jisuan(Az,Bz):#Az和Bz分别为三维模版A和B的直方图
    a=0
    for n in range(len(Az)):
        a=a+abs(Az[n]-Bz[n])
    return a

def ZFTcount(A,B,valist):#输入AB计算两者直方图相似性得分
    return jisuan(zhifangtutiqu(A,valist),zhifangtutiqu(B,valist))

def zhifangtumatch(Alist,Blist,valist,N):#A为邻近领域
    s=np.zeros(len(valist))
    for n in range(len(Alist)):
        print(valist)
        print(zhifangtutiqu(Alist[n],valist))
        s=s+zhifangtutiqu(Alist[n],valist)
    S=s/len(Alist)
    sc=[]
    #print(S.shape,len(Alist),len(Blist))
    for n in range(len(Blist)):
        sc.append(compareJS(S,zhifangtutiqu(Blist[n],valist)))
    D=getListMinNumIndex(sc,N)
    r=random.randint(0,len(D)-1)
    t=D[r]    
    return Blist[t]
    #返还最合适的
def zhifangtumatchonly(A,Blist,valist,N):#A为周围
    
    sc=[]
    for n in range(len(Blist)):
        sc.append(jisuan(A,Blist[n]))
    sc=getListMinNumIndex(d,N)
    r=random.randint(0,len(sc)-1)
    t=sc[r]    
    return Blist[t]
    #返还最合适的
    
def Alistget(Exm,h0,x0,y0,patternsize): #提取待模拟区域周围已经赋值的模版,h0,x0,y0为待模拟点的坐标,Exm2为扩展后的模拟网格
    ll=patternsize//2+2
    Alist=[]
    for h in range(-ll,ll+1,1):
        for x in range(-ll,ll+1,1):
            for y in range(-ll,ll+1,1):
                h1=h0+h
                x1=x0+x
                y1=y0+y
                tem=template1(Exm,patternsize,patternsize,patternsize,h1,x1,y1)

                if temdetect(tem) and tem.shape[0]==patternsize and tem.shape[1]==patternsize and tem.shape[2]==patternsize:#不包含待模拟点
                   Alist.append(tem)
    return Alist



def patternsearchNHY(tem,Exm,Sclass,zuobiaolist,lag,N,Alist,valuelist):
    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    c1=99999
    cc=0
    
    for n in range(len(Sclass)):#选取距离最近的类别
        #print(len(Sclass[n]))
        rrr=random.randint(0,len(Sclass[n])-1)
        h0=zuobiaolist[Sclass[n][rrr]][0]+lag
        x0=zuobiaolist[Sclass[n][rrr]][1]+lag
        y0=zuobiaolist[Sclass[n][rrr]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
            c1=fun
            cc=n
    d=[]
    p=[]
    '''
    for n in range(len(Sclass[cc])):

        h0=zuobiaolist[Sclass[cc][n]][0]+lag
        x0=zuobiaolist[Sclass[cc][n]][1]+lag
        y0=zuobiaolist[Sclass[cc][n]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    '''
    for n in range(min(len(Sclass[cc]),1000)):
        rrr=random.randint(0,len(Sclass[cc])-1)
        h0=zuobiaolist[Sclass[cc][rrr]][0]+lag
        x0=zuobiaolist[Sclass[cc][rrr]][1]+lag
        y0=zuobiaolist[Sclass[cc][rrr]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
        p.append(rrr)
    #N选1
    si=getListMinNumIndex(d,N)
    #print(len(si),si)
    r=random.randint(0,len(si)-1)
    assassin=si[r]
    t=p[assassin]
    h0=zuobiaolist[Sclass[cc][t]][0]+lag
    x0=zuobiaolist[Sclass[cc][t]][1]+lag
    y0=zuobiaolist[Sclass[cc][t]][2]+lag
    resulttem=template1(Exm,template_h,template_x,template_y,h0,x0,y0)

    Blist=[]
    for n in range(len(si)):
        t=si[n]
        t=p[t]
        h0=zuobiaolist[Sclass[cc][t]][0]+lag
        x0=zuobiaolist[Sclass[cc][t]][1]+lag
        y0=zuobiaolist[Sclass[cc][t]][2]+lag

        resulttem=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        
        Blist.append(resulttem)
    resulttem=zhifangtumatch(Alist,Blist,valuelist,1)
    return resulttem

def initialgridAIHY(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm,Sclass,zuobiaolist,N,hardlist,valuelist):
    #优化版，Exm为拓展数据库
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAI(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    m2=Exm

    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]

        
        
    #############################################
    sqflag=0
    while sqflag==0:
        sqflag=0
        for n in range(len(lujing)):
            #print(n,len(lujing))
            #print(valuelist)
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            flag=0
            Alist=Alistget(m2,h1,x1,y1,template_h)#获取待模拟区域周遭模版
            tem=patternsearchNHY(o1,Exm,Sclass,zuobiaolist,lag,N,Alist,valuelist)
            
            
          

            #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
            m2=template1R(m2,tem,h1,x1,y1)
        sqflag=1
        '''
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

        lujing=[]
        disss=[]    
        ms,disss=checkunreal2(m2,lag)
        if len(disss)==0:
            sqflag=1
        else:
            
            lujing=subroadlistinitialfornew(m2,disss,template_h,template_x,template_y,lag)
        '''
            
        
    m=cut(m2,lag)
    return m


    
def patternsearchClassNHY(o1,Exm2,Fclass,Fzuobiaolist,dark,lag,N,Alist,valist):
    template_h=o1.shape[0]
    template_x=o1.shape[1]
    template_y=o1.shape[2]
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    drill1=o1.reshape(ss,1)
    c1=99999
    cc=0
    for n in range(len(Fclass[dark])):#选取距离最近的类别
        rrr=random.randint(0,len(Fclass[dark][n])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        #print(tem2.shape[0],tem2.shape[1],tem2.shape[2])
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
            c1=fun
            cc=n
    d=[]
    p=[]
    for n in range(min(len(Fclass[dark][cc]),10000)):
        rrr=random.randint(0,len(Fclass[dark][cc])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
        p.append(rrr)
    
    #N选1
    
    
    
    
    
    
    si=getListMinNumIndex(d,N)
    Blist=[]
    for n in range(len(si)):
        t=si[n]
        t=p[t]
        h0=Fzuobiaolist[dark][Fclass[dark][cc][t]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][t]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][t]][2]+lag

        resulttem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        
        Blist.append(resulttem)
    resulttem=zhifangtumatch(Alist,Blist,valist,1)
    
    return resulttem

def initialroadlistAIY(m,template_h,template_x,template_y,lag):#改进版
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
    count=0
    while n<len(lujing):
        flag=0
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
                flag=1
            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前 
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,0:lag]):
                #左
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                k=k+1
                flag=1


            Alist=Alistget(m2,h1,x1,y1,template_h)
            if (flag!=0) and (k!=0) and (count<=40) and len(Alist)>0:
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
            elif (flag!=0) and (k>=2) and (count>40) and len(Alist)>0:
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
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

        if len(lujing)-n<=40:
            Roadlist.append((h1-lag,x1-lag,y1-lag))
            n=n+1
        #print len(Roadlist)
    print('roadlist initial done')
    return Roadlist




def initialgridAIClassHY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm1,Exm2,Fzuobiaolist,Fclass,N,Fenjilist,hardlist,valist):
    #优化版处理分级数据
    #valist为该模型中所有值类型列表
    #此处m2为软数据
    #添加了直方图匹配
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIY(m,template_h,template_x,template_y,lag)
    #lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=Exm2#拓展
    ms2=Exm1#软数据拓展
    
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
  
    
  
    for n in range(len(lujing)):
        print(n,len(lujing))
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
        Alist=Alistget(ms,h1,x1,y1,template_h)#获取待模拟区域周遭模版
        k=0#重叠区计数器
        flag=0
        tem=patternsearchClassNHY(o1,ms,Fclass,Fzuobiaolist,dark,lag,N,Alist,valist)
            
            
       

        #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
        m2=template1R(ms,tem,h1,x1,y1)   
        
    m=cut(ms,lag)
    return m

    
def gosiminitialAIClassHY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,valist,softTilist):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
    #m为已经导入了Ti的模拟网格
    #优化版
    hardlist=[]
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

 
    my_file = Path("./database/Sclass.npy")
    if my_file.exists():
        Softdatabase=np.load('./database/Sdatabase.npy')
        Softzuobiaolist=np.load('./database/Softzuobiaolist.npy')
        Fdatabase=np.load('./database/Fdatabase.npy')
        Fzuobiaolist=np.load('./database/Fzuobiaolist.npy')
        Sclass=np.load('./database/Sclass.npy')
        Fclass=np.load('./database/Fclass.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Softdatabase,Softzuobiaolist=databasebuildAIY(m2,template_h,template_x,template_y,lag)#软数据数据库构建
        np.save('./database/Sdatabase.npy',Softdatabase)#softdata为扩充后的网格
        np.save('./database/Softzuobiaolist.npy',Softzuobiaolist)
        Fdatabase,Fzuobiaolist=databasebuildAIClassY(m,m2,template_h,template_x,template_y,lag,Fenjilist)
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        
        #以及分级数据库构建
        Sclass,Fclass=databaseclusterAIY(Softdatabase,Fdatabase,Softzuobiaolist,Fzuobiaolist,template_h,template_x,template_y,lag,U)
        
        print(len(Sclass))
        for n in range(len(Fclass)):
            print(len(Fclass[n]))
        np.save('./database/Sclass.npy',Sclass)
        np.save('./database/Fclass.npy',Fclass)


        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    data=Softdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/softdatabas.vtk') 
    data=Fdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/Fdatabase.vtk') 

    m2=initialgridAIHY(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Sclass,Softzuobiaolist,N,hardlist,Fenjilist)
    ###################################

    mm=-np.ones((int(m2.shape[0]),int(m2.shape[1]),int(m2.shape[2])),int)

    Tilist=[]
    Tizuobiaolist=[]


    m2= patchmatchmultiTiB(m2,softTilist,size,itr,1)
    print("softdata em done")
       

    ################################

    data=m2.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial2soft.vtk') 
    
    print('softdatadone')
    
    
    
    m=initialgridAIClassHY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Fdatabase,Fzuobiaolist,Fclass,N,Fenjilist,hardlist,valist)
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


def GosimAIClassHY(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,valist):
    #带直方图匹配版本
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    #m,m2,Tilist,Tizuobiaolist=sectionloadandextendFenji(m,patternSizex,patternSizey,0,1)
    m,m2,Tilist,Tizuobiaolist,softTilist,softTizuobiaolist=sectionloadandextendFenjiAFA(m,patternSizex,patternSizey,1,1)
    #m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    #m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m2,patternSizex,patternSizey,0,1)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIClassHY(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,valist,softTilist)
    
    print('initial done')

    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial3.vtk') 
    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)

        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=Recodesectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)


        mm=extendTimodel(mm,patternSizeh,patternSizex,patternSizey)
        np.save('./output/Ti.npy',mm)
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
        m= patchmatchmultiTiB(m,Tilist,size,itr,1)
        #m= patchmatchmultiTiBforz(m,Tilist,size,itr,1,valist)
        CTI=[]#检测使用率剖面
        #m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        #m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,8,0)#并行进程的数目
        #计算量加大过多
        #m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path="./output/reconstruction.npy"
        data=m.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial2.vtk') 
        np.save(path,m)
        time_end=time.time()
        #size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end-time_start)
        #print('对比1：',TIvsM(mm,m,patternSizeh,lag,1000))
        #print('对比2：',TIvsM2(mm,m,patternSizeh,lag,1000))
        Exm=np.pad(mm,lag,'edge')
        #print('对比3，JS散度',TIandModelcompare(Exm,m,lag,patternSizeh,patternSizex,patternSizey))
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
    
    
    
    return m
    
def TIvsM(Exm,m,patternsize,lag,count):#对训练图像以及结果进行直方图对比 ，count为对比个数,随机选取
    zb=[]
    pa=[]
    for h in range(m.shape[0]):    
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                tem=template1(Exm,patternsize,patternsize,patternsize,h+lag,x+lag,y+lag)
                if temdetect(tem):#不包含待模拟点
                    pa.append(tem)
                    zb.append([h,x,y])
    zbM=[]
    paM=[]
    for h in range(lag,m.shape[0]-lag):    
        for x in range(lag,m.shape[1]-lag):
            for y in range(lag,m.shape[2]-lag):
                tem=template1(m,patternsize,patternsize,patternsize,h+lag,x+lag,y+lag)
                if temdetect(tem):#不包含待模拟点
                    paM.append(tem)
                    zbM.append([h,x,y])  
    malajisi=0#计量列表                
    for n in range(count):
        r1=random.randint(0,len(zb)-1)
        r2=random.randint(0,len(zbM)-1)
        A=pa[r1]
        B=paM[r2]
        malajisi=malajisi+ZFTcount(A,B,valist)
        
    return float(malajisi)/count


def TIvsM2(Exm,m,patternsize,lag,count):#对训练图像以及结果进行直方图对比 ，count为对比个数,等高度随机选取
    zb=[]
    pa=[]
    for h in range(m.shape[0]):    
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                tem=template1(Exm,patternsize,patternsize,patternsize,h+lag,x+lag,y+lag)
                if temdetect(tem):#不包含待模拟点
                    pa.append(tem)
                    zb.append([h,x,y])


    malajisi=0#计量列表                
    for n in range(count):
        r1=random.randint(0,len(zb)-1)
        x0=random.randint(lag,m.shape[1]-lag)
        y0=random.randint(lag,m.shape[2]-lag)
        B=template1(Exm,patternsize,patternsize,patternsize,zb[r1][0],x0,y0)
        A=pa[r1]

        malajisi=malajisi+ZFTcount(A,B,valist)
    return float(malajisi)/count




################################################中心点加直方图匹配版本###########################################################################

def Alistgetforcenter(Exm,h0,x0,y0,patternsize): #提取待模拟区域周围已经赋值的模版,h0,x0,y0为待模拟点的坐标,Exm2为扩展后的模拟网格
    ll=patternsize
    Alist=[]
    for x in range(-ll,ll+1,1):
        for y in range(-ll,ll+1,1):
            h1=h0
            x1=x0+x
            y1=y0+y
            
            tem=template1(Exm,patternsize,patternsize,patternsize,h1,x1,y1)
            #print(tem)
            if temdetect(tem) and tem.shape[0]==patternsize and tem.shape[1]==patternsize and tem.shape[2]==patternsize :#不包含待模拟点
               #print(tem,tem.shape[0],tem.shape[1],tem.shape[2])
               #print(tem[0,0,0])
               #print(h1,x1,y1)
               Alist.append(tem[patternsize//2,patternsize//2,patternsize//2])
    return np.array(Alist).astype(np.int)

def valuelisttiqu(M):#M为待测模版
    valuelist=[]
    for n in range(M.shape[0]):
        for x in range(M.shape[1]):
            for y in range(M.shape[2]):
                if M[n,x,y] not in valuelist:
                    valuelist.append(M[n,x,y])
    return valuelist



def zhongxinmatch(Alist,Blist,N):#返还中心匹配的一系列模板 alist为邻近周边中心点列表，Blist为候选模板列表
    juli=[]
    counts = np.bincount(Alist)
    #返回众数 返回最大值在数列中的索引位置
    np.argmax(counts)
    lag=Blist[n].shape[0]//2
    for n in range(len(Blist)):
        juli1.append(abs(counts-Blist[n][lag,lag,lag]))
    if N>=len(Blist):
        si=getListMinNumIndex(d,N)
    else:
        si=getListMinNumIndex(d,len(Blist))

    
    Clist=[]
    for choose in range(len(si)):
        Clist.append(Blist[si[choose]])
    return Clist

def zhongxinmatch2(Alist,Blist,N):#返还中心匹配的一系列模板 alist为邻近周边中心点列表，Blist为候选模板列表中心点列表
    juli1=[]
    Alist=np.array(Alist)
    #print(Alist)
    counts = np.bincount(Alist)
    #返回众数 返回最大值在数列中的索引位置
    c=np.argmax(counts)

    
    #print(c)
    for n in range(len(Blist)):
        juli1.append(abs(c-Blist[n]))
    #print(juli1)
    if N>=len(Blist):
        si=getListMinNumIndex(juli1,N)
    else:
        si=getListMinNumIndex(juli1,len(Blist))

    
    
    return si

def patternsearchAFA(tem,Exm,Sclass,zuobiaolist,lag,N,Alist,valuelist):#中心点语义距离softdata版本
    template_h=tem.shape[0]
    template_x=tem.shape[1]
    template_y=tem.shape[2]
    ss=tem.shape[0]*tem.shape[1]*tem.shape[2]
    drill1=tem.reshape(ss,1)
    c1=99999
    cc=0
    
    for n in range(len(Sclass)):#选取距离最近的类别
        #print(len(Sclass[n]))
        rrr=random.randint(0,len(Sclass[n])-1)
        h0=zuobiaolist[Sclass[n][rrr]][0]+lag
        x0=zuobiaolist[Sclass[n][rrr]][1]+lag
        y0=zuobiaolist[Sclass[n][rrr]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)	
        if fun<=c1:#选最小的序号
            c1=fun
            cc=n
    d=[]
    p=[]
    '''
    for n in range(len(Sclass[cc])):

        h0=zuobiaolist[Sclass[cc][n]][0]+lag
        x0=zuobiaolist[Sclass[cc][n]][1]+lag
        y0=zuobiaolist[Sclass[cc][n]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
    '''
    Blist=[]
    for n in range(min(len(Sclass[cc]),1000)):
        rrr=random.randint(0,len(Sclass[cc])-1)
        h0=zuobiaolist[Sclass[cc][rrr]][0]+lag
        x0=zuobiaolist[Sclass[cc][rrr]][1]+lag
        y0=zuobiaolist[Sclass[cc][rrr]][2]+lag
        tem2=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        Blist.append(tem2[template_h//2,template_x//2,template_y//2])
        
                     
                     
                     
        #drill2=tem2.reshape(ss,1)
        #d.append(hamming_distance(drill1,drill2))
        #p.append(rrr)
        
    Clist=zhongxinmatch2(Alist,Blist,N)
    
    c1=99999
    cb=0
    for n in range(len(Clist)):
        h0=zuobiaolist[Sclass[cc][Clist[n]]][0]+lag
        x0=zuobiaolist[Sclass[cc][Clist[n]]][1]+lag
        y0=zuobiaolist[Sclass[cc][Clist[n]]][2]+lag
        resulttem=template1(Exm,template_h,template_x,template_y,h0,x0,y0)
        drill2=resulttem.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
            c1=fun
            cb=n
        
    h0=zuobiaolist[Sclass[cc][Clist[cb]]][0]+lag
    x0=zuobiaolist[Sclass[cc][Clist[cb]]][1]+lag
    y0=zuobiaolist[Sclass[cc][Clist[cb]]][2]+lag
    resulttem=template1(Exm,template_h,template_x,template_y,h0,x0,y0)    

    return resulttem

def initialroadlistAIAFA(m,template_h,template_x,template_y,lag):#改进版
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
    count=0
    while n<len(lujing):
        flag=0
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
                flag=1
            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前 
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,0:lag]):
                #左
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                k=k+1
                flag=1
            Alist=Alistgetforcenter(m2,h1,x1,y1,template_h)


            if (flag!=0) and (k!=0) and (count<=40) and len(Alist)!=0:
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
            elif (flag!=0) and (k>=2) and (count>40) and len(Alist)!=0:
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
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
    print('roadlist initial done')
    return Roadlist



def initialgridAIAFA(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm,Sclass,zuobiaolist,N,hardlist,valuelist):
    #优化版，Exm为拓展数据库
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIAFA(m,template_h,template_x,template_y,lag)
    print('initialize start')
    #print len(lujing)
    m2=Exm

    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]

        
        
    #############################################
    sqflag=0
    while sqflag==0:
        sqflag=0
        for n in range(len(lujing)):
            #print(n,len(lujing))
            #print(valuelist)
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            o1=template1(m2,template_h,template_x,template_y,h1,x1,y1)
            k=0#重叠区计数器
            flag=0
            #print(m2)
            #Alist=Alistget(m2,h1,x1,y1,template_h)#获取待模拟区域周遭模版
            Alist=Alistgetforcenter(m2,h1,x1,y1,template_h)#获取待模拟
            tem=patternsearchAFA(o1,Exm,Sclass,zuobiaolist,lag,N,Alist,valuelist)
        

            
          

            #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
            m2=template1R(m2,tem,h1,x1,y1)
        sqflag=1
        '''
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

        lujing=[]
        disss=[]    
        ms,disss=checkunreal2(m2,lag)
        if len(disss)==0:
            sqflag=1
        else:
            
            lujing=subroadlistinitialfornew(m2,disss,template_h,template_x,template_y,lag)
        '''
            
        
    m=m2[lag:m2.shape[0]-lag,lag:m2.shape[1]-lag,lag:m2.shape[2]-lag]
    return m


'''    
def patternsearchClassNHY(o1,Exm2,Fclass,Fzuobiaolist,dark,lag,N,Alist,valist):
    template_h=o1.shape[0]
    template_x=o1.shape[1]
    template_y=o1.shape[2]
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    drill1=o1.reshape(ss,1)
    c1=99999
    cc=0
    for n in range(len(Fclass[dark])):#选取距离最近的类别
        rrr=random.randint(0,len(Fclass[dark][n])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
            c1=fun
            cc=n
    d=[]
    p=[]
    for n in range(min(len(Fclass[dark][cc]),1000)):
        rrr=random.randint(0,len(Fclass[dark][cc])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
        p.append(rrr)
    
    #N选1
    
    
    
    
    
    
    si=getListMinNumIndex(d,N)
    Blist=[]
    for n in range(len(si)):
        t=si[n]
        t=p[t]
        h0=Fzuobiaolist[dark][Fclass[dark][cc][t]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][t]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][t]][2]+lag

        resulttem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        
        Blist.append(resulttem)
    resulttem=zhifangtumatch(Alist,Blist,valist,1)
    
    return resulttem

def initialroadlistAIY(m,template_h,template_x,template_y,lag):#改进版
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
    count=0
    while n<len(lujing):
        flag=0
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
                flag=1
            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前 
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,0:lag]):
                #左
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                k=k+1
                flag=1
           
            if (flag!=0) and (k!=0) and (count<=40):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
            elif (flag!=0) and (k>=2) and (count>40):
                m2=template1R(m2,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
                
                #for hb in range(h1-lag-lag,h1+1):
                #    for xb in range(x1-lag-lag,x1+1):
                #        for yb in range(y1-lag-lag,y1+1):
                #            Banlujing.append((hb,xb,yb))#将已经模拟的区域坐标加入黑名单
                
            else:
                lujing.append(lujing[n])
        #print(len(Roadlist),len(lujing)-n)
        n=n+1
        #print len(Roadlist)
    #print('roadlist initial done')
    return Roadlist




def initialgridAIClassHY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm1,Exm2,Fzuobiaolist,Fclass,N,Fenjilist,hardlist,valist):
    #优化版处理分级数据
    #valist为该模型中所有值类型列表
    #此处m2为软数据
    #添加了直方图匹配
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIY(m,template_h,template_x,template_y,lag)
    #lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=Exm2#拓展
    ms2=Exm1#软数据拓展
    
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
  
    
  
    for n in range(len(lujing)):
        print(n,len(lujing))
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
        Alist=Alistget(ms,h1,x1,y1,template_h)#获取待模拟区域周遭模版
        k=0#重叠区计数器
        flag=0
        tem=patternsearchClassNHY(o1,ms,Fclass,Fzuobiaolist,dark,lag,N,Alist,valist)
            
            
       

        #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
        m2=template1R(ms,tem,h1,x1,y1)   
        
    m=cut(ms,lag)
    return m
'''
    
def gosiminitialAIClassAFA(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,valist,softTilist,softTizuobiaolist):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
    #m为已经导入了Ti的模拟网格
    #优化版
    hardlist=[]
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

 
    my_file = Path("./database/Sclass.npy")
    if my_file.exists():
        Softdatabase=np.load('./database/Sdatabase.npy')
        Softzuobiaolist=np.load('./database/Softzuobiaolist.npy')
        Fdatabase=np.load('./database/Fdatabase.npy')
        Fzuobiaolist=np.load('./database/Fzuobiaolist.npy')
        Sclass=np.load('./database/Sclass.npy')
        Fclass=np.load('./database/Fclass.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Softdatabase,Softzuobiaolist=databasebuildAIY(m2,template_h,template_x,template_y,lag)#软数据数据库构建
        np.save('./database/Sdatabase.npy',Softdatabase)#softdata为扩充后的网格
        np.save('./database/Softzuobiaolist.npy',Softzuobiaolist)
        Fdatabase,Fzuobiaolist=databasebuildAIClassY(m,m2,template_h,template_x,template_y,lag,Fenjilist)
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        
        #以及分级数据库构建
        Sclass,Fclass=databaseclusterAIY(Softdatabase,Fdatabase,Softzuobiaolist,Fzuobiaolist,template_h,template_x,template_y,lag,U)
        
        print(len(Sclass))
        for n in range(len(Fclass)):
            print(len(Fclass[n]))
        np.save('./database/Sclass.npy',Sclass)
        np.save('./database/Fclass.npy',Fclass)


        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    data=Softdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/softdatabas.vtk') 
    data=Fdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/Fdatabase.vtk') 
    print(m.shape[0],m.shape[1],m.shape[2])
    print(Softdatabase.shape[0],Softdatabase.shape[1],Softdatabase.shape[2])
    mm=m2.copy()
    m2=initialgridAIAFA(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Sclass,Softzuobiaolist,N,hardlist,Fenjilist)

###################################

    mm=-np.ones((int(m2.shape[0]),int(m2.shape[1]),int(m2.shape[2])),int)

    Tilist=[]
    Tizuobiaolist=[]


    m2= patchmatchmultiTiB(m2,softTilist,size,itr,1)
    print("softdata em done")
       

################################


    data=m2.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial2soft.vtk') 
    
    print('softdatadone')
    
    

    m=initialgridAIClassHY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Fdatabase,Fzuobiaolist,Fclass,N,Fenjilist,hardlist,valist)
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

def sectionloadandextendFenjiAFA(m,template_x,template_y,flag,scale):#flag==1为patchmatch步骤，0为initial步骤
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
    Tizuobiaolist2=[]
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
        if flag==1:
            Ti,Tizuobiao=TIextendforEM(section,m2,template_x,template_y,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)
            Tilist2.append(Ti)
            Tizuobiaolist2.append(Tizuobiao)
    return m,m2,Tilist,Tizuobiaolist,Tilist2,Tizuobiaolist2

def sectionloadandextendFenjiAFAone(m,template_x,template_y,flag,scale):#flag==1为patchmatch步骤，0为initial步骤
    #对剖面进行导入和Ti提取的函数  #scale为当前倍率
    Tilist=[]
    Tizuobiaolist=[]
    file1=open('./Ti/Tiparametersoft.txt')
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
    Tizuobiaolist2=[]
    file2=open('./Ti/Tiparametersoft.txt')
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
        if flag==1:
            Ti,Tizuobiao=TIextendforEM(section,m2,template_x,template_y,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale)
            Tilist2.append(Ti)
            Tizuobiaolist2.append(Tizuobiao)
    return m,m2,Tilist,Tizuobiaolist,Tilist2,Tizuobiaolist2



def GosimAIClassAFA(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,valist):
    #带直方图匹配版本
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    m,m2,Tilist,Tizuobiaolist,softTilist,softTizuobiaolist=sectionloadandextendFenjiAFA(m,patternSizex,patternSizey,1,1)
    #m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    #m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m2,patternSizex,patternSizey,0,1)
    #m2为辅助分级模型
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIClassAFA(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,valist,softTilist,softTizuobiaolist)
    
    print('initial done')

    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial3.vtk') 
    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)

        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=Recodesectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)


        mm=extendTimodel(mm,patternSizeh,patternSizex,patternSizey)
        np.save('./output/Ti.npy',mm)
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
        #m= patchmatchmultiTiBforz(m,Tilist,size,itr,1,valist)
        CTI=[]#检测使用率剖面
        #m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,8,0)#并行进程的数目
        #计算量加大过多
        #m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path="./output/reconstruction.npy"
        data=m.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial2.vtk') 
        np.save(path,m)
        time_end=time.time()
        #size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end-time_start)
        #print('对比1：',TIvsM(mm,m,patternSizeh,lag,1000))
        #print('对比2：',TIvsM2(mm,m,patternSizeh,lag,1000))
        Exm=np.pad(mm,lag,'edge')
        #print('对比3，JS散度',TIandModelcompare(Exm,m,lag,patternSizeh,patternSizex,patternSizey))
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
    
    
    
    return m
    





####################################################pythia+hgosim################################################################################
def initialroadlistAIPY(m,ms,template_h,template_x,template_y,lag):#改进版
    #lag为重叠区
    #自动初始化网格系统
    lujing=[]
    Roadlist=[]#最终路径名单
    lujing=lujinglistAI2(m,template_h,template_x,template_y,lag-1)
    random.shuffle(lujing)
    #print len(lujing)
    
    #print len(lujing)
    m2=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(ms,lag,'edge')
    Fin=m.shape[0]*m.shape[1]*m.shape[2]
    Fin=Fin*1000#最大循环次数
    DevilTrigger=False
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
    Banlujing=[]
    b=np.zeros((template_h,template_x,template_y),int)
    n=0
    count=0
    while n<len(lujing):
        flag=0

        if m2[lujing[n][0]+lag,lujing[n][1]+lag,lujing[n][2]+lag]==-1:
            #if lujing[n] not in Banlujing:
            h1=lujing[n][0]+lag
            x1=lujing[n][1]+lag
            y1=lujing[n][2]+lag
            dark=ms2[h1,x1,y1]
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
                flag=1
            if temdetectD1(o1[:,template_x-lag:template_x,:]):
                #前 
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,0:lag]):
                #左
                k=k+1
                flag=1
            if temdetectD1(o1[:,:,template_y-lag:template_y]):
                #右
                k=k+1
                flag=1

            Alist=Alistget(m2,h1,x1,y1,template_h)
            '''
            if (flag!=0) and (k!=0) and (count<=40):
                #m2=template1R(m2,b,h1,x1,y1)
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
            elif (flag!=0) and (k>=2) and (count>40):
                #m2=template1R(m2,b,h1,x1,y1)
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
            '''
                
            Alist=Alistget(m2,h1,x1,y1,template_h)
            if (flag!=0) and (k!=0) and (count<=40) and len(Alist)>0:
                #m2=template1R(m2,b,h1,x1,y1)
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
            elif (flag!=0) and (k>=2) and (count>40) and len(Alist)>0:
                #m2=template1R(m2,b,h1,x1,y1)
                m2=template1RAIFenji(m2,ms2,dark,b,h1,x1,y1)
                Roadlist.append((h1-lag,x1-lag,y1-lag))
                count=count+1
          

               
            else:
                lujing.append(lujing[n])




         
            print(len(Roadlist),len(lujing)-n)
        n=n+1

        if len(lujing)-n<=40:
            Roadlist.append((h1-lag,x1-lag,y1-lag))
            n=n+1
        #print len(Roadlist)
    print('roadlist initial done')
    return Roadlist




def patternsearchClassNHY2(o1,Exm2,Fclass,Fzuobiaolist,dark,lag,N,valist):
    template_h=o1.shape[0]
    template_x=o1.shape[1]
    template_y=o1.shape[2]
    ss=o1.shape[0]*o1.shape[1]*o1.shape[2]
    drill1=o1.reshape(ss,1)
    c1=99999
    cc=0
    for n in range(len(Fclass[dark])):#选取距离最近的类别
        rrr=random.randint(0,len(Fclass[dark][n])-1)
        h0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][n][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        #print(tem2.shape[0],tem2.shape[1],tem2.shape[2])
        drill2=tem2.reshape(ss,1)
        fun=hamming_distance(drill1,drill2)
        if fun<=c1:#选最小的序号
            c1=fun
            cc=n
    d=[]
    #p=[]
    for n in range(min(len(Fclass[dark][cc]),10000)):
        #rrr=random.randint(0,len(Fclass[dark][cc])-1)
        rrr=n
        h0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][rrr]][2]+lag
        tem2=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        #计算距离
        drill2=tem2.reshape(ss,1)
        d.append(hamming_distance(drill1,drill2))
        #p.append(rrr)
    
    #N选1
    
    
    
    
    
    
    si=getListMinNumIndex(d,N)
    #Blist=[]
    rrr=random.randint(0,len(si)-1)
    t=si[rrr]
    #t=p[t]
    h0=Fzuobiaolist[dark][Fclass[dark][cc][t]][0]+lag
    x0=Fzuobiaolist[dark][Fclass[dark][cc][t]][1]+lag
    y0=Fzuobiaolist[dark][Fclass[dark][cc][t]][2]+lag
    resulttem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
    '''
    for n in range(len(si)):
        t=si[n]
        t=p[t]
        h0=Fzuobiaolist[dark][Fclass[dark][cc][t]][0]+lag
        x0=Fzuobiaolist[dark][Fclass[dark][cc][t]][1]+lag
        y0=Fzuobiaolist[dark][Fclass[dark][cc][t]][2]+lag

        resulttem=template1(Exm2,template_h,template_x,template_y,h0,x0,y0)
        
        Blist.append(resulttem)
    '''
    #resulttem=zhifangtumatch(Alist,Blist,valist,1)
    
    return resulttem

def initialgridAIClassPY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm1,Exm2,Fzuobiaolist,Fclass,N,Fenjilist,hardlist,valist):
    #优化版处理分级数据
    #valist为该模型中所有值类型列表
    #此处m2为软数据
    #添加了直方图匹配
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIPY(m,m2,template_h,template_x,template_y,lag)
    #lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#软数据拓展
    
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
  
    
  
    for n in range(len(lujing)):
        print(n,len(lujing))
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
        Alist=Alistget(ms,h1,x1,y1,template_h)#获取待模拟区域周遭模版
        k=0#重叠区计数器
        flag=0
        tem=patternsearchClassNHY(o1,ms,Fclass,Fzuobiaolist,dark,lag,N,Alist,valist)
            
            
       

        #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
        #ms=template1R(ms,tem,h1,x1,y1)   
        ms=template1RAIFenji(ms,ms2,ms2[h1,x1,y1],tem,h1,x1,y1)
    m=cut(ms,lag)
    return m





def gosiminitialAIClassPY(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,valist,softTilist,softTizuobiaolist,msoft):
    #全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
    #m为已经导入了Ti的模拟网格
    #优化版
    hardlist=[]
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

 
    my_file = Path("./database/Sclass.npy")
    if my_file.exists():
        Softdatabase=np.load('./database/Sdatabase.npy')
        Softzuobiaolist=np.load('./database/Softzuobiaolist.npy')
        Fdatabase=np.load('./database/Fdatabase.npy')
        Fzuobiaolist=np.load('./database/Fzuobiaolist.npy')
        Sclass=np.load('./database/Sclass.npy')
        Fclass=np.load('./database/Fclass.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Softdatabase,Softzuobiaolist=databasebuildAIY(m2,template_h,template_x,template_y,lag)#软数据数据库构建
        np.save('./database/Sdatabase.npy',Softdatabase)#softdata为扩充后的网格
        np.save('./database/Softzuobiaolist.npy',Softzuobiaolist)
        Fdatabase,Fzuobiaolist=databasebuildAIClassY(m,m2,template_h,template_x,template_y,lag,Fenjilist)
        np.save('./database/Fdatabase.npy',Fdatabase)
        np.save('./database/Fzuobiaolist.npy',Fzuobiaolist)
        
        #以及分级数据库构建
        Sclass,Fclass=databaseclusterAIY(Softdatabase,Fdatabase,Softzuobiaolist,Fzuobiaolist,template_h,template_x,template_y,lag,U)
        
        print(len(Sclass))
        for n in range(len(Fclass)):
            print(len(Fclass[n]))
        np.save('./database/Sclass.npy',Sclass)
        np.save('./database/Fclass.npy',Fclass)


        print('Patterndatabase has been builded!')
    time_end1=time.time()
    print('timecost:')
    print(time_end1-time_start1)

    time_start=time.time()
    print('initial start:')
    data=Softdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/softdatabas.vtk') 
    data=Fdatabase.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/Fdatabase.vtk') 
    #print(m.shape[0],m.shape[1],m.shape[2])
    #print(Softdatabase.shape[0],Softdatabase.shape[1],Softdatabase.shape[2])
    #mm=m2.copy()
    #m2=initialgridAIAFA(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Sclass,Softzuobiaolist,N,hardlist,Fenjilist)

###################################

    #mm=-np.ones((int(m2.shape[0]),int(m2.shape[1]),int(m2.shape[2])),int)

    #Tilist=[]
    #Tizuobiaolist=[]


    #m2= patchmatchmultiTiB(m2,softTilist,size,itr,1)
    #print("softdata em done")
       

################################


    
    
    

    m=initialgridAIClassPY(m,msoft,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Fdatabase,Fzuobiaolist,Fclass,N,Fenjilist,hardlist,valist)
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

def GosimAIClassPY(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,valist,msoft):
    #带直方图匹配版本
    time_starts=time.time()#计时开始
    #IDW算法
    #初始模型构建 m为已经构建好的模拟网格
    m,m2,Tilist,Tizuobiaolist,softTilist,softTizuobiaolist=sectionloadandextendFenjiAFA(m,patternSizex,patternSizey,1,1)
    #m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    #m2=pictureclass(m,Fenjilist,Fenjineironglist)
    #m2,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m2,patternSizex,patternSizey,0,1)
    #m2为辅助分级模型
    #使用外部输入的分级
    print('Please wait for the initial simulated grid building:')
    m=gosiminitialAIClassPY(m,m2,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,Fenjilist,valist,softTilist,softTizuobiaolist,msoft)
    
    print('initial done')

    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/outputinitial3.vtk') 
    
    sancheck=1#sectionloadandextend倍率机制
    #EM迭代阶段
    #EM迭代阶段
    for ni in range(len(scale)):
        sancheck=sancheck*scale[ni]
        #构建新初始网格mm
        mm=-np.ones((int(m.shape[0]*scale[ni]),int(m.shape[1]*scale[ni]),int(m.shape[2]*scale[ni])),int)

        Tilist=[]
        Tizuobiaolist=[]
        mm,Tilist,Tizuobiaolist=Recodesectionloadandextend(mm,patternSizex,patternSizey,1,sancheck)


        mm=extendTimodel(mm,patternSizeh,patternSizex,patternSizey)
        np.save('./output/Ti.npy',mm)
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
        m= patchmatchmultiTiB(m,Tilist,size,itr,1)
        #m= patchmatchmultiTiBforz(m,Tilist,size,itr,1,valist)
        CTI=[]#检测使用率剖面
        #m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        #m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,8,0)#并行进程的数目
        #计算量加大过多
        #m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path="./output/reconstruction.npy"
        data=m.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
         dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, './output/outputinitial2.vtk') 
        np.save(path,m)
        time_end=time.time()
        #size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end-time_start)
        #print('对比1：',TIvsM(mm,m,patternSizeh,lag,1000))
        #print('对比2：',TIvsM2(mm,m,patternSizeh,lag,1000))
        Exm=np.pad(mm,lag,'edge')
        #print('对比3，JS散度',TIandModelcompare(Exm,m,lag,patternSizeh,patternSizex,patternSizey))
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
    
    
    
    return m
###########################################载入已经模拟成层分类版####################################################################################
def gosiminitialAIClassPY2(m, m2, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U, Fenjilist, valist,
                           softTilist, softTizuobiaolist, msoft, alreadylist):
    # 全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
    # m为已经导入了Ti的模拟网格
    # 优化版
    hardlist = []
    time_start1 = time.time()
    m = extendTimodel(m, template_h, template_x, template_y)  # 拓展模拟网格
    m2 = extendTimodel(m2, template_h, template_x, template_y)  # 拓展模拟网格
    moft = extendTimodel(msoft, template_h*3, template_x*3, template_y*3)  # 拓展模拟网格
    # np.save('./output/wtf.npy',m)

    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/outputinitial1.vtk')

    print('extend done')

    my_file = Path("./database/Sclass.npy")
    if my_file.exists():
        Softdatabase = np.load('./database/Sdatabase.npy')
        Softzuobiaolist = np.load('./database/Softzuobiaolist.npy')
        Fdatabase = np.load('./database/Fdatabase.npy')
        Fzuobiaolist = np.load('./database/Fzuobiaolist.npy')
        Sclass = np.load('./database/Sclass.npy')
        Fclass = np.load('./database/Fclass.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Softdatabase, Softzuobiaolist = databasebuildAIY(m2, template_h, template_x, template_y, lag)  # 软数据数据库构建
        np.save('./database/Sdatabase.npy', Softdatabase)  # softdata为扩充后的网格
        np.save('./database/Softzuobiaolist.npy', Softzuobiaolist)
        Fdatabase, Fzuobiaolist = databasebuildAIClassY(m, m2, template_h, template_x, template_y, lag, Fenjilist)
        np.save('./database/Fdatabase.npy', Fdatabase)
        np.save('./database/Fzuobiaolist.npy', Fzuobiaolist)

        # 以及分级数据库构建
        Sclass, Fclass = databaseclusterAIY(Softdatabase, Fdatabase, Softzuobiaolist, Fzuobiaolist, template_h,
                                            template_x, template_y, lag, U)

        print(len(Sclass))
        for n in range(len(Fclass)):
            print(len(Fclass[n]))
        np.save('./database/Sclass.npy', Sclass)
        np.save('./database/Fclass.npy', Fclass)

        print('Patterndatabase has been builded!')
    time_end1 = time.time()
    print('timecost:')
    print(time_end1 - time_start1)

    time_start = time.time()
    print('initial start:')
    data = Softdatabase.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/softdatabas.vtk')
    data = Fdatabase.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/Fdatabase.vtk')
    # print(m.shape[0],m.shape[1],m.shape[2])
    # print(Softdatabase.shape[0],Softdatabase.shape[1],Softdatabase.shape[2])
    # mm=m2.copy()
    # m2=initialgridAIAFA(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Sclass,Softzuobiaolist,N,hardlist,Fenjilist)

    ###################################

    # mm=-np.ones((int(m2.shape[0]),int(m2.shape[1]),int(m2.shape[2])),int)

    # Tilist=[]
    # Tizuobiaolist=[]

    # m2= patchmatchmultiTiB(m2,softTilist,size,itr,1)
    # print("softdata em done")

    ################################

    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if (msoft[h, x, y] in alreadylist) and (m[h,x,y]==-1):
                    
                    m[h,x,y] = msoft[h, x, y]
    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/outputinitial2.vtk')
    m = initialgridAIClassPY(m, msoft, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Softdatabase,
                              Fdatabase, Fzuobiaolist, Fclass, N, Fenjilist, hardlist, valist)
    time_end = time.time()



    print('initial done')
    print('timecost:')
    print(time_end - time_start)
    # 初始化
    return m


def GosimAIClassPY2(m, patternSizeh, patternSizex, patternSizey, lag, lag_h, lag_x, lag_y, N, U, scale, size, itr,
                    Fenjilist, valist, msoft, alreadylist):
    # 带直方图匹配版本
    time_starts = time.time()  # 计时开始
    # IDW算法
    # 初始模型构建 m为已经构建好的模拟网格
    m, m2, Tilist, Tizuobiaolist, softTilist, softTizuobiaolist = sectionloadandextendFenjiAFA(m, patternSizex,
                                                                                               patternSizey, 1, 1)
    # m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    # m2=pictureclass(m,Fenjilist,Fenjineironglist)
    # m2,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m2,patternSizex,patternSizey,0,1)
    # m2为辅助分级模型
    # 使用外部输入的分级
    print('Please wait for the initial simulated grid building:')
    m = gosiminitialAIClassPY2(m, m2, patternSizeh, patternSizex, patternSizey, lag, lag_h, lag_x, lag_y, N, U,
                               Fenjilist, valist, softTilist, softTizuobiaolist, msoft,alreadylist)

    print('initial done')

    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/outputinitial2.vtk')

    sancheck = 1  # sectionloadandextend倍率机制
    # EM迭代阶段
    # EM迭代阶段
    for ni in range(len(scale)):
        sancheck = sancheck * scale[ni]
        # 构建新初始网格mm
        mm = -np.ones((int(m.shape[0] * scale[ni]), int(m.shape[1] * scale[ni]), int(m.shape[2] * scale[ni])), int)

        Tilist = []
        Tizuobiaolist = []
        mm, Tilist, Tizuobiaolist = Recodesectionloadandextend(mm, patternSizex, patternSizey, 1, sancheck)

        mm = extendTimodel(mm, patternSizeh, patternSizex, patternSizey)
        np.save('./output/Ti.npy', mm)
        # 上一个尺度升采样
        m = simgridex(m, scale[ni])
        # 重新导入
        for hi in range(m.shape[0]):
            for xi in range(m.shape[1]):
                for yi in range(m.shape[2]):
                    if mm[hi, xi, yi] != -1:
                        m[hi, xi, yi] = mm[hi, xi, yi]
        print("该尺度扩展Ti完成")
        time_start = time.time()  # 计时开始
        # np.save('./output/m1.npy',m)
        m = patchmatchmultiTiB(m, Tilist, size, itr, 1)
        # m= patchmatchmultiTiBforz(m,Tilist,size,itr,1,valist)
        CTI = []  # 检测使用率剖面
        # m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        # m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,8,0)#并行进程的数目
        # 计算量加大过多
        # m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path = "./output/reconstruction.npy"
        data = m.transpose(-1, -2, 0)  # 转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                              dimensions=data.shape)
        grid.point_data.scalars = np.ravel(data, order='F')
        grid.point_data.scalars.name = 'lithology'
        write_data(grid, './output/outputinitial4.vtk')
        np.save(path, m)
        time_end = time.time()
        # size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end - time_start)
        # print('对比1：',TIvsM(mm,m,patternSizeh,lag,1000))
        # print('对比2：',TIvsM2(mm,m,patternSizeh,lag,1000))
        Exm = np.pad(mm, lag, 'edge')
        # print('对比3，JS散度',TIandModelcompare(Exm,m,lag,patternSizeh,patternSizex,patternSizey))
    time_ends = time.time()
    print('总消耗时间:')
    print(time_ends - time_starts)
    print('Simulating done!')
    ##########################转换vtk格式########################

    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    path = "./output/output.vtk"
    write_data(grid, path)

    print('并行程序结束')

    return m

#######################################################################################################################改进版，根据输入的valist长度判断是否需要分级


def template1RAIFenjione(Ti,Ti2,dark,tem,h0,x0,y0,valueneironglist):#非柱状模板返还判断硬数据版
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
                    if tem[nn1,nn2,nn3] in valueneironglist:
                       Ti[n1,n2,n3]=tem[nn1,nn2,nn3]
                    else:
                       Ti[n1,n2,n3]=-2
                nn3=nn3+1
                #print(nn3)
            nn2=nn2+1
            nn3=0
        nn1=nn1+1
        nn2=0

        #print(nn2)
    return Ti  #提取坐标x0,y0处模板  


def initialgridAIClassPYone(m,m2,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Exm1,Exm2,Fzuobiaolist,Fclass,N,Fenjilist,Fenjineironglist,hardlist,valist):
    #优化版处理分级数据
    #valist为该模型中所有值类型列表
    #此处m2为软数据
    #添加了直方图匹配
    lujing=[]
    Banlujing=[]#已模拟黑名单
    lujing=initialroadlistAIPY(m,m2,template_h,template_x,template_y,lag)
    #lujing=initialroadlistAIFenjiZ(m,m2,template_h,template_x,template_y,lag,Fenjilist)
    print('initialize start')
    #print len(lujing)
    ms=np.pad(m,lag,'edge')#拓展
    ms2=np.pad(m2,lag,'edge')#软数据拓展
    
    H=m.shape[0]
    X=m.shape[1]
    Y=m.shape[2]
  
    
  
    for n in range(len(lujing)):
        print(n,len(lujing))
        h1=lujing[n][0]+lag
        x1=lujing[n][1]+lag
        y1=lujing[n][2]+lag
        dark=Fenjilist.index(ms2[h1,x1,y1])#获得分级类别序号
        o1=template1(ms,template_h,template_x,template_y,h1,x1,y1)
        Alist=Alistget(ms,h1,x1,y1,template_h)#获取待模拟区域周遭模版
        k=0#重叠区计数器
        flag=0
        tem=patternsearchClassNHY(o1,ms,Fclass,Fzuobiaolist,dark,lag,N,Alist,valist)
            
            
       

        #m2=TemplateHard(m2,tem,h1,x1,y1,hardlist)
        #ms=template1R(ms,tem,h1,x1,y1)   
        ms=template1RAIFenjione(ms,ms2,ms2[h1,x1,y1],tem,h1,x1,y1,Fenjineironglist[dark])
    flag=100
    for h in range(ms.shape[0]):
        for x in range(ms.shape[1]):
            for y in range(ms.shape[2]):
                if ms[h,x,y]==-2:
                   ms[h,x,y]=-1
    while flag==100:
       ms = extendTimodel(ms, template_h, template_x, template_y)
       flag=99
       for h in range(ms.shape[0]):
           for x in range(ms.shape[1]):
               for y in range(ms.shape[2]):
                   if ms[h,x,y]==-1:
                      flag=100
    print(ms)
    m=ms[lag:m.shape[0]+lag,lag:m.shape[1]+lag,lag:m.shape[2]+lag]
    print(m)
    np.save('./output/initial.npy',m)
    return m

def gosiminitialAIClassPYone(m, m2, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U, Fenjilist,Fenjineironglist, valist,
                           softTilist, softTizuobiaolist, msoft, alreadylist):
    # 全自动初始化流程整合,m为导入好剖面的待模拟网格,m2为导入了分级剖面的待填充分级模型
    # m为已经导入了Ti的模拟网格
    # 优化版
    hardlist = []
    time_start1 = time.time()
    m = extendTimodel(m, template_h, template_x, template_y)  # 拓展模拟网格
    m2 = extendTimodel(m2, template_h, template_x, template_y)  # 拓展模拟网格
    moft = extendTimodel(msoft, template_h*3, template_x*3, template_y*3)  # 拓展模拟网格
    # np.save('./output/wtf.npy',m)

    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/outputinitial1.vtk')

    print('extend done')

    my_file = Path("./database/Sclass.npy")
    if my_file.exists():
        Softdatabase = np.load('./database/Sdatabase.npy')
        Softzuobiaolist = np.load('./database/Softzuobiaolist.npy')
        Fdatabase = np.load('./database/Fdatabase.npy')
        Fzuobiaolist = np.load('./database/Fzuobiaolist.npy')
        Sclass = np.load('./database/Sclass.npy')
        Fclass = np.load('./database/Fclass.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        Softdatabase, Softzuobiaolist = databasebuildAIY(m2, template_h, template_x, template_y, lag)  # 软数据数据库构建
        np.save('./database/Sdatabase.npy', Softdatabase)  # softdata为扩充后的网格
        np.save('./database/Softzuobiaolist.npy', Softzuobiaolist)
        Fdatabase, Fzuobiaolist = databasebuildAIClassY(m, m2, template_h, template_x, template_y, lag, Fenjilist)
        np.save('./database/Fdatabase.npy', Fdatabase)
        np.save('./database/Fzuobiaolist.npy', Fzuobiaolist)

        # 以及分级数据库构建
        Sclass, Fclass = databaseclusterAIY(Softdatabase, Fdatabase, Softzuobiaolist, Fzuobiaolist, template_h,
                                            template_x, template_y, lag, U)

        print(len(Sclass))
        for n in range(len(Fclass)):
            print(len(Fclass[n]))
        np.save('./database/Sclass.npy', Sclass)
        np.save('./database/Fclass.npy', Fclass)

        print('Patterndatabase has been builded!')
    time_end1 = time.time()
    print('timecost:')
    print(time_end1 - time_start1)

    time_start = time.time()
    print('initial start:')
    data = Softdatabase.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/softdatabas.vtk')
    data = Fdatabase.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/Fdatabase.vtk')
    # print(m.shape[0],m.shape[1],m.shape[2])
    # print(Softdatabase.shape[0],Softdatabase.shape[1],Softdatabase.shape[2])
    # mm=m2.copy()
    # m2=initialgridAIAFA(m,template_h,template_x,template_y,lag,lag_h,lag_x,lag_y,Softdatabase,Sclass,Softzuobiaolist,N,hardlist,Fenjilist)

    ###################################

    # mm=-np.ones((int(m2.shape[0]),int(m2.shape[1]),int(m2.shape[2])),int)

    # Tilist=[]
    # Tizuobiaolist=[]

    # m2= patchmatchmultiTiB(m2,softTilist,size,itr,1)
    # print("softdata em done")

    ################################

    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if (msoft[h, x, y] in alreadylist) and (m[h,x,y]==-1):
                    
                    m[h,x,y] = msoft[h, x, y]
    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/outputinitial2.vtk')
    m= initialgridAIClassPYone(m, msoft, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Softdatabase,
                              Fdatabase, Fzuobiaolist, Fclass, N, Fenjilist, Fenjineironglist,hardlist, valist)
    time_end = time.time()


    print('initial done')
    print('timecost:')
    print(time_end - time_start)
    # 初始化
    return m

def GosimAIClassPYone(m, patternSizeh, patternSizex, patternSizey, lag, lag_h, lag_x, lag_y, N, U, scale, size, itr,
                    Fenjilist,Fenjineironglist, valist, msoft, alreadylist):
    #分级确认版
    time_starts = time.time()  # 计时开始

    '''
    for npsa in range(len(Fenjineironglist)):
        if len(Fenjineironglist[npsa])==1 and Fenjineironglist[npsa][0] not in alreadylist:
           alreadylist.append(Fenjineironglist[npsa][0])
    '''


    # IDW算法
    # 初始模型构建 m为已经构建好的模拟网格
    m, m2, Tilist, Tizuobiaolist, softTilist, softTizuobiaolist = sectionloadandextendFenjiAFAone(m, patternSizex,patternSizey, 1, 1)
    # m,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m,patternSizex,patternSizey,0,1)
    # m2=pictureclass(m,Fenjilist,Fenjineironglist)
    # m2,Tilist,Tizuobiaolist=sectionloadandextendFenji2(m2,patternSizex,patternSizey,0,1)
    # m2为辅助分级模型
    # 使用外部输入的分级
    print('Please wait for the initial simulated grid building:')
    m = gosiminitialAIClassPYone(m, m2, patternSizeh, patternSizex, patternSizey, lag, lag_h, lag_x, lag_y, N, U,Fenjilist, Fenjineironglist,valist, softTilist, softTizuobiaolist, msoft,alreadylist)
    m=np.load('./output/initial.npy')



    sancheck = 1  # sectionloadandextend倍率机制
    # EM迭代阶段
    # EM迭代阶段
    for ni in range(len(scale)):
        sancheck = sancheck * scale[ni]
        # 构建新初始网格mm
        mm = -np.ones((int(m.shape[0] * scale[ni]), int(m.shape[1] * scale[ni]), int(m.shape[2] * scale[ni])), int)

        Tilist = []
        Tizuobiaolist = []
        mm, Tilist, Tizuobiaolist = Recodesectionloadandextend(mm, patternSizex, patternSizey, 1, sancheck)

        mm = extendTimodel(mm, patternSizeh, patternSizex, patternSizey)
        np.save('./output/Ti.npy', mm)
        # 上一个尺度升采样
        m = simgridex(m, scale[ni])
        # 重新导入
        for hi in range(m.shape[0]):
            for xi in range(m.shape[1]):
                for yi in range(m.shape[2]):
                    if mm[hi, xi, yi] != -1:
                        m[hi, xi, yi] = mm[hi, xi, yi]
        print("该尺度扩展Ti完成")
        time_start = time.time()  # 计时开始
        # np.save('./output/m1.npy',m)
        m = patchmatchmultiTiB(m, Tilist, size, itr, 1)
        # m= patchmatchmultiTiBforz(m,Tilist,size,itr,1,valist)
        CTI = []  # 检测使用率剖面
        # m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        # m,CTI=Recodepatchmatch(m,mm,Tilist,Tizuobiaolist,size,itr,8,0)#并行进程的数目
        # 计算量加大过多
        # m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path = "./output/reconstruction.npy"
        data = m.transpose(-1, -2, 0)  # 转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                              dimensions=data.shape)
        grid.point_data.scalars = np.ravel(data, order='F')
        grid.point_data.scalars.name = 'lithology'
        write_data(grid, './output/outputinitial4.vtk')
        np.save(path, m)
        time_end = time.time()
        # size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end - time_start)
        # print('对比1：',TIvsM(mm,m,patternSizeh,lag,1000))
        # print('对比2：',TIvsM2(mm,m,patternSizeh,lag,1000))
        #Exm = np.pad(mm, lag, 'edge')
        # print('对比3，JS散度',TIandModelcompare(Exm,m,lag,patternSizeh,patternSizex,patternSizey))
    time_ends = time.time()
    print('总消耗时间:')
    print(time_ends - time_starts)
    print('Simulating done!')
    ##########################转换vtk格式########################

    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    path = "./output/output.vtk"
    write_data(grid, path)

    print('并行程序结束')

    return m
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

#一次模拟的个数
content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
Modelcount=int(''.join(string1))
#print Modelcount

#####################运行################################################################
msoft=np.load('./output/soft.npy')
flag=100
listc=[]
while flag==100:
  msoft= extendTimodel(msoft, 3, 3, 3)
  flag=99
  for h in range(msoft.shape[0]):
      for x in range(msoft.shape[1]):
          for y in range(msoft.shape[2]):
              if msoft[h,x,y]==-1:
                 flag=100
              if msoft[h,x,y] not in listc:
                 listc.append(msoft[h,x,y])
print(listc)

#msoft=simgridex(msoft, 0.5)

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
m=-np.ones((Mh,Mx,My),int)#默认空值为-1
#Fenjilist=[10,20,30,40,50,60]#softdata cata
#GosimAIClassY(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist)
valist=[]
Fenjilist=[]
Fenjineironglist=[]

alreadylist=[]
#GosimAIClassHY(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,valist)
m=GosimAIClassPYone(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,scale,size,itr,Fenjilist,Fenjineironglist,valist,msoft,alreadylist)


data = m.transpose(-1, -2, 0)  # 转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                      dimensions=data.shape)
grid.point_data.scalars = np.ravel(data, order='F')
grid.point_data.scalars.name = 'lithology'
path = "./output/output" + str(id) + ".vtk"
write_data(grid, path)

print('并行程序结束')
