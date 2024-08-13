#!/usr/bin/env python
# coding: utf-8

# In[141]:
import keras
from keras.models import Sequential 
from keras.layers import Dense,Dropout
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math 
from MDS import*
import tensorflow as tf
from AIinitial import*
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d
from tvtk.api import tvtk, write_data 
import itertools as it
from matplotlib import pyplot
#################################################断裂带建模#################################
def twodsurface(m,value):#转换为二维俯视面#有多层翻转的用这个
    L=m.shape[1]
    W=m.shape[2]
    mm=np.pad(m,((1,1),(0,0),(0,0)),'constant', constant_values=0)
    top=[]
    for n in range(3):
        top.append(np.zeros((L,W),int))
    bottom=[]
    for n in range(3):
        bottom.append(np.zeros((L,W),int))
    for x in range(L):
        for y in range(W):
            count=0
            for h in range(1,m.shape[0]+1):
                if count==0:
                    if mm[h,x,y]==value:
                        if mm[h-1,x,y]!=value:
                            top[0][x,y]=h
                        if mm[h+1,x,y]!=value:
                            bottom[0][x,y]=h
                            count=1
                elif count==1:
                    if mm[h,x,y]==value:
                        if mm[h-1,x,y]!=value:
                            top[1][x,y]=h
                        if mm[h+1,x,y]!=value:
                            bottom[1][x,y]=h
                            count=2
                elif count==2:
                    if mm[h,x,y]==value:
                        if mm[h-1,x,y]!=value:
                            top[2][x,y]=h
                        if mm[h+1,x,y]!=value:
                            bottom[2][x,y]=h
                            count=3
                else:
                    print('超过三层翻转')
    return top,bottom

def twodfaultsurface(m,value):#fault trans to 二维俯视面.value为模拟值
    L=m.shape[1]
    W=m.shape[2]
    mm=np.pad(m,((1,1),(0,0),(0,0)),'constant', constant_values=-1)
    top=-np.ones((L,W),int)
    bottom=-np.ones((L,W),int)
    for x in range(L):
        for y in range(W):
            count=0
            for h in range(1,m.shape[0]+1):
                if mm[h,x,y]==value:
                    if mm[h-1,x,y]!=value and count==0:
                        top[x,y]=h
                    if mm[h+1,x,y]!=value:
                        bottom[x,y]=h
                        count=1
    return top,bottom
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

def twodfaultsurfacealf(hard,value,usinglist):#fault trans to 二维俯视面.value为模拟值,添加去除水平顶底功能
    #usinglist为使用的剖面和内容
    
    
    
    jvalue=99999
    
    L=hard.shape[1]
    W=hard.shape[2]
    
    
    top=-np.ones((L,W),int)
    bottom=-np.ones((L,W),int)
    for n in range(len(usinglist)):
        guding=[]
        mm=np.ones((hard.shape[0],L,W),int)
        file1=open('./Ti/Tiparameter.txt')
        for nb in range(6*(usinglist-1)+1):
            content=file1.readline()
   

        for aa in range(6):
            content=file1.readline()
            string1=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string1))
            guding.append(xx)
        file1.close()
        path='./Ti/'+str(usinglist[n])+'.bmp'
        section=cv2.imread(path,0)
        #print(guding)

        top1=-np.ones((L,W),int)
        bottom1=-np.ones((L,W),int)
        #print(guding[0],guding[1],guding[2],guding[3])
        mm=sectionload_xG2(mm,section,guding[0],guding[1],guding[2],guding[3],guding[4],guding[5],jvalue)#载入剖面
        mmm=np.pad(mm,((1,1),(0,0),(0,0)),'constant', constant_values=-1)

        highlist=[]
        lowlist=[]
        for x in range(L):
            for y in range(W):
                count1=0
                count2=0

                for h in range(1,hard.shape[0]+1):
                    if mmm[h,x,y]==value:
                        if mmm[h-1,x,y]!=value and count1==0:
                            top1[x,y]=h
                            highlist.append(h)
                            count1=1
                        if mmm[h+1,x,y]!=value and count2==0:
                            bottom1[x,y]=h
                            lowlist.append(h)
                            count2=1
        if len(highlist)!=0:
           ah=max(highlist)
           bh=min(lowlist)
           for x in range(L):
               for y in range(W):    
                   if top1[x,y]!=ah and top1[x,y]!=-1:
                      top[x,y]=top1[x,y]
                   if bottom1[x,y]!=bh and bottom1[x,y]!=-1:
                      bottom[x,y]=bottom1[x,y]

    
    return top,bottom
def twodfaultsurfacedelta(hard,value,usinglist,nanvalue):#fault trans to 二维俯视面.value为模拟值,添加去除水平顶底功能
    #usinglist为使用的剖面和内容
    #导出顶底图,nanvalue为将-1转化为的特定值，方便导成图片
    
    
    jvalue=99999
    
    L=hard.shape[1]
    W=hard.shape[2]
    
    
    top=-np.ones((L,W),int)
    bottom=-np.ones((L,W),int)
    for n in range(len(usinglist)):
        guding=[]
        mm=np.ones((hard.shape[0],L,W),int)
        file1=open('./Ti/Tiparameter.txt')
        for nb in range(6*(usinglist[n]-1)+1):
            content=file1.readline()
   

        for aa in range(6):
            content=file1.readline()
            string1=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string1))
            guding.append(xx)
        file1.close()
        path='./Ti/'+str(usinglist[n])+'.bmp'
        section=cv2.imread(path,0)
        #print(guding)

        top1=-np.ones((L,W),int)
        bottom1=-np.ones((L,W),int)
        #print(guding[0],guding[1],guding[2],guding[3])
        mm=sectionload_xG2(mm,section,guding[0],guding[1],guding[2],guding[3],guding[4],guding[5],jvalue)#载入剖面
        mmm=np.pad(mm,((1,1),(0,0),(0,0)),'constant', constant_values=-1)

        highlist=[]
        lowlist=[]
        for x in range(L):
            for y in range(W):
                count1=0
                count2=0

                for h in range(1,hard.shape[0]+1):
                    if mmm[h,x,y]==value:
                        if mmm[h-1,x,y]!=value and count1==0:
                            top1[x,y]=h
                            highlist.append(h)
                            count1=1
                        if mmm[h+1,x,y]!=value and count2==0:
                            bottom1[x,y]=h
                            lowlist.append(h)
                            count2=1

        if len(highlist)!=0:
           ah=min(highlist)
           bh=max(lowlist)
           for x in range(L):
               for y in range(W):    
                   if top1[x,y]!=ah and top1[x,y]!=-1:
                      top[x,y]=top1[x,y]
                   if bottom1[x,y]!=bh and bottom1[x,y]!=-1:
                      bottom[x,y]=bottom1[x,y]
    top2=top.copy()
    bottom2=bottom.copy()
    for x in range(L):
        for y in range(W):
            if top2[x,y]==-1:
               top2[x,y]=nanvalue
            if bottom2[x,y]==-1:
               bottom2[x,y]=nanvalue

    return top,bottom    

def twodfaultsurface2(m,value):#fault trans to 二维俯视面,value为背景值
    L=m.shape[1]
    W=m.shape[2]
    mm=np.pad(m,((1,1),(0,0),(0,0)),'constant', constant_values=999)
    top=-np.ones((L,W),int)
    bottom=-np.ones((L,W),int)
    for x in range(L):
        for y in range(W):
            count=0
            for h in range(1,m.shape[0]+1):
                if mm[h,x,y]!=value:
                    if mm[h-1,x,y]==value:
                        top[x,y]=h-1
                    if mm[h+1,x,y]==value:
                        bottom[x,y]=h-1
                        count=1
    return top,bottom
def duliyizhi(value,m,h,x,y,banlist,dianlist):#勾选独立同值个体智能识别系统
    #print h,x,y
    flag=[]
    if m[h,x,y]==value:
        dianlist.append((h,x,y))
        for n1 in range(-1,2):
            for n2 in range(-1,2):
                for n3 in range(-1,2):
                    if (h+n1>=0)&(h+n1<m.shape[0])&(x+n2>=0)&(x+n2<m.shape[1])&(y+n3>=0)&(y+n3<m.shape[2]):
                         if [h+n1,x+n2,y+n3] not in banlist:
                                if m[h+n1,x+n2,y+n3]==value:
                                    flag.append((h+n1,x+n2,y+n3))
        #遍历之后
        banlist.append([h,x,y])
        for n in range(len(flag)):
            dianlist=duliyizhi(value,m,flag[n][0],flag[n][1],flag[n][2],banlist,dianlist)       
    #print surface
    return dianlist

def devilmaycry(m,value):#独立同值个体勾选细分,细分值从7开始
    banlist=[] 
    dante=np.zeros_like(m)
    xifenzhi=7
    jishuqi=0
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if m[h,x,y]==value:
                    if [h,x,y] not in banlist:
                        dianlist=[]
                        dianlist=duliyizhi(value,m,h,x,y,banlist,dianlist)
                        for n in range(len(dianlist)):
                            dante[dianlist[n][0],dianlist[n][1],dianlist[n][2]]=xifenzhi
                        xifenzhi=xifenzhi+1
                        jishuqi=jishuqi+1
    print('细分个体数：')
    print(jishuqi)
    return dante

def minzhuziyou(value,m,h,x,y,banlist,dianlist):#勾选独立非同值个体智能识别系统，value为背景值（默认为1或者0
    #print h,x,y
    flag=[]
    if m[h,x,y]!=value:
        dianlist.append((h,x,y))
        for n1 in range(-1,2):
            for n2 in range(-1,2):
                for n3 in range(-1,2):
                    if (h+n1>=0)&(h+n1<m.shape[0])&(x+n2>=0)&(x+n2<m.shape[1])&(y+n3>=0)&(y+n3<m.shape[2]):
                         if [h+n1,x+n2,y+n3] not in banlist:
                                if m[h+n1,x+n2,y+n3]!=value:
                                    flag.append((h+n1,x+n2,y+n3))
        #遍历之后
        banlist.append([h,x,y])
        for n in range(len(flag)):
            dianlist=minzhuziyou(value,m,flag[n][0],flag[n][1],flag[n][2],banlist,dianlist)       
    #print surface
    return dianlist

def devilmaydie(m,value):#独立非同值个体勾选细分,细分值从7开始，value为背景值（一般为0或者1
    banlist=[] 
    vergil=np.zeros_like(m)
    xifenzhi=7
    jishuqi=0
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if m[h,x,y]!=value:
                    if [h,x,y] not in banlist:
                        dianlist=[]
                        dianlist=minzhuziyou(value,m,h,x,y,banlist,dianlist)
                        for n in range(len(dianlist)):
                            vergil[dianlist[n][0],dianlist[n][1],dianlist[n][2]]=xifenzhi
                        xifenzhi=xifenzhi+1
                        jishuqi=jishuqi+1
    print('细分个体数：')
    print(jishuqi)
    return vergil



def minzhuziyou2(value,m,x,y,banlist,dianlist):#勾选独立非同值个体智能识别系统，value为背景值（默认为1或者0
    #print x,y
    #2d特供版
    flag=[]
    if m[x,y]!=value:
        dianlist.append((x,y))
        for n2 in range(-1,2):
            for n3 in range(-1,2):
                if (x+n2>=0)&(x+n2<m.shape[0])&(y+n3>=0)&(y+n3<m.shape[1]):
                        if [x+n2,y+n3] not in banlist:
                            if m[x+n2,y+n3]!=value:
                                flag.append((x+n2,y+n3))
        #遍历之后
        banlist.append([x,y])
        for n in range(len(flag)):
            dianlist=minzhuziyou2(value,m,flag[n][0],flag[n][1],banlist,dianlist)       
    #print surface
    return dianlist

def devilmaycry2(m,value):#独立非同值个体勾选细分,细分值从7开始，value为背景值（一般为0或者1
    #2d特供版
    banlist=[] 
    vergil=np.zeros_like(m)
    xifenzhi=7
    jishuqi=0
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            if m[x,y]!=value:
                if [x,y] not in banlist:
                    dianlist=[]
                    dianlist=minzhuziyou2(value,m,x,y,banlist,dianlist)
                    for n in range(len(dianlist)):
                        vergil[dianlist[n][0],dianlist[n][1]]=xifenzhi
                    xifenzhi=xifenzhi+1
                    jishuqi=jishuqi+1
    #print('细分个体数：')
    #print(jishuqi)
    return vergil,jishuqi


def jiugongge(m,value,x,y):#返还该点周围背景值个数
    n=0
    m2=np.pad(m,((1,1),(1,1)),'constant', constant_values=0)
    for n1 in range(-1,2):
        for n2 in range(-1,2):
            if m2[x+n1+1,y+n2+1]==value:
                n=n+1
    return n

def maxmatrix(M):
    #最大坐标
    x,y = divmod(np.argmax(M), np.shape(M)[1])
    return x,y
    
def minmatrix(M):
    #最小坐标
    x,y = divmod(np.argmin(M), np.shape(M)[1])
    return x,y
 
'''
def lianliankan(section,value,fenleisection,S):#二维连线内插,value 为断层带值,zhilist为不同源同值断层的分类图,S为源的分类值
    L=section.shape[0]
    W=section.shape[1]
    minlist=[]
    maxlist=[]
    C=[]
    for n in range(len(S)):
        for x in range(L):
            for y in range(W):
                if fenleisection[x,y]==S[n]:
                    if jiugongge(m,value,x,y)>=7:
                        C.append((x,y))
        if len(C)>1:
            if m[C[0][0],C[0][1]]>m[C[1][0],C[1][1]]:
                maxlist.append(C[0])
                minlist.append(C[1])
            else:
                maxlist.append(C[1])
                minlist.append(C[0])
        else:
            #maxlist.append[C[0]]
            print('暂不支持点数据')
    s=len(C)
    for n in range(len(maxlist)):
        #计算点与点之间的距离矩阵
        D=np.zeros((s,s),int)
        for x in range(s):
            for y in range(s):
                D[x,y]=(C[x][0]-C[y][0])**2+(C[x][1]-C[y][1])
    #选出距离最长的两点连线
    
    
    for n in range(len(S)):
        Min=99999        
        Max=0
        for x in range(L):
            for y in range(W):
                if fenleisection[x,y]==S[n]:
                    if section[x,y]<Min:
                        Min=section[x,y]
                        minx=x
                        miny=y
                    if section[x,y]>Max:
                        Min=section[x,y]
                        maxx=x
                        maxy=y
        minlist.append((minx,miny))
        maxlist.append((maxx,maxy))
    print minlist
    print maxlist
    #勾连最小与最小，最大与最大
    
    
    
def faultgenerate(m,value):
    bottom,top=twodfaultsurface(m,value)
    print('done')
'''    
def imgloader(kfc):
    L=kfc.shape[0]
    W=kfc.shape[1]
    x_data=[]
    y_data=[]
    z_data=[]
    count=0
    for x in range(L):
        for y in range(W):
            if kfc[x,y]!=-1:#当非空值时
                x_data.append(float(x))
                y_data.append(float(y))
                z_data.append(kfc[x,y])
                count=count+1
    A=L
    B=W
    C=max(z_data)-min(z_data)
    if C==0:
       C=1
    H=max(z_data)
    x_data=np.array(x_data).astype(np.float32)
    x_data=x_data.reshape(1,count)
    y_data=np.array(y_data).astype(np.float32)
    y_data=y_data.reshape(1,count)
    z_data=np.array(z_data).astype(np.float32)
    z_data=z_data.reshape(1,count)

    #print x_data,y_data,z_data
    return x_data,y_data,z_data,A,B,C,H
    
def faultnihetrain(x_data,y_data,z_data,epoch):
    #构造线性模型
    b=tf.Variable(tf.zeros([1]))
    w=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w2=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    z=tf.matmul(w,x_data)+tf.matmul(w2,y_data)+b
    #最小化方差
    loss=tf.reduce_mean(tf.square(z-z_data))
    optimizer=tf.train.GradientDescentOptimizer(0.5)
    train=optimizer.minimize(loss)
    #初始化变量
    init=tf.initialize_all_variables()
    #启动图
    sess=tf.Session()
    sess.run(init)
    #拟合平面
    print ("拟合结果：")
    for step in range(0,epoch+1):
        sess.run(train)
        if step%20==0:
            print( "step:",step,"w:",sess.run(w),"w2:",sess.run(w2),"b:",sess.run(b))
            W1=sess.run(w)
            W2=sess.run(w2)
            B=sess.run(b)
    return W1,W2,b

def faultnihetrain2(x_data,y_data,z_data,epoch):
    #构造线性模型
    b=tf.Variable(tf.zeros([1]))
    w=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w2=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w3=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w4=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w5=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
   
    z=tf.matmul(w,x_data)+tf.matmul(w2,y_data)+tf.matmul(w3,x_data**2)+tf.matmul(w4,y_data**2)+tf.matmul(w5,x_data*y_data)+b
    #最小化方差
    loss=tf.reduce_mean(tf.square(z-z_data))
    optimizer=tf.train.GradientDescentOptimizer(0.2)
    train=optimizer.minimize(loss)
    #初始化变量
    init=tf.initialize_all_variables()
    #启动图
    sess=tf.Session()
    sess.run(init)
    #拟合平面
    print("拟合结果：")
    for step in range(0,epoch+1):
        sess.run(train)
        print (sess.run(loss))
        if step%10==0:
            print ("step:",step,"w:",sess.run(w),"w2:",sess.run(w2),"w3:",sess.run(w3),"w4:",sess.run(w5),"w5:",sess.run(w5),"b:",sess.run(b))
        if step==epoch:    
            W1=sess.run(w)
            if np.isnan(W1[0][0]):
                W1=0
            else:
                W1=W1[0][0]
            W2=sess.run(w2)
            if np.isnan(W2[0][0]):
                W2=0
            else:
                W2=W2[0][0]
            W3=sess.run(w3)
            if np.isnan(W3[0][0]):
                W3=0
            else:
                W3=W3[0][0]
            W4=sess.run(w4)
            if np.isnan(W4[0][0]):
                W4=0
            else:
                W4=W4[0][0]
            W5=sess.run(w5)
            if np.isnan(W5[0][0]):
                W5=0
            else:
                W5=W5[0][0]
            B=sess.run(b)
            if np.isnan(B[0]):
                B=0
            else:
                B=B[0]
        
            print (W1,W2,W3,W4,W5,B)
    return W1,W2,W3,W4,W5,B

def faultnihetrain3(x_data,y_data,z_data,epoch):
    #构造线性模型
    b=tf.Variable(tf.zeros([1]))
    w=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w2=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w3=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w4=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w5=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w6=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w7=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w8=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w9=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    z=tf.matmul(w,x_data)+tf.matmul(w2,y_data)+tf.matmul(w3,x_data**2)+tf.matmul(w4,y_data**2)+tf.matmul(w5,x_data*y_data)+b+tf.matmul(w6,(x_data**2)*y_data)+tf.matmul(w7,x_data*(y_data**2))+tf.matmul(w8,x_data**3)+tf.matmul(w9,y_data**3)

    #最小化方差
    loss=tf.reduce_mean(tf.square(z-z_data))
    optimizer=tf.train.GradientDescentOptimizer(0.03)
    train=optimizer.minimize(loss)
    #初始化变量
    init=tf.initialize_all_variables()
    #启动图
    sess=tf.Session()
    sess.run(init)
    #拟合平面
    print("拟合结果：")
    for step in range(0,epoch+1):
        sess.run(train)
        #print sess.run(loss)
        if step%1000==0:
            print ("step:",step,"w:",sess.run(w),"w2:",sess.run(w2),"w3:",sess.run(w3),"w4:",sess.run(w5),"w5:",sess.run(w5),"b:",sess.run(b))
            print ("w6:",sess.run(w6),"w7:",sess.run(w7),"w8:",sess.run(w8),"w9:",sess.run(w9))

        if step==epoch:    
            W1=sess.run(w)
            if np.isnan(W1[0][0]):
                W1=0
            else:
                W1=W1[0][0]
            W2=sess.run(w2)
            if np.isnan(W2[0][0]):
                W2=0
            else:
                W2=W2[0][0]
            W3=sess.run(w3)
            if np.isnan(W3[0][0]):
                W3=0
            else:
                W3=W3[0][0]
            W4=sess.run(w4)
            if np.isnan(W4[0][0]):
                W4=0
            else:
                W4=W4[0][0]
            W5=sess.run(w5)
            if np.isnan(W5[0][0]):
                W5=0
            else:
                W5=W5[0][0]
            W6=sess.run(w6)
            if np.isnan(W6[0][0]):
                W6=0
            else:
                W6=W6[0][0]
            W7=sess.run(w7)
            if np.isnan(W7[0][0]):
                W7=0
            else:
                W7=W7[0][0]
            W8=sess.run(w8)
            if np.isnan(W8[0][0]):
                W8=0
            else:
                W8=W8[0][0]
            W9=sess.run(w9)
            if np.isnan(W9[0][0]):
                W9=0
            else:
                W9=W9[0][0]
            B=sess.run(b)
            if np.isnan(B[0]):
                B=0
            else:
                B=B[0]
            
            print (W1,W2,W3,W4,W5,W6,W7,W8,W9,B)
    return W1,W2,W3,W4,W5,W6,W7,W8,W9,B

def faultnihetrain4(x_data,y_data,z_data,epoch):
    #构造线性模型
    b=tf.Variable(tf.zeros([1]))
    w=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w2=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w3=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w4=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w5=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w6=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w7=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w8=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w9=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w10=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w11=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w12=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w13=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w14=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    z=tf.matmul(w,x_data)+tf.matmul(w2,y_data)+tf.matmul(w3,x_data**2)+tf.matmul(w4,y_data**2)+tf.matmul(w5,x_data*y_data)+b+tf.matmul(w6,(x_data**2)*y_data)+tf.matmul(w7,x_data*(y_data**2))+tf.matmul(w8,x_data**3)+tf.matmul(w9,y_data**3)+tf.matmul(w10,y_data**4)+tf.matmul(w11,x_data**4)+tf.matmul(w12,x_data*(y_data**3))+tf.matmul(w13,y_data*(x_data**3))+tf.matmul(w14,(y_data**2)*(x_data**2))

    #最小化方差
    loss=tf.reduce_mean(tf.square(z-z_data))
    optimizer=tf.train.GradientDescentOptimizer(0.08)
    train=optimizer.minimize(loss)
    #初始化变量
    init=tf.initialize_all_variables()
    #启动图
    sess=tf.Session()
    sess.run(init)
    #拟合平面
    print ("拟合结果：")
    for step in range(0,epoch+1):
        sess.run(train)
        #print sess.run(loss)
        if step%1000==0:
            print ("step:",step,"w:",sess.run(w),"w2:",sess.run(w2),"w3:",sess.run(w3),"w4:",sess.run(w5),"w5:",sess.run(w5),"b:",sess.run(b))
            print ("w6:",sess.run(w6),"w7:",sess.run(w7),"w8:",sess.run(w8),"w9:",sess.run(w9))
            print ("w10:",sess.run(w10),"w11:",sess.run(w11),"w12:",sess.run(w12),"w13:",sess.run(w13),"w14:",sess.run(w14))
            print (sess.run(loss))
        if step==epoch:    
            W1=sess.run(w)
            if np.isnan(W1[0][0]):
                W1=0
            else:
                W1=W1[0][0]
            W2=sess.run(w2)
            if np.isnan(W2[0][0]):
                W2=0
            else:
                W2=W2[0][0]
            W3=sess.run(w3)
            if np.isnan(W3[0][0]):
                W3=0
            else:
                W3=W3[0][0]
            W4=sess.run(w4)
            if np.isnan(W4[0][0]):
                W4=0
            else:
                W4=W4[0][0]
            W5=sess.run(w5)
            if np.isnan(W5[0][0]):
                W5=0
            else:
                W5=W5[0][0]
            W6=sess.run(w6)
            if np.isnan(W6[0][0]):
                W6=0
            else:
                W6=W6[0][0]
            W7=sess.run(w7)
            if np.isnan(W7[0][0]):
                W7=0
            else:
                W7=W7[0][0]
            W8=sess.run(w8)
            if np.isnan(W8[0][0]):
                W8=0
            else:
                W8=W8[0][0]
            W9=sess.run(w9)
            if np.isnan(W9[0][0]):
                W9=0
            else:
                W9=W9[0][0]
            B=sess.run(b)
            if np.isnan(B[0]):
                B=0
            else:
                B=B[0]
            W10=sess.run(w10)
            if np.isnan(W10[0][0]):
                W10=0
            else:
                W10=W10[0][0]
            W11=sess.run(w11)
            if np.isnan(W11[0][0]):
                W11=0
            else:
                W11=W11[0][0]
            W12=sess.run(w12)
            if np.isnan(W12[0][0]):
                W12=0
            else:
                W12=W12[0][0]
            W13=sess.run(w13)
            if np.isnan(W13[0][0]):
                W13=0
            else:
                W13=W13[0][0]
            W14=sess.run(w14)
            if np.isnan(W14[0][0]):
                W14=0
            else:
                W14=W14[0][0]
            print (W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,B)
    return W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,B

def faultnihetrain5(x_data,y_data,z_data,epoch):
    #构造线性模型
    b=tf.Variable(tf.zeros([1]))
    w=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w2=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w3=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w4=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w5=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w6=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w7=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w8=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w9=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w10=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w11=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w12=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w13=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w14=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w15=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w16=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w17=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w18=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w19=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w20=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))

    z=tf.matmul(w,x_data)+tf.matmul(w2,y_data)+tf.matmul(w3,x_data**2)+tf.matmul(w4,y_data**2)+tf.matmul(w5,x_data*y_data)+b+tf.matmul(w6,(x_data**2)*y_data)+tf.matmul(w7,x_data*(y_data**2))+tf.matmul(w8,x_data**3)+tf.matmul(w9,y_data**3)+tf.matmul(w10,y_data**4)+tf.matmul(w11,x_data**4)+tf.matmul(w12,x_data*(y_data**3))+tf.matmul(w13,y_data*(x_data**3))+tf.matmul(w14,(y_data**2)*(x_data**2))+tf.matmul(w15,x_data**5)+tf.matmul(w16,y_data**5)+tf.matmul(w17,y_data*(x_data**4))+tf.matmul(w18,x_data*(y_data**4))+tf.matmul(w19,(y_data**2)*(x_data**3))+tf.matmul(w20,(y_data**3)*(x_data**2))

    #最小化方差
    loss=tf.reduce_mean(tf.square(z-z_data))
    optimizer=tf.train.GradientDescentOptimizer(0.08)
    train=optimizer.minimize(loss)
    #初始化变量
    init=tf.initialize_all_variables()
    #启动图
    sess=tf.Session()
    sess.run(init)
    #拟合平面
    print ("拟合结果：")
    for step in range(0,epoch+1):
        sess.run(train)
        #print sess.run(loss)
        if step%1000==0:
            print ("step:",step,"w:",sess.run(w),"w2:",sess.run(w2),"w3:",sess.run(w3),"w4:",sess.run(w5),"w5:",sess.run(w5),"b:",sess.run(b))
            print ("w6:",sess.run(w6),"w7:",sess.run(w7),"w8:",sess.run(w8),"w9:",sess.run(w9))
            print ("w10:",sess.run(w10),"w11:",sess.run(w11),"w12:",sess.run(w12),"w13:",sess.run(w13),"w14:",sess.run(w14))
            print ("w15:",sess.run(w15),"w16:",sess.run(w16),"w17:",sess.run(w17),"w18:",sess.run(w18),"w19:",sess.run(w19),"w20:",sess.run(w20))
            print ("loss:",sess.run(loss))

        if step==epoch:    
            W1=sess.run(w)
            if np.isnan(W1[0][0]):
                W1=0
            else:
                W1=W1[0][0]
            W2=sess.run(w2)
            if np.isnan(W2[0][0]):
                W2=0
            else:
                W2=W2[0][0]
            W3=sess.run(w3)
            if np.isnan(W3[0][0]):
                W3=0
            else:
                W3=W3[0][0]
            W4=sess.run(w4)
            if np.isnan(W4[0][0]):
                W4=0
            else:
                W4=W4[0][0]
            W5=sess.run(w5)
            if np.isnan(W5[0][0]):
                W5=0
            else:
                W5=W5[0][0]
            W6=sess.run(w6)
            if np.isnan(W6[0][0]):
                W6=0
            else:
                W6=W6[0][0]
            W7=sess.run(w7)
            if np.isnan(W7[0][0]):
                W7=0
            else:
                W7=W7[0][0]
            W8=sess.run(w8)
            if np.isnan(W8[0][0]):
                W8=0
            else:
                W8=W8[0][0]
            W9=sess.run(w9)
            if np.isnan(W9[0][0]):
                W9=0
            else:
                W9=W9[0][0]
            B=sess.run(b)
            if np.isnan(B[0]):
                B=0
            else:
                B=B[0]
            W10=sess.run(w10)
            if np.isnan(W10[0][0]):
                W10=0
            else:
                W10=W10[0][0]
            W11=sess.run(w11)
            if np.isnan(W11[0][0]):
                W11=0
            else:
                W11=W11[0][0]
            W12=sess.run(w12)
            if np.isnan(W12[0][0]):
                W12=0
            else:
                W12=W12[0][0]
            W13=sess.run(w13)
            if np.isnan(W13[0][0]):
                W13=0
            else:
                W13=W13[0][0]
            W14=sess.run(w14)
            if np.isnan(W14[0][0]):
                W14=0
            else:
                W14=W14[0][0]
            W15=sess.run(w15)
            if np.isnan(W15[0][0]):
                W15=0
            else:
                W15=W15[0][0]
            W16=sess.run(w16)
            if np.isnan(W16[0][0]):
                W16=0
            else:
                W16=W16[0][0]
            W17=sess.run(w17)
            if np.isnan(W17[0][0]):
                W17=0
            else:
                W17=W17[0][0]
            W18=sess.run(w18)
            if np.isnan(W18[0][0]):
                W18=0
            else:
                W18=W18[0][0]
            W19=sess.run(w19)
            if np.isnan(W19[0][0]):
                W19=0
            else:
                W19=W19[0][0]
            W20=sess.run(w20)
            if np.isnan(W20[0][0]):
                W20=0
            else:
                W20=W20[0][0]
            print (W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,B)
    return W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,B
def extend22dAI(m,x1,y1):#9格子内随机选取一个值  2d version
    listcs=[]
    for ss2 in range(-1,2):
        for ss3 in range(-1,2):
            c=m[x1+ss2,y1+ss3]
            if c!=-1:#默认空值为1
                listcs.append(c)

    if len(listcs)>=2:
    #if len(listcs)!=0:
        value= max_list(listcs)
    else:
        value=-1
    return value

def extendTimodel2d(m,template_x,template_y):#全自动拓展插入硬数据的待模拟网格2d version
    lag=max(template_x,template_y)//2
    m2=np.pad(m,lag,'edge')
    d=[]
    for x in range(lag,m2.shape[0]-lag):
        for y in range(lag,m2.shape[1]-lag):
            d.append((x,y))
    
    for cc in range(lag):
        #random.shuffle(d)
        flag=0
        for n in range(len(d)):
            x=d[n][0]
            y=d[n][1]
            if m2[x,y]==-1:
                value=extend22dAI(m2,x,y)
                flag=1
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

def rebuild(W1,W2,W3,W4,W5,b,m,A,B,C): #ABC为归一化系数  平方拟合
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            if m[x,y]==-1:
                m[x,y]=(W1*x/A+W2*y/B+W3*(x**2)/(A**2)+W4*(y**2)/(B**2)+W5*x*y/(A*B)+b)*C
                #print  m[x,y]
    return m

def rebuild2(W1,W2,W3,W4,W5,W6,W7,W8,W9,b,m,A,B,C): #ABC为归一化系数  3次方拟合
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            if m[x,y]==-1:
                m[x,y]=(W1*x/A+W2*y/B+W3*(x**2)/(A**2)+W4*(y**2)/(B**2)+W5*x*y/(A*B)+W6*(x**2)*y/(A*A*B)+W7*x*(y**2)/(B*B*A)+W8*(x**3)/(A**3)+W9*(y**3)/(B**3)+b)*C
                #print  m[x,y]
    return m

def rebuild3(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b,m,A,B,C): #ABC为归一化系数 4次方拟合
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            if m[x,y]==-1:
                m[x,y]=(W1*x/A+W2*y/B+W3*(x**2)/(A**2)+W4*(y**2)/(B**2)+W5*x*y/(A*B)+W6*(x**2)*y/(A*A*B)+W7*x*(y**2)/(B*B*A)+W8*(x**3)/(A**3)+W9*(y**3)/(B**3)+b+W10*(y**4)/(B**4)+W11*(y**4)/(A**4)+W12*x*(y**3)/(A*(B**3))+W13*y*(x**3)/(B*(A**3))+W14*(x**2)*(y**2)/(A*A*B*B))*C
                #print  m[x,y]
    return m

def rebuild4(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b,m,A,B,C): #ABC为归一化系数 4次方拟合
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            if m[x,y]==-1:
                m[x,y]=(W1*x/A+W2*y/B+W3*(x**2)/(A**2)+W4*(y**2)/(B**2)+W5*x*y/(A*B)+W6*(x**2)*y/(A*A*B)+W7*x*(y**2)/(B*B*A)+W8*(x**3)/(A**3)+W9*(y**3)/(B**3)+b+W10*(y**4)/(B**4)+W11*(y**4)/(A**4)+W12*x*(y**3)/(A*(B**3))+W13*y*(x**3)/(B*(A**3))+W14*(x**2)*(y**2)/(A*A*B*B))*C+W15*(x**5)/(A**5)+W16*(y**5)/(B**5)+W17*y*(x**4)/(B*(A**4))+W18*x*(y**4)/(A*(B**4))+W19*(y**2)*(x**3)/((B**2)*(A**3))+W20*(x**2)*(y**3)/((A**2)*(B**3))
                #print  m[x,y]
    return m

def buildfault(m,value,epoch):#重构地层体
    
    bottom,top=twodfaultsurface(m,value)
    mp=-np.ones_like(m)
    x_data,y_data,z_data,A,B,C,H=imgloader(top)
    #tap=max(z_data)
    x_data=x_data/A
    y_data=y_data/B
    z_data=z_data/C
    #W1,W2,W3,W4,W5,b=faultnihetrain2(x_data,y_data,z_data,epoch)
    W1,W2,W3,W4,W5,W6,W7,W8,W9,b=faultnihetrain3(x_data,y_data,z_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b=faultnihetrain4(x_data,y_data,z_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b=faultnihetrain5(x_data,y_data,z_data,epoch)
    #top=rebuild(W1,W2,W3,W4,W5,b,top,A,B,C)
    top=rebuild2(W1,W2,W3,W4,W5,W6,W7,W8,W9,b,top,A,B,C)
    #top=rebuild3(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b,top,A,B,C)
    #top=rebuild4(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b,top,A,B,C)

    if value!=0 and (W1==0) and (W2==0) and (W3==0) and (W4==0) and (W5==0) and (W6==0) and (W7==0) and (W8==0) and (W9==0) and (b==0):
      top=(m.shape[2]-1)*np.ones((m.shape[1],m.shape[2]),int)

    x2_data,y2_data,z2_data,A2,B2,C2,H2=imgloader(bottom)
    x2_data=x2_data/A2
    y2_data=y2_data/B2
    z2_data=z2_data/C2
    #W1,W2,W3,W4,W5,b=faultnihetrain2(x2_data,y2_data,z2_data,epoch)
    W1,W2,W3,W4,W5,W6,W7,W8,W9,b=faultnihetrain3(x2_data,y2_data,z2_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b=faultnihetrain4(x2_data,y2_data,z2_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b=faultnihetrain5(x2_data,y2_data,z2_data,epoch)
    #bottom=rebuild(W1,W2,W3,W4,W5,b,bottom,A2,B2,C2)
    bottom=rebuild2(W1,W2,W3,W4,W5,W6,W7,W8,W9,b,bottom,A2,B2,C2)
    #bottom=rebuild3(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b,bottom,A2,B2,C2)
    #bottom=rebuild4(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b,bottom,A2,B2,C2)
    '''
    if value!=0 and (W1==0) and (W2==0):
       bottom=(m.shape[2]-1)*np.ones((m.shape[1],m.shape[2]),int)
    '''
    for x in range(m.shape[1]):
        for y in range(m.shape[2]):
            if top[x,y]>bottom[x,y]:
                for h in range(max(0,bottom[x,y]-1),min(H,top[x,y]+1)):
                    mp[h,x,y]=value        
            #else:
                #print('出现地层反转')
                #加入厚度判断
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if mp[h,x,y]!=-1:
                   if m[h,x,y]==-1:
                      m[h,x,y]=value

    return mp,m

def buildfault2(m,value,epoch):#重构地层体 不归一化
    
    bottom,top=twodfaultsurface(m,value)
    mp=-np.ones_like(m)
    x_data,y_data,z_data,A,B,C,H=imgloader(top)
    #tap=max(z_data)
    A=1
    B=1
    C=1
    x_data=x_data/A
    y_data=y_data/B
    z_data=z_data/C
    #W1,W2,W3,W4,W5,b=faultnihetrain2(x_data,y_data,z_data,epoch)
    W1,W2,W3,W4,W5,W6,W7,W8,W9,b=faultnihetrain3(x_data,y_data,z_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b=faultnihetrain4(x_data,y_data,z_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b=faultnihetrain5(x_data,y_data,z_data,epoch)
    #top=rebuild(W1,W2,W3,W4,W5,b,top,A,B,C)
    top=rebuild2(W1,W2,W3,W4,W5,W6,W7,W8,W9,b,top,A,B,C)
    #top=rebuild3(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b,top,A,B,C)
    #top=rebuild4(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b,top,A,B,C)

    if value!=0 and (W1==0) and (W2==0) and (W3==0) and (W4==0) and (W5==0) and (W6==0) and (W7==0) and (W8==0) and (W9==0) and (b==0):
      top=(m.shape[2]-1)*np.ones((m.shape[1],m.shape[2]),int)

    x2_data,y2_data,z2_data,A2,B2,C2,H2=imgloader(bottom)
    A2=1
    B2=1
    C2=1
    x2_data=x2_data/A2
    y2_data=y2_data/B2
    z2_data=z2_data/C2
    #W1,W2,W3,W4,W5,b=faultnihetrain2(x2_data,y2_data,z2_data,epoch)
    W1,W2,W3,W4,W5,W6,W7,W8,W9,b=faultnihetrain3(x2_data,y2_data,z2_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b=faultnihetrain4(x2_data,y2_data,z2_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b=faultnihetrain5(x2_data,y2_data,z2_data,epoch)
    #bottom=rebuild(W1,W2,W3,W4,W5,b,bottom,A2,B2,C2)
    bottom=rebuild2(W1,W2,W3,W4,W5,W6,W7,W8,W9,b,bottom,A2,B2,C2)
    #bottom=rebuild3(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b,bottom,A2,B2,C2)
    #bottom=rebuild4(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b,bottom,A2,B2,C2)
    '''
    if value!=0 and (W1==0) and (W2==0):
       bottom=(m.shape[2]-1)*np.ones((m.shape[1],m.shape[2]),int)
    '''
    for x in range(m.shape[1]):
        for y in range(m.shape[2]):
            if top[x,y]>bottom[x,y]:
                for h in range(max(0,bottom[x,y]-1),min(H,top[x,y]+1)):
                    mp[h,x,y]=value        
            #else:
                #print('出现地层反转')
                #加入厚度判断
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if mp[h,x,y]!=-1:
                   if m[h,x,y]==-1:
                      m[h,x,y]=value

    return mp,m



def extractlist(Ti,listvalue):#提取训练图像中所有值的类型
    for x in range(Ti.shape[0]):
        for y in range(Ti.shape[1]):
            if Ti[x,y] not in listvalue:
               listvalue.append(Ti[x,y])
    return listvalue



def Tilistvalueextract():#自动提取训练图像中所有值类型
    file1=open('./Ti/Tiparameter.txt')
    content=file1.readline()
    string1=[i for i in content if str.isdigit(i)]
    num=int(''.join(string1))
    print('剖面数目：')
    print (num)
    valuelist=[]
    for n in range(num):
        path='./Ti/'+str(n+1)+'.bmp'
        section=cv2.imread(path,0)
        valuelist=extractlist(section,valuelist)
    print( valuelist)
    return valuelist







#######################################################################################
def listforweight():
    lista=[]
    for i in it.product(range(2),repeat=9):
        lista.append(i)
    return lista

def faultnihetrainC(x_data,y_data,z_data,suiji,epoch):
    #构造线性模型
    b=tf.Variable(tf.zeros([1]))
    w=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w2=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w3=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w4=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w5=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w6=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w7=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w8=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    w9=tf.Variable(tf.random_uniform([1,1],-1.0,1.0))
    z=tf.matmul(w*suiji[0],x_data)+tf.matmul(w2*suiji[1],y_data)+tf.matmul(w3*suiji[2],x_data**2)+tf.matmul(w4*suiji[3],y_data**2)+tf.matmul(w5*suiji[4],x_data*y_data)+tf.matmul(w6*suiji[5],(x_data**2)*y_data)+tf.matmul(w7*suiji[6],x_data*(y_data**2))+tf.matmul(w8*suiji[7],x_data**3)+tf.matmul(w9*suiji[8],y_data**3)+b
    
    #最小化方差
    loss=tf.reduce_mean(tf.square(z-z_data))
    optimizer=tf.train.GradientDescentOptimizer(0.03)
    train=optimizer.minimize(loss)
    #初始化变量
    init=tf.initialize_all_variables()
    #启动图
    sess=tf.Session()
    sess.run(init)
    #拟合平面
    print("拟合结果：")
    for step in range(0,epoch+1):
        sess.run(train)
        #print sess.run(loss)
        if step%1000==0:
            print ("step:",step,"w:",sess.run(w),"w2:",sess.run(w2),"w3:",sess.run(w3),"w4:",sess.run(w5),"w5:",sess.run(w5),"b:",sess.run(b))
            print ("w6:",sess.run(w6),"w7:",sess.run(w7),"w8:",sess.run(w8),"w9:",sess.run(w9))

        if step==epoch:    
            W1=sess.run(w)
            if np.isnan(W1[0][0]):
                W1=0
            else:
                W1=W1[0][0]
            W2=sess.run(w2)
            if np.isnan(W2[0][0]):
                W2=0
            else:
                W2=W2[0][0]
            W3=sess.run(w3)
            if np.isnan(W3[0][0]):
                W3=0
            else:
                W3=W3[0][0]
            W4=sess.run(w4)
            if np.isnan(W4[0][0]):
                W4=0
            else:
                W4=W4[0][0]
            W5=sess.run(w5)
            if np.isnan(W5[0][0]):
                W5=0
            else:
                W5=W5[0][0]
            W6=sess.run(w6)
            if np.isnan(W6[0][0]):
                W6=0
            else:
                W6=W6[0][0]
            W7=sess.run(w7)
            if np.isnan(W7[0][0]):
                W7=0
            else:
                W7=W7[0][0]
            W8=sess.run(w8)
            if np.isnan(W8[0][0]):
                W8=0
            else:
                W8=W8[0][0]
            W9=sess.run(w9)
            if np.isnan(W9[0][0]):
                W9=0
            else:
                W9=W9[0][0]
            B=sess.run(b)
            if np.isnan(B[0]):
                B=0
            else:
                B=B[0]
            lost=sess.run(loss)
            print (W1,W2,W3,W4,W5,W6,W7,W8,W9,B)
    return W1,W2,W3,W4,W5,W6,W7,W8,W9,B,lost

def buildfaultall(m,value,epoch):#重构地层体,随机选取参数版
    
    bottom,top=twodfaultsurface(m,value)
    mp=-np.ones_like(m)
    x_data,y_data,z_data,A,B,C,H=imgloader(top)
    #tap=max(z_data)
    x_data=x_data/A
    y_data=y_data/B
    z_data=z_data/C
    suijican=listforweight()
    print(len(suijican))
    #W1,W2,W3,W4,W5,b=faultnihetrain2(x_data,y_data,z_data,epoch)
    canshulist=[]
    lostlist=[]
    for ava in range(len(suijican)):
        W1,W2,W3,W4,W5,W6,W7,W8,W9,b,loss=faultnihetrainC(x_data,y_data,z_data,suijican[ava],epoch)
        canshulist.append([W1,W2,W3,W4,W5,W6,W7,W8,W9,b])
        lostlist.append(loss)
    #获取参数最小的lost的编号
    nova=lostlist.index(min(lostlist))
    W1=canshulist[nova][0]
    W2=canshulist[nova][1]
    W3=canshulist[nova][2]
    W4=canshulist[nova][3]
    W5=canshulist[nova][4]
    W6=canshulist[nova][5]
    W7=canshulist[nova][6]
    W8=canshulist[nova][7]
    W9=canshulist[nova][8]
    b=canshulist[nova][9]
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b=faultnihetrain4(x_data,y_data,z_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b=faultnihetrain5(x_data,y_data,z_data,epoch)
    #top=rebuild(W1,W2,W3,W4,W5,b,top,A,B,C)
    top=rebuild2(W1,W2,W3,W4,W5,W6,W7,W8,W9,b,top,A,B,C)
    #top=rebuild3(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b,top,A,B,C)
    #top=rebuild4(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b,top,A,B,C)

    if value!=0 and (W1==0) and (W2==0) and (W3==0) and (W4==0) and (W5==0) and (W6==0) and (W7==0) and (W8==0) and (W9==0) and (b==0):
        top=(m.shape[2]-1)*np.ones((m.shape[1],m.shape[2]),int)

    
    
    x2_data,y2_data,z2_data,A2,B2,C2,H2=imgloader(bottom)
    x2_data=x2_data/A2
    y2_data=y2_data/B2
    z2_data=z2_data/C2
    
    
    
    
    #W1,W2,W3,W4,W5,b=faultnihetrain2(x2_data,y2_data,z2_data,epoch)
    canshulist=[]
    lostlist=[]
    for ava in range(len(suijican)):
        W1,W2,W3,W4,W5,W6,W7,W8,W9,b,loss=faultnihetrainC(x2_data,y2_data,z2_data,suijican[ava],epoch)
        canshulist.append([W1,W2,W3,W4,W5,W6,W7,W8,W9,b])
        lostlist.append(loss)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b=faultnihetrain4(x2_data,y2_data,z2_data,epoch)
    #W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b=faultnihetrain5(x2_data,y2_data,z2_data,epoch)
    #bottom=rebuild(W1,W2,W3,W4,W5,b,bottom,A2,B2,C2)
    nova=lostlist.index(min(lostlist))
    W1=canshulist[nova][0]
    W2=canshulist[nova][1]
    W3=canshulist[nova][2]
    W4=canshulist[nova][3]
    W5=canshulist[nova][4]
    W6=canshulist[nova][5]
    W7=canshulist[nova][6]
    W8=canshulist[nova][7]
    W9=canshulist[nova][8]
    b=canshulist[nova][9]
    bottom=rebuild2(W1,W2,W3,W4,W5,W6,W7,W8,W9,b,bottom,A2,B2,C2)
    #bottom=rebuild3(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,b,bottom,A2,B2,C2)
    #bottom=rebuild4(W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W20,b,bottom,A2,B2,C2)
    '''
    if value!=0 and (W1==0) and (W2==0):
       bottom=(m.shape[2]-1)*np.ones((m.shape[1],m.shape[2]),int)
    '''
    for x in range(m.shape[1]):
        for y in range(m.shape[2]):
            if top[x,y]>bottom[x,y]:
                for h in range(max(0,bottom[x,y]-1),min(H,top[x,y]+1)):
                    mp[h,x,y]=value        
            #else:
                #print('出现地层反转')
                #加入厚度判断
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if mp[h,x,y]!=-1:
                    if m[h,x,y]==-1:
                        m[h,x,y]=value

    return mp,m



##########################################################
def imgloaderforkeras(kfc):
    L=kfc.shape[0]
    W=kfc.shape[1]
    x_data=[]
    y_data=[]
    z_data=[]
    inputdata=[]
    count=0
    xcount=0
    for x in range(L):
        for y in range(W):
            if kfc[x,y]!=-1:#当非空值时
                x_data.append([float(x),float(y)])
                y_data.append(kfc[x,y])
                count=count+1
            else:
                inputdata.append([float(x),float(y)])
                xcount=xcount+1
    x_data=np.array(x_data).astype(np.float32)
    x_data=x_data.reshape(count,2)
    inputdata=np.array(inputdata).astype(np.float32)
    inputdata=inputdata.reshape(xcount,2)
    y_data=np.array(y_data).astype(np.float32)
    y_data=y_data.reshape(count,1)
    '''
    A=L
    B=W
    C=max(z_data)-min(z_data)
    if C==0:
        C=1
    H=max(z_data)
    x_data=np.array(x_data).astype(np.float32)
    x_data=x_data.reshape(1,count)
    y_data=np.array(y_data).astype(np.float32)
    y_data=y_data.reshape(1,count)
    z_data=np.array(z_data).astype(np.float32)
    z_data=z_data.reshape(1,count)
    '''
    #print x_data,y_data,z_data
    return x_data,y_data,inputdata


class earlystopcc(keras.callbacks.Callback):
      def __init__(self,monitor='loss',value=0.00001,verbose=0):
          super(keras.callbacks.Callback,self).__init__()
          self.monitor=monitor
          self.value=value
          self.verbose=verbose
      def on_epoch_end(self,epoch,logs={}):
          current=logs.get(self.monitor)
          if current is None:
             warning.warn("Early stop requires %s availabel!"%self.monitor,RuntimeWarning)
          if abs(current)<abs(self.value):

             #if self.verbose>0:
             print("Epoch %05d: early stopping THR"%epoch)
             self.model.stop_training=True

def faultmodelbuild(X,Y,inputdata,epoch):
    model=Sequential()
    rate=0
    model.add(Dense(50,input_dim=2,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(300,activation='relu'))
    model.add(Dropout(rate))



    model.add(Dense(300,activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(rate))


 
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(rate))

    model.add(Dense(1))
    #model.add(Dense(1,activation='linear'))
    model.compile(loss='mse',optimizer='Adamax')

    model.summary()
    Mycallback=[
        earlystopcc(monitor='loss',value=0.00005,verbose=0)#0.00005
        #keras.callbacks.EarlyStopping(monitor='loss',patience=100,verbose=0,mode='auto',min_delta=100)
]

    history=model.fit(X,Y,epochs=epoch,verbose=1,callbacks=Mycallback)
    '''
    #history=model.fit(X,Y,epochs=epoch,verbose=0)
    #keras.callbacks.EarlyStopping(monitor='loss',patience=0,verbose=0,mode='auto',min_delta=1)
    pyplot.plot(history.history['loss'])
    pyplot.show()
    '''
    output=model.predict(inputdata)
    return output

def buildfaultkeras(m,value,epoch):#重构地层体
    
    bottom,top=twodfaultsurface(m,value)
    mp=-np.ones_like(m)
    x_data,y_data,inputdata=imgloaderforkeras(top)
    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    for h in range(len(inputdata)):
        #print(inputdata[h])
        top[int(inputdata[h][0]),int(inputdata[h][1])]=s[h]

    x_data,y_data,inputdata=imgloaderforkeras(bottom)
    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    for h in range(len(inputdata)):
        #print(inputdata[h])
        bottom[int(inputdata[h][0]),int(inputdata[h][1])]=s[h]


    for x in range(m.shape[1]):
        for y in range(m.shape[2]):
            if top[x,y]>bottom[x,y]:
                for h in range(max(0,bottom[x,y]-1),min(m.shape[0],top[x,y]+1)):
                    mp[h,x,y]=value        
            #else:
                #print('出现地层反转')
                #加入厚度判断
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if mp[h,x,y]!=-1:
                    if m[h,x,y]==-1:
                        m[h,x,y]=value

    return mp,m

def buildfaultkerasS(m3,value,epoch,name):#重构地层体
    m=m3.copy()
    bottom,top=twodfaultsurface(m,value)
    mp=-np.ones_like(m)
    x_data,y_data,inputdata=imgloaderforkeras(top)
    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    for h in range(len(inputdata)):
        #print(inputdata[h])
        top[int(inputdata[h][0]),int(inputdata[h][1])]=s[h]

    x_data,y_data,inputdata=imgloaderforkeras(bottom)
    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    for h in range(len(inputdata)):
        #print(inputdata[h])
        bottom[int(inputdata[h][0]),int(inputdata[h][1])]=s[h]


    for x in range(m.shape[1]):
        for y in range(m.shape[2]):
            if top[x,y]>bottom[x,y]:
                for h in range(max(0,bottom[x,y]-1),min(m.shape[0],top[x,y]+1)):
                    mp[h,x,y]=value        
            #else:
                #print('出现地层反转')
                #加入厚度判断
    for h in range(m.shape[0]):
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if mp[h,x,y]!=-1:
                    if m[h,x,y]==-1:
                        m[h,x,y]=value
    path='./database/ini'+str(name)+'.npy'
    np.save(path,m)


def buildfaultkerasM(m,valuelist,epoch):#重构地层体multiprocess
    processes=list()
    for n in range(len(valuelist)):
        print('process')
    
        s=multiprocessing.Process(target=buildfaultkerasS,args=(m,valuelist[n],epoch,n))
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    for n in range(len(valuelist)):
        path='./database/ini'+str(n)+'.npy'
        m2=np.load(path)
        for h in range(m.shape[0]):
            for x in range(m.shape[1]):
                for y in range(m.shape[2]):
                    if m2[h,x,y]==valuelist[n]:
                       m[h,x,y]=m2[h,x,y]

    return m

def buildfaultkerasS2(m,value,epoch,name):#重构地层体
    
    bottom,top=twodfaultsurface(m,value)

    x_data,y_data,inputdata=imgloaderforkeras(top)
    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    for h in range(len(inputdata)):
        #print(inputdata[h])
        top[int(inputdata[h][0]),int(inputdata[h][1])]=s[h]

    x_data,y_data,inputdata=imgloaderforkeras(bottom)
    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    for h in range(len(inputdata)):
        #print(inputdata[h])
        bottom[int(inputdata[h][0]),int(inputdata[h][1])]=s[h]



    path='./database/init'+str(name)+'.npy'
    np.save(path,top)
    path='./database/inib'+str(name)+'.npy'
    np.save(path,bottom)


def buildfaultkerasM2(m,valuelist,epoch):#重构地层体multiprocess
    processes=list()
    for n in range(len(valuelist)):
        print('process')
    
        s=multiprocessing.Process(target=buildfaultkerasS2,args=(m,valuelist[n],epoch,n))
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    for n in range(len(valuelist)):
        path='./database/init'+str(n)+'.npy'
        top=np.load(path)
        path='./database/inib'+str(n)+'.npy'
        bottom=np.load(path)
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                if top[x,y]!=-1:
                   if top[x,y]-5>bottom[x,y]+5:
                      for h in range(max(0,bottom[x,y]-5),min(m.shape[0],top[x,y]+5)):
                          if m[h,x,y]==-1:
                             m[h,x,y]=valuelist[n]

    return m


'''
np.set_printoptions(threshold=np.inf)  
m=np.ones((100,95),int)

c=2    
for x in range(40,95):
    m[0,x]=c
    c=c+1


for x in range(50,95):
    m[99,x]=c
    c=c-1
for x in range(20,70):
    m[x,94]=45

m=extendTimodel2d(m,7,7)
#print m
x_data,y_data,z_data=imgloader(m)
x_data=x_data/100
y_data=y_data/95
z_data=z_data/62
#print x_data,y_data,x_data**2,x_data*y_data
#print x_data.shape
W1,W2,W3,W4,W5,b=faultnihetrain2(x_data,y_data,z_data,2000)
rebuild(W1,W2,W3,W4,W5,b,m)
print m


n = 1000
x, y = np.meshgrid(np.linspace(-100,100, n),
                   np.linspace(-95, 95, n))
z = (W1*x/100+W2*y/95+W3*(x**2)/10000+W4*(y**2)/(95**2)+W5*x*y/9500+b)*62
mp.figure('3D Surface')
ax = mp.gca(projection='3d')
mp.title('3D Surface', fontsize=20)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_zlabel('z', fontsize=14)
mp.tick_params(labelsize=10)
ax.plot_surface(x, y, z, rstride=30,
                cstride=30, cmap='jet')
mp.show()
''' 

'''
m=np.load('./output/wtf.npy')

mp,m=buildfault(m,255,4000)
mp,m=buildfault(m,25,4000)
mp,m=buildfault(m,50,4000)
mp,m=buildfault(m,75,4000)
mp,m=buildfault(m,99,4000)
mp,m=buildfault(m,50,4000)
mp,m=buildfault(m,230,4000)
mp,m=buildfault(m,125,4000)
mp,m=buildfault(m,151,4000)
mp,m=buildfault(m,176,4000)
mp,m=buildfault(m,199,4000)
mp,m=buildfault(m,0,4000)
data=m.transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape) 
grid.point_data.scalars = np.ravel(data,order='F') 
grid.point_data.scalars.name = 'lithology' 
write_data(grid, './output/outputfault.vtk') 
'''
# In[ ]:




