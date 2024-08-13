#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import time
import cv2
from PIL import Image
import threading
import math
import matplotlib.pyplot as plt
from listheap import*
import Levenshtein
from scipy.cluster.vq import *
#from scipy.misc import imresize
import multiprocessing
from pylab import *
import random
import os
import sys
from tvtk.api import tvtk, write_data 
from Patchmatch import *



def reconstructionTTTZ(A,mm,F,Fore,BTilist):
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                for n in range(len(F)):
                    if mm[i,j,k]==-1:
                       if Fore[i,j,k]==n:
                           temp[i,j,k] = BTilist[n][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]]
                    else:
                       temp[i,j,k]=mm[i,j,k]
    #Image.fromarray(temp).show()
    return temp

    
def patchmatchmultiTiBZ(m,mm,Tilist,size,itr,name):#patchmatch做优化 并行版
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
        print (name,n)
        s=multiprocessing.Process(target=NNSB, args=(m,Tilist[n],size,itr,name,n))
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
    Re=reconstructionTTTZ(m,mm,F,Fore,BTilist)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re



def reconstructionTTTZ2(A,mm,F,Fore,BTilist):
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = np.zeros_like(A)
    CTilist=[]#构建模板使用统计TI
    for n in range(len(BTilist)):
        ti = np.zeros_like(BTilist[n])
        CTilist.append(ti)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                for n in range(len(F)):
                    if mm[i,j,k]==-1:
                       if Fore[i,j,k]==n:
                           temp[i,j,k] = BTilist[n][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]]
                           CTilist[n][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]]=CTilist[n][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]]+1
                    else:
                       temp[i,j,k]=mm[i,j,k]
    #Image.fromarray(temp).show()
    return temp,CTilist

    
def patchmatchmultiTiBZ2(m,mm,Tilist,size,itr,name):#patchmatch做优化 并行版,同时返回模板使用统计Ti
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
        s=multiprocessing.Process(target=NNSB, args=(m,Tilist[n],size,itr,name,n))
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
    Re,CTilist=reconstructionTTTZ2(m,mm,F,Fore,BTilist)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re,CTilist

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

def initializationzuobiaover(A, B, Bzuobiao,p_size):#初始化，构建映射
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
    BZuobiaolist=[]
    for n in range(3):
        Y=np.pad(Bzuobiao[n],p,'edge')
        BZuobiaolist.append(Y)
    random_B_r = np.random.randint(p, B_h-p, [A_h, A_l,A_w])#分别储存对应B的坐标
    #print random_B_r
    random_B_c = np.random.randint(p, B_l-p, [A_h, A_l,A_w])
    random_B_v = np.random.randint(p, B_w-p, [A_h, A_l,A_w])
    #A_padding = np.ones([A_h+p*2, A_l+p*2, 1]) * np.nan
    #A_padding[p:A_h+p, p:A_l+p, :] = A
    A_padding = np.ones([A_h+p*2, A_l+p*2,A_w+p*2]) * np.nan
    A_padding[p:A_h+p, p:A_l+p,p:A_w+p] = A

    f = np.zeros([A_h, A_l,A_w], dtype=object)
    dist = np.zeros([A_h, A_l,A_w])
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                a = np.array([i+p, j+p,k+p])
                b = np.array([random_B_r[i, j,k], random_B_c[i, j,k],random_B_v[i, j,k]], dtype=np.int32)
                #print b
                f[i, j,k] = b
                dist[i, j,k] = cal_distance(a, b, A_padding, B, p_size)
    #print f
    return f, dist, A_padding,B,BZuobiaolist
def NNSBzuobiaover(img, ref,zuobiao, p_size, itr,name,idcard):#寻找最近零并行版
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding ,Bref,Brefzuobiao= initializationzuobiaover(img, ref,zuobiao, p_size)
    p=p_size//2
    print("initialization done")
    for itr in range(1, itr+1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_l - 1, -1, -1):
                    for k in range(A_w - 1, -1, -1):
                        a = np.array([i+p, j+p,k+p])
                        propagation(f, a, dist, img_padding, Bref, p_size, False)
                        random_search(f, a, dist, img_padding, Bref, p_size)
        else:
            for i in range(A_h):
                for j in range(A_l):
                    for k in range(A_w):
                        a = np.array([i+p, j+p,k+p])
                        propagation(f, a, dist, img_padding, Bref, p_size, True)
                        random_search(f, a, dist, img_padding, Bref, p_size)
        print("iteration: %d"%(itr))
    path1='./database/patchmatch('+str(name)+')('+str(idcard)+')f.npy'
    path2='./database/patchmatch('+str(name)+')('+str(idcard)+')dist.npy'
    path3='./database/patchmatch('+str(name)+')('+str(idcard)+')Bref.npy'
    path4='./database/patchmatch('+str(name)+')('+str(idcard)+')Brefzuobiao.npy'
    np.save(path1,f)
    np.save(path2,dist)
    np.save(path3,Bref)
    np.save(path4,Brefzuobiao)
    return f,dist,Bref,Brefzuobiao


def projectTTTzuobiaover(m,F,DIST,Tizuobiaolist):
    Fore=np.zeros((m.shape[0], m.shape[1],m.shape[2]),int)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            for k in range(m.shape[2]):
                Dis2=[]
                Dis2acc=0
                dispro2=[]#IDW
                for n in range(len(F)):
                    #print len(F)
                    #print F[n][i,j,k][0], F[n][i,j,k][1],F[n][i,j,k][2],Tizuobiaolist[n][0].shape
                    #print Tizuobiaolist[n][0][F[n][i,j,k][0], F[n][i,j,k][1],F[n][i,j,k][2]]
                    #print Tizuobiaolist[n][1][F[n][i,j,k][0], F[n][i,j,k][1],F[n][i,j,k][2]]
                    #print Tizuobiaolist[n][2][F[n][i,j,k][0], F[n][i,j,k][1],F[n][i,j,k][2]]
                    bigdady=(i-Tizuobiaolist[n][0][F[n][i,j,k][0], F[n][i,j,k][1],F[n][i,j,k][2]])**2+(j-Tizuobiaolist[n][1][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]])**2+(k-Tizuobiaolist[n][2][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]])**2
                    #print bigdady 
                    if bigdady==0:
                       bigdady=1
                    dis2=math.sqrt(bigdady)
                    Dis2.append((1/dis2))
                    Dis2acc=Dis2acc+(1/dis2)
                for n in range(len(F)):
                    dispro2.append((Dis2[n]/Dis2acc))
                #print dispro2
                Fore[i,j,k]=random_index(dispro2)

    return Fore
def patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,name):#patchmatch做优化 并行版,同时返回模板使用统计Ti #IDW版
    #size为模板大小，itr为迭代次数
    Fore = np.zeros([m.shape[0], m.shape[1],m.shape[2]])
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    F=[]
    DIST=[]
    BTilist=[]
    processes=list()
    BTizuobiao=[]
    for n in range(len(Tilist)):
        print('porcess:')
        print (name,n)
        s=multiprocessing.Process(target=NNSBzuobiaover, args=(m,Tilist[n],Tizuobiaolist[n],size,itr,name,n))
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    for n in range(len(Tilist)):
        path1='./database/patchmatch('+str(name)+')('+str(n)+')f.npy'
        path2='./database/patchmatch('+str(name)+')('+str(n)+')dist.npy'
        path3='./database/patchmatch('+str(name)+')('+str(n)+')Bref.npy'
        path4='./database/patchmatch('+str(name)+')('+str(n)+')Brefzuobiao.npy'
        f=np.load(path1)
        dist=np.load(path2)
        Bref=np.load(path3)
        Brefzuobiao=np.load(path4)
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)
        BTizuobiao.append(Brefzuobiao)
    '''
    for n in range(len(Tilist)):
        p=multiprocessing.Process(target=NNS, args=(m,Tilist[n],size,itr))
        print('process start!')
    '''
    print("Searching done!")
    #print F[0],F[1],F[2],F[3]
    Fore=projectTTTzuobiaover(m,F,DIST,BTizuobiao)
    #print Fore
    print("Pick done!")
    Re,CTilist=reconstructionTTTZ2(m,mm,F,Fore,BTilist)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re,CTilist

def projectTTTzuobiaover2(m,F,DIST,Tizuobiaolist):#信息熵权集成版
    Fore=np.zeros((m.shape[0], m.shape[1],m.shape[2]),int)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            for k in range(m.shape[2]):
                score=np.zeros((len(F),2),float)
                for n in range(len(F)):
                    score[n,0]=DIST[n][i,j,k]#记录下距离
                for n in range(len(F)):
                    dis2=math.sqrt(((i-Tizuobiaolist[n][0][F[n][i,j,k][0], F[n][i,j,k][1],F[n][i,j,k][2]])**2)+((j-Tizuobiaolist[n][1][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]])**2)+((k-Tizuobiaolist[n][2][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]])**2))
                    score[n,1]=dis2#记录下欧式距离

                
                Fore[i,j,k]=maxscore(score)#得到得分最高的序列号

    return Fore


#待测试
def patchmatchmultiTiBZzuobiaover2(m,mm,Tilist,Tizuobiaolist,size,itr,name):#patchmatch做优化 并行版,同时返回模板使用统计Ti #信息熵权集成版
    Fore = np.zeros([m.shape[0], m.shape[1],m.shape[2]])
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    F=[]
    DIST=[]
    BTilist=[]
    processes=list()
    BTizuobiao=[]
    for n in range(len(Tilist)):
        print('porcess:')
        print(name,n)
        s=multiprocessing.Process(target=NNSBzuobiaover, args=(m,Tilist[n],Tizuobiaolist[n],size,itr,name,n))
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    for n in range(len(Tilist)):
        path1='./database/patchmatch('+str(name)+')('+str(n)+')f.npy'
        path2='./database/patchmatch('+str(name)+')('+str(n)+')dist.npy'
        path3='./database/patchmatch('+str(name)+')('+str(n)+')Bref.npy'
        path4='./database/patchmatch('+str(name)+')('+str(n)+')Brefzuobiao.npy'
        f=np.load(path1)
        dist=np.load(path2)
        Bref=np.load(path3)
        Brefzuobiao=np.load(path4)
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)
        BTizuobiao.append(Brefzuobiao)
    '''
    for n in range(len(Tilist)):
        p=multiprocessing.Process(target=NNS, args=(m,Tilist[n],size,itr))
        print('process start!')
    '''
    print("Searching done!")
    #print F[0],F[1],F[2],F[3]

    Fore=projectTTTzuobiaover2(m,F,DIST,BTizuobiao)
    #print Fore
    print("Pick done!")
    Re,CTilist=reconstructionTTTZ2(m,mm,F,Fore,BTilist)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re,CTilist

def initialization2(A, mm,B,p_size):#初始化，构建映射
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
    dist = np.zeros([A_h, A_l,A_w])
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                a = np.array([i+p, j+p,k+p])
                b = np.array([random_B_r[i, j,k], random_B_c[i, j,k],random_B_v[i, j,k]], dtype=np.int32)
                #print b


                f[i, j,k] = b
                dist[i, j,k] = cal_distance(a, b, A_padding, B, p_size)
                if mm[i,j,k]!=-1:#偷懒硬数据赋值法，待改进
                   d=np.array([i+p,min(B_l-p-1,j+p),p])
                   c=np.array([i+p,p,min(B_w-p-1,k+p)])
                   dc=cal_distance(a, c, A_padding, B, p_size)
                   marvel=cal_distance(a, d, A_padding, B, p_size)
                   if dc<=marvel:
                      f[i,j,k]=c
                      dist[i,j,k]=dc
                   else:
                      f[i,j,k]=d
                      dist[i,j,k]=marvel
    #print f
    return f, dist, A_padding,B


def NNSB2(img, mm,ref, p_size, itr,name,idcard):#寻找最近零并行版改进
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding ,Bref= initialization2(img,mm, ref, p_size)
    p=p_size//2
    print("initialization done")
    for itr in range(1, itr+1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_l - 1, -1, -1):
                    for k in range(A_w - 1, -1, -1):
                        a = np.array([i+p, j+p,k+p])
                        propagation(f, a, dist, img_padding, Bref, p_size, False)
                        random_search(f, a, dist, img_padding, Bref, p_size)
        else:
            for i in range(A_h):
                for j in range(A_l):
                    for k in range(A_w):
                        a = np.array([i+p, j+p,k+p])
                        propagation(f, a, dist, img_padding, Bref, p_size, True)
                        random_search(f, a, dist, img_padding, Bref, p_size)
        print("iteration: %d"%(itr))
    path1='./database/patchmatch('+str(name)+')('+str(idcard)+')f.npy'
    path2='./database/patchmatch('+str(name)+')('+str(idcard)+')dist.npy'
    path3='./database/patchmatch('+str(name)+')('+str(idcard)+')Bref.npy'
    np.save(path1,f)
    np.save(path2,dist)
    np.save(path3,Bref)
    return f,dist,Bref

def patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,name):#patchmatch做优化 并行版,同时返回模板使用统计Ti
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
        print (name,n)
        s=multiprocessing.Process(target=NNSB2, args=(m,mm,Tilist[n],size,itr,name,n))
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
    Re,CTilist=reconstructionTTTZ2(m,mm,F,Fore,BTilist)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re,CTilist

def apartlist(ls, size):#分割列表工具
    return [ls[i:i+size] for i in range(0, len(ls), size)]

def propagationsub(f, a, dist, A_padding, B, p_size, is_odd):#传播
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
            dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
        if idx == 2:
            f[x, y,z] = f[x, max(y - 1, 0),z]
            dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
        if idx == 3:
            f[x, y,z] = f[x, y,max(z - 1, 0)]
            dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
    else:
        d_right = dist[min(x + 1, A_h-1), y,z]
        d_down = dist[x, min(y + 1, A_l-1),z]
        d_current = dist[x, y,z]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y,z] = f[min(x + 1, A_h-1), y,z]
            dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
        if idx == 2:
            f[x, y,z] = f[x, min(y + 1, A_l-1),z]
            dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
        if idx == 3:
            f[x, y,z] = f[x,y, min(z + 1, A_w-1)]
            dist[x, y,z] = cal_distance(a, f[x, y,z], A_padding, B, p_size)
    return f,dist

def random_searchsub(f, a, dist, A_padding, B, p_size, alpha=0.5):#随机搜索
    p = p_size // 2
    x = a[0]-p
    y = a[1]-p
    z = a[2]-p
    B_h = np.size(B, 0)
    B_l = np.size(B, 1)
    B_w = np.size(B, 2)
    
    i = 2
    search_h = B_h * alpha ** i
    search_l = B_l * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y,z][0]
    b_y = f[x, y,z][1]
    b_z = f[x, y,z][2]
    while search_h > 1 and search_l > 1 and search_w > 1:
        #print('ssssss')
        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h-p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        '''
        search_min_c = max(b_y - search_l, p)
        search_max_c = min(b_y + search_l, B_l - p)
        #print search_min_c, search_max_c
        random_b_y = np.random.randint(search_min_c, search_max_c)
        search_min_v = max(b_z - search_w, p)
        search_max_v = min(b_z + search_w, B_w - p)
        #print search_min_v, search_max_v
        random_b_z = np.random.randint(search_min_z, search_max_z) 
        '''
        if B_l>=B_w:
           search_min_c = max(b_y - search_l, p)
           search_max_c = min(b_y + search_l, B_l - p)
           #print search_min_c, search_max_c
           random_b_y = np.random.randint(search_min_c, search_max_c)
           random_b_z = np.random.randint(p,B_w-1-p)
           search_l = B_l * alpha ** i
           search_w = B_w
        else:
           random_b_y = np.random.randint(p,B_l-1-p)
           search_min_v = max(b_z - search_w, p)
           search_max_v = min(b_z + search_w, B_w - p)
           #print search_min_v, search_max_v
           random_b_z = np.random.randint(search_min_v, search_max_v)
           search_l = B_l
           search_w = B_w * alpha ** i
       
        search_h = B_h * alpha ** i
        #search_l = B_l * alpha ** i
        #search_w = B_w * alpha ** i
        b = np.array([random_b_x, random_b_y,random_b_z])
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < dist[x, y,z]:
            dist[x, y,z] = d
            f[x, y,z] = b
        i += 1
    #print('stop')
    return f,dist

def NNSBsub(f,dist,img_padding,Bref,p_size,item,name,core):
    for n in range(len(item)):
        f,dist=propagationsub(f, item[n], dist, img_padding, Bref, p_size, bool(random.getrandbits(1)))
        f,dist=random_searchsub(f, item[n], dist, img_padding, Bref, p_size)
    path1='./database/patchmatchprocess('+str(name)+')('+str(core)+')f.npy'
    path2='./database/patchmatchprocess('+str(name)+')('+str(core)+')dist.npy'
    np.save(path1,f)
    np.save(path2,dist)
'''
def reNNSBsub(f,dist,items,name,core,p_size):#重组
    FFF=[]
    DISTDISTDIST=[]
    p=p_size//2
    for n in range(len(items)):#获取总共分了几个进程
        path1='./database/patchmatchprocess('+str(name)+')('+str(n)+')f.npy'
        path2='./database/patchmatchprocess('+str(name)+')('+str(n)+')dist.npy'
        fff=np.load(path1)
        distdistdist=np.load(path2)
        FFF.append(fff)
        DISTDISTDIST.append(distdistdist)
    for n in range(len(items)):
        for xixi in range(len(items[n])):
            f[items[n][xixi][0]-p,items[n][xixi][1]-p,items[n][xixi][2]-p]=FFF[n][items[n][xixi][0]-p,items[n][xixi][1]-p,items[n][xixi][2]-p]
            dist[items[n][xixi][0]-p,items[n][xixi][1]-p,items[n][xixi][2]-p]=DISTDISTDIST[n][items[n][xixi][0]-p,items[n][xixi][1]-p,items[n][xixi][2]-p]
    return f,dist
'''
def reNNSBsub(f,dist,items,name,core,p_size):#重组
    p=p_size//2
    for n in range(len(items)):
        path1='./database/patchmatchprocess('+str(name)+')('+str(n)+')f.npy'
        path2='./database/patchmatchprocess('+str(name)+')('+str(n)+')dist.npy'
        fff=np.load(path1)
        distdistdist=np.load(path2)
        for xixi in range(len(items[n])):
            f[items[n][xixi][0]-p,items[n][xixi][1]-p,items[n][xixi][2]-p]=fff[items[n][xixi][0]-p,items[n][xixi][1]-p,items[n][xixi][2]-p]
            dist[items[n][xixi][0]-p,items[n][xixi][1]-p,items[n][xixi][2]-p]=distdistdist[items[n][xixi][0]-p,items[n][xixi][1]-p,items[n][xixi][2]-p]
    return f,dist
def NNSB3(img, mm,ref, p_size, itr,name,core):#寻找最近零并行版改进 name 为ti编号，core为并行模拟核数
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    #f, dist, img_padding ,Bref= initialization(img,ref, p_size)
    f, dist, img_padding ,Bref= initialization2(img,mm,ref, p_size)
    p=p_size//2
    print("initialization done")
    zuobiaoarr=[]
    for h in range(A_h):
        for x in range(A_l):
            for y in range(A_w):
                if mm[h,x,y]==-1:
                   zuobiaoarr.append(np.array([h+p,x+p,y+p]))
    items=apartlist(zuobiaoarr, int(len(zuobiaoarr)/core))
    #内嵌式多进程
    for itr in range(1,itr+1):
        if itr % 2 == 0:
           processes=list()
           for n in range(len(items)):
               s=multiprocessing.Process(target=NNSBsub, args=(f, dist, img_padding, Bref, p_size,items[n],name,n)) 
               print('process:',n)
               s.start()
               processes.append(s)
           for s in processes:
               s.join()
           f,dist=reNNSBsub(f,dist,items,name,core,p_size)
        else:
           processes=list()
           for n in range(len(items)):
               newList = list(reversed(items[n]))#倒转列表顺序
               s=multiprocessing.Process(target=NNSBsub, args=(f, dist, img_padding, Bref, p_size,newList,name,n)) 
               print('process:',n)
               s.start()
               processes.append(s)
           for s in processes:
               s.join()
           f,dist=reNNSBsub(f,dist,items,name,core,p_size)
        print("iteration: %d"%(itr))
    return f,dist,Bref

def patchmatch3dnewver(m,mm,Tilist,size,itr,core):#patchmatch做优化 并行版3,同时返回模板使用统计Ti core为并行核数
    #size为模板大小，itr为迭代次数
    Fore = np.zeros([m.shape[0], m.shape[1],m.shape[2]])
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    F=[]
    DIST=[]
    BTilist=[]
    processes=list()
    for n in range(len(Tilist)):
        f,dist,Bref=NNSB3(m,mm,Tilist[n],size,itr,n,core)
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)
    print("Searching done!")
    #print F[0],F[1],F[2],F[3]
    Fore=projectTTT(m,F,DIST)
    #print Fore
    print("Pick done!")
    Re,CTilist=reconstructionTTTZ2(m,mm,F,Fore,BTilist)
    del F,Fore,BTilist,f,dist,Bref
    #清除内存
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re,CTilist
def patchmatch3dnewver2(m,mm,Tilist,size,itr,core):#patchmatch做优化 并行版4,同时返回模板使用统计Ti core为并行核数
    #size为模板大小，itr为迭代次数
    #增加新加速模式，首先提取列表，按照一定重叠区选择待模拟点
    #重构过程根据多源融合结果整块复制
    Fore = np.zeros([m.shape[0], m.shape[1],m.shape[2]])
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    F=[]
    DIST=[]
    BTilist=[]
    processes=list()
    for n in range(len(Tilist)):
        f,dist,Bref=NNSB3(m,mm,Tilist[n],size,itr,n,core)
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)
    print("Searching done!")
    #print F[0],F[1],F[2],F[3]
    Fore=projectTTT(m,F,DIST)
    #print Fore
    print("Pick done!")
    Re,CTilist=reconstructionTTTZ2(m,mm,F,Fore,BTilist)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re,CTilist

