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

def cal_distance(a, b, A_padding, B, p_size):
    #print(b)
    p = p_size // 2
    patch_a = A_padding[a[0]-p:a[0]+p+1, a[1]-p:a[1]+p+1, a[2]-p:a[2]+p+1]
    #print(a[0]-p,a[0]+p+1,patch_a.shape[0])
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, b[2]-p:b[2]+p+1]
    #print(patch_a.shape[2])

    temp = patch_b - patch_a
    smstr=np.nonzero(temp)
    #print smstr
    dist=np.shape(smstr[0])[0]
    #print dist
    return dist

def mse(imageA, imageB):
 
    err = np.sum((imageA.astype("int") - imageB.astype("int")) ** 2)
    # 进行误差归一化
    err /= float(imageA.shape[0] * imageA.shape[1])

    # 返回结果，该值越小越好，越小说明两张图像越相似
    return err

def cal_distance2(a, b, A_padding, B, p_size):
    #print(b)
    p = p_size // 2
    patch_a = A_padding[a[0]-p:a[0]+p+1, a[1]-p:a[1]+p+1, a[2]-p:a[2]+p+1]
    #print(a[0]-p,a[0]+p+1,patch_a.shape[0])
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, b[2]-p:b[2]+p+1]
    #print(patch_a.shape[2])
    '''
    temp = patch_b - patch_a
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    '''
    return  mse(patch_a, patch_b)



class MyThread(threading.Thread):#多线程使用实例
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)
 
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
        




'''
def cal_distance(a, b, A_padding, B, p_size):#距离计算
    p = p_size // 2
    #patch_a = A_padding[a[0]:a[0]+p_size, a[1]:a[1]+p_size, :]
    #patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, :]
    patch_a = A_padding[a[0]:a[0]+p_size, a[1]:a[1]+p_size,a[2]:a[2]+p_size]
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1,b[2]-p:b[2]+p+1]
    temp = patch_b - patch_a
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    return dist
'''
def reconstruction(f, A, B):#重新构建模型
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                #temp[i, j, :] = B[f[i, j][0], f[i, j][1], :]
                temp[i,j,k] = B[f[i, j,k][0], f[i, j,k][1],f[i, j,k][2]]
    #Image.fromarray(temp).show()
    return temp

def initialization(A, B, p_size):#初始化，构建映射
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
    #print f
    return f, dist, A_padding,B

def propagation(f, a, dist, A_padding, B, p_size, is_odd):#传播
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

def random_search(f, a, dist, A_padding, B, p_size, alpha=0.5):#随机搜索
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
        #random_b_z = np.random.randint(search_min_z, search_max_z) 
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

def NNS(img, ref, p_size, itr):#寻找最近零
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding ,Bref= initialization(img, ref, p_size)
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
    return f,dist,Bref

def NNSB(img, ref, p_size, itr,name,idcard):#寻找最近零并行版
    A_h = np.size(img, 0)
    A_l = np.size(img, 1)
    A_w = np.size(img, 2)
    f, dist, img_padding ,Bref= initialization(img, ref, p_size)
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

def patchmatchT(m,Ti,size,itr):#patchmatch做优化
    #size为模板大小，itr为迭代次数
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    f,dist,Bref=NNS(m,Ti,size,itr)
    print("Searching done")
    Re=reconstruction(f,m,Bref)
    np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re

def projectTTT(m,F,DIST):
    Fore=np.zeros((m.shape[0], m.shape[1],m.shape[2]),int)
    sw=999999
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            for k in range(m.shape[2]):
                for n in range(len(F)):
                    if DIST[n][i,j,k]<=sw:
                       Thor=n
                       sw=DIST[n][i,j,k]

                Fore[i,j,k]=Thor
                sw=999999
    return Fore
def reconstructionTTT(A,F,Fore,BTilist):
    A_h = np.size(A, 0)
    A_l = np.size(A, 1)
    A_w = np.size(A, 2)
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_l):
            for k in range(A_w):
                for n in range(len(F)):
                    if Fore[i,j,k]==n:
                        temp[i,j,k] = BTilist[n][F[n][i, j,k][0], F[n][i, j,k][1],F[n][i, j,k][2]]
    #Image.fromarray(temp).show()
    return temp

    
def patchmatchmultiTiB(m,Tilist,size,itr,name):#patchmatch做优化 并行版
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
    Re=reconstructionTTT(m,F,Fore,BTilist)
    #np.save('./output/Reconstruction.npy',Re)
    print('更新步骤完成')
    end = time.time()
    print(end - start)#计时结束
    return Re

def patchmatchmultiTi(m,Tilist,size,itr):#patchmatch做优化
    #size为模板大小，itr为迭代次数
    Fore = np.zeros([m.shape[0], m.shape[1],m.shape[2]])
    print('本轮迭代步骤开始')
    start = time.time()#计时开始
    F=[]
    DIST=[]
    BTilist=[]

    for n in range(len(Tilist)):
        f,dist,Bref=NNS(m,Tilist[n],size,itr)
        F.append(f)
        DIST.append(dist)
        BTilist.append(Bref)

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










'''
#测试代码

if __name__ == "__main__":
    #img1 = cv2.imread("./1.jpg")
    #ref1 = cv2.imread("./0.jpg")
    m= np.load('./output/outputinitial.npy')
    Tilist=[]
    ref1= np.load('./output/T1.npy')
    
    ref2= np.load('./output/T2.npy')
    ref3= np.load('./output/T3.npy')
    ref4= np.load('./output/T4.npy')
    Tilist.append(ref1)
    Tilist.append(ref2)
    Tilist.append(ref3)
    Tilist.append(ref4)
    #print(ref1.shape[2])
    #print ref1
    #img = np.array(Image.open("./1.bmp"))
    #ref = np.array(Image.open("./2.bmp"))
    p_size =3
    itr = 4
    #x=patchmatch4Ti(m,ref1,ref2,ref3,ref4,p_size,itr)
    patchmatchmultiTi(m,Tilist,p_size,itr)
    
'''
