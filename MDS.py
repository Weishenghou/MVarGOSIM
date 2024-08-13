#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import os
import math
import re
def sgemstonp(filename):#转换代码sgems文件至np文件，filename为文件个数
    idnum=0
    for nnn in range(1, filename+1):
        ifileob='./data/result-'+str(nnn)+'.txt'
        
        f = open(ifileob)
        #print (f)
        content = f.readline()
        nums=[]
        nums=re.findall(r'\d+(?:\.\d+)?',content)
        print (nums[0],nums[1],nums[2])
        line = f.readline()
        line = f.readline()
    #    while line:
    #        line = f.readline()
    #        ldata.append(line)
        #按三维数组来存储
        m1=np.zeros((int(nums[0]),int(nums[1]),int(nums[2])),float)
        for iz in range(int(nums[2])):
            for iy in range(int(nums[1])):
                for ix in range(int(nums[0])):
                    line=f.readline()
                    line=line.strip('/r/n')
                    #print iz,iy,ix
                    val=float(line)
                    #val=re.findall(r'\d+(?:\.\d+)?',line)
                    #print val
                    m1[ix,iy,iz]=int(val)
        path3='./model/'+str(idnum)+'.npy'
        np.save(path3,m1)
        print(m1)
        f.close()
        idnum=idnum+1
        print('one done')

def hamming_distance2(drill1,drill2):#模型汉明距离计算
    smstr=0
    for n in range(drill1.shape[0]):
        vector1 = np.mat(drill1[n,:,:])
        vector2 = np.mat(drill2[n,:,:])

        vector3 = vector1-vector2

        #print("vector3 = vector1-vector2",vector3)
        smstr =smstr+np.shape(np.nonzero(vector1-vector2)[0])[0]
    return smstr

def dismatrix(Modellist):#距离矩阵构建，Modellist为输入的模型列表
    s=len(Modellist)
    D=np.zeros((s,s),int)
    for x in range(s):
        for y in range(s):
            D[x,y]=hamming_distance2(Modellist[x],Modellist[y])
    return D
def dismatrix2(s):#距离矩阵构建改良版，s为输入的模型数量

    D=np.zeros((s,s),int)
    for x in range(s):
        path1='./model/'+str(x)+'.npy'
        Model1=np.load(path1)
        for y in range(s):
            path2='./model/'+str(y)+'.npy'
            Model2=np.load(path2)
            D[x,y]=hamming_distance2(Model1,Model2)
    return D
def MDSshow(D,d):#d为维数，D为输入的距离矩阵
    S=MDS(d)
    S.fit(D)
    SS=S.fit_transform(D)
    #plt.scatter(SS[:,0],SS[:,1])
    col=list(range(0, SS.shape[0]))
    plt.scatter(SS[:,0],SS[:,1],c=col)#默认为二维，三维则添加SS[:,2]
    plt.title(' sklearn MDS')

    
    
    
    
''' #测试代码  1
D = [[0,587,1212,701,1936,604,748,2139,2182,543],
[587,0,920,940,1745,1188,713,1858,1737,597],
[1212,920,0,879,831,1726,1631,949,1021,1494],
[701,940,879,0,1374,968,1420,1645,1891,1220],
[1936,1745,831,1374,0,2339,2451,347,959,2300],
[604,1188,1726,968,2339,0,1092,2594,2734,923],
[748,713,1631,1420,2451,1092,0,2571,2408,205],
[2139,1858,949,1645,347,2594,2571,0,678,2442],
[2182,1737,1021,1891,959,2734,2408,678,0,2329],
[543,597,1494,1220,2300,923,205,2442,2329,0]]

MDSshow(D,2)
''' 
'''
S=MDS(2)
print S
S.fit(D)
print S
SS=S.fit_transform(D)
print SS
plt.scatter(SS[:,0],SS[:,1])
plt.title(' sklearn MDS')
a=(math.sqrt((SS[3,0]-SS[4,0])**2+(SS[3,1]-SS[4,1])**2))
b=(math.sqrt((SS[1,0]-SS[8,0])**2+(SS[1,1]-SS[8,1])**2))
print(a/b)
print (1374/float(1737))


'''



'''#测试代码  2
a=np.load('./output/T3.npy')
b=np.load('./output/T4.npy')
Modellist=[]                  #构建模型列表
Modellist.append(a)
Modellist.append(b)
D=dismatrix(Modellist)         #获得距离矩阵
print D
MDSshow(D,2)                    #MDS分析与可视化
'''

'''
sgemstonp(2)
D=dismatrix2(2)
np.savetxt('D.txt',D)

'''