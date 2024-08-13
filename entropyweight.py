#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
def Normalize(data):
    data2=np.zeros_like(data)
    for r in range(data.shape[1]):
        for s in range(data.shape[0]):
            if (max(data[:,r])-min(data[:,r]))!=0:
                data2[s,r]=(data[s,r]-min(data[:,r]))/(max(data[:,r])-min(data[:,r]))
            else:
                data2[s,r]=1
    return data2
def countscore(w,data):
    data=data*w
    #print data
    data2=np.zeros((data.shape[0]),float)
    for r in range(data.shape[0]):
        data2[r]=sum(data[r,:])
    return data2

def entropy(data):#计算信息熵
    m,n=data.shape
    k=1/np.log(m)
    yij=data.sum(axis=0)
    #print yij
    pij=data/yij
    #第二步，计算pij
    test=pij*np.log(pij)
    test=np.nan_to_num(test)
    ej=-k*(test.sum(axis=0))
    #计算每种指标的信息熵
    wi=(1-ej)/np.sum(1-ej)#计算每种指标的权重
    return wi
def entropyweight(data):#根据信息熵计算各模板分数
    data2=data
    data=Normalize(data)
    m,n=data.shape
    k=1/np.log(m)
    yij=data.sum(axis=0)
    #print yij
    pij=data/yij
    #第二步，计算pij
    test=pij*np.log(pij)
    test=np.nan_to_num(test)
    ej=-k*(test.sum(axis=0))
    #计算每种指标的信息熵
    wi=(1-ej)/np.sum(1-ej)#计算每种指标的权重
    score=countscore(wi,data2)
    return score

def maxscore(data):#计算最高得分的序列数（从0开始
    a=entropyweight(data) 
    #print(np.argmax(a, axis=0))
    return a



'''
#test
li=[[100,90,100,84,90,100,100,100,100],
    [100,100,78.6,100,90,100,100,100,100],
    [75,100,85.7,100,90,100,100,100,100],
    [100,100,78.6,100,90,100,94.4,100,100],
    [100,90,100,100,100,90,100,100,80],
    [100,100,100,100,90,100,100,85.7,100],
    [100 ,100 ,78.6,100 ,90 , 100, 55.6,    100, 100],
    [87.5 ,100 ,85.7,100 ,100 ,100, 100 ,100 ,100],
    [100 ,100, 92.9 ,  100 ,80 , 100 ,100 ,100 ,100],
    [100,90 ,100 ,100, 100, 100, 100, 100, 100],
    [100,100 ,92.9 ,100, 90 , 100, 100 ,100 ,100]]
data = np.array(li)
data2=data
print data2
data=Normalize(data)
print data
m,n=data.shape
#第一步读取文件，如果未标准化，则标准化
#data=data.as_matrix(columns=None)
#将dataframe格式转化为matrix格式
k=1/np.log(m)
yij=data.sum(axis=0)
#print yij
pij=data/yij
#第二步，计算pij
test=pij*np.log(pij)
test=np.nan_to_num(test)
ej=-k*(test.sum(axis=0))
#计算每种指标的信息熵
wi=(1-ej)/np.sum(1-ej)
#计算每种指标的权重
print wi
data3=countscore(wi,data2)
print data3
print entropyweight(np.array(li))
maxscore(np.array(li))
''' 

