#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np

def Normalize(data): #归一化
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

def entropy(data):#计算交叉熵--(1-H)权重
    m,n=data.shape
    #k=1/np.log(m)
    k=1
    yij=data.sum(axis=0)
    #print yij
    pij=data/yij
    #第二步，计算pij
    test=np.zeros((m,n))
    test[:,0]=pij[:,0]*np.log(pij[:,0])
    test[:,1]=pij[:,0]*np.log(pij[:,1])
    test[:,2]=pij[:,0]*np.log(pij[:,2])
    test=np.nan_to_num(test)
    ej=-k*(test.sum(axis=0))
    #计算每种指标的交叉熵
    wi=(1-ej)/np.sum(1-ej)#计算每种指标的权重
    return wi
def Cross_entropyweight(data):#根据交叉熵计算各模板分数--倒数权重
    data2=data
    m,n=data.shape
    k=1/np.log(m)
    yij=data.sum(axis=0)
    #print yij
    pij=data/yij
    #第二步，计算pij
    test=np.zeros((m,n))
    test[:,0]=pij[:,0]*np.log(pij[:,0])
    test[:,1]=pij[:,0]*np.log(pij[:,1])
    test[:,2]=pij[:,0]*np.log(pij[:,2])
    #计算每种指标的交叉熵
    test = np.nan_to_num(test)
    ej = -k * (test.sum(axis=0))
    print(ej)
    d = ej
    d = np.reciprocal(d) * 1#倒数
    d_sum2 = d.sum()
    wi = d / d_sum2#计算每种指标的权重
    print(wi)
    score=countscore(wi,data2)
    return score

def maxscore(data):#计算最高得分的序列数（从0开始
    a=Cross_entropyweight(data)
    print(np.argmax(a, axis=0))
    return a




#test
'''''
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
'''''
'''
li=[[0.1,0.3,0.3],[0.2,0.1,0.3],[0.2,0.2,0.1],[0.2,0.2,0.1],[0.3,0.2,0.2]]
data = np.array(li)
data2=data
print(data2)
data=Normalize(data)
print (data)
m,n=data.shape
print((m,n))
print(entropyweight(np.array(li)))
maxscore(np.array(li))

#第一步读取文件，如果未标准化，则标准化
#data=data.as_matrix(columns=None)
#将dataframe格式转化为matrix格式
k=1/np.log(m)
yij=data.sum(axis=0)
#print yij
pij=data/yij
print(pij)
#第二步，计算pij
'''
li=[[0.1,0.1,0.5],[0.2,0.2,0.1],[0.2,0.2,0.05],[0.2,0.2,0.1],[0.3,0.3,0.25]]
data = np.array(li)
m,n=data.shape
pij = np.array(li)
k=1/np.log(m)
test = np.zeros((m, n),float)

test[:, 0] = pij[:, 0] * np.log(pij[:, 0])
test[:, 1] = pij[:, 0] * np.log(pij[:, 1])
test[:, 2] = pij[:, 0] * np.log(pij[:, 2])

test=np.nan_to_num(test)
ej=-k*(test.sum(axis=0))
print(ej)
#计算每种指标的信息熵
d=ej
d = np.reciprocal(d) * 1
d_sum2 = d.sum()
wi = d / d_sum2

#计算每种指标的权重
print (wi)
#data3=countscore(wi,data2)
#print (data3)
p=Cross_entropyweight(data)
print(p)
r = np.argmax(p)
print(r)














