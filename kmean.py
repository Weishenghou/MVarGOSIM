#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from zuobiao import*
from patterntool import*
from scipy import stats
from AIinitial import*


def datacatch(Re,patternsize):
    Data=np.zeros((Re.shape[0]*Re.shape[1]*Re.shape[2],pow(patternsize,3)),int)
    lag=patternsize//2
    Repad=np.pad(Re,lag,'edge')
    for h in range(lag,Re.shape[0]+lag):
        for x in range(lag,Re.shape[1]+lag):
            for y in range(lag,Re.shape[2]+lag):
                tem=template1(Repad,patternsize,patternsize,patternsize,int(h/beilv),int(x/beilv),int(y/beilv))
                tem2=np.reshape(tem,pow(patternsize,3))
                Data[count,:]=tem2

    #Datareal = np.unique(Data,axis=0)            
    return Data

def Tidatacatch(Re,patternsize):#for Ti detect 
    Data=np.zeros((Re.shape[0]*Re.shape[1]*Re.shape[2],pow(patternsize,3)),int)
    lag=patternsize//2
    Repad=np.pad(Re,lag,'edge')
    count=0
    for h in range(lag,Re.shape[0]+lag):
        for x in range(lag,Re.shape[1]+lag):
            for y in range(lag,Re.shape[2]+lag):
                tem=template1(Repad,patternsize,patternsize,patternsize,int(h),int(x),int(y))
                if temdetect(tem):
                    tem2=np.reshape(tem,pow(patternsize,3))
                    Data[count,:]=tem2
                    count=count+1
    np.save('./output/Tidatasetforanodi.npy',Data)
    #Datareal = np.unique(Data,axis=0)            
    return Data

def Tidatacatchmulti(Re,patternsize,beilv):#for Ti detect 
    Data=np.zeros((Re.shape[0]*Re.shape[1]*Re.shape[2],pow(patternsize,3)),int)
    lag=patternsize//2
    Repad=np.pad(Re,lag,'edge') 
    count=0
    for h in range(lag,Re.shape[0]+lag):
        for x in range(lag,Re.shape[1]+lag):
            for y in range(lag,Re.shape[2]+lag):
                tem=template1(Repad,patternsize,patternsize,patternsize,int(h//beilv),int(x//beilv),int(y//beilv))
                if temdetect(tem):
                    tem2=np.reshape(tem,pow(patternsize,3))
                    
                    Data[count,:]=tem2
                    count=count+1
    np.save('./output/Tidatasetforanodi.npy',Data)
    #Datareal = np.unique(Data,axis=0)            
    return Data

#加载数据
def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter='\t')
    #data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
    return data
    #data =  np.loadtxt(fileName)

#欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))



#欧氏距离计算
def distHamming(x,y):
    smstr=np.nonzero(x-y)
    return np.shape(smstr[0])[0]
# 为给定数据集构建一个包含K个随机质心的集合

def hammingmean(Cluster):#汉明距离条件下求均值的方法
    Ham=np.zeros(Cluster.shape[1],int)
    for n  in range(Cluster.shape[1]):
        Ham[n]=stats.mode(Cluster[:,n])[0][0]
    return Ham


# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    # 获取样本数与特征值
    m,n = dataSet.shape#把数据集的行数和列数赋值给m,n
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = np.zeros((k,n))
    # 循环遍历特征值
    for i in range(k):
        index = int(np.random.uniform(0,m))
        # 计算每一列的质心,并将值赋给centroids
        centroids[i,:] = dataSet[index,:]
        # 返回质心
    return centroids


# k均值聚类
def KMeans(dataSet,k):
    m = np.shape(dataSet)[0]
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True

    # 创建质心,随机K个质心
    centroids = randCent(dataSet,k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    while clusterChange:
        clusterChange = False

        #遍历所有样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有数据找到距离每个点最近的质心,
            # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distance = distHamming(centroids[j,:],dataSet[i,:])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
                clusterAssment[i,:] = minIndex,minDist
        # 遍历所有质心并更新它们的取值
        for j in range(k):
            # 通过数据过滤来获得给定簇的所有点
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            #print(centroids)
            #print(pointsInCluster)
            #centroids[j,:] = np.mean(pointsInCluster,axis=0)
            centroids[j,:]=hammingmean(pointsInCluster)
    print("done")
    # 返回所有的类质心与点分配结果
    return centroids,clusterAssment

def showCluster(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    #if n != 2:
       # print("数据不是二维的")
        #return 1

    mark = ['or','ob','og','ok','^r','+r','sr','dr','<r','pr']
    if k > len(mark):
        print("k值太大了")
        return 1
    #绘制所有样本
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    #绘制质心
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i])

    plt.show()
    
def cluster2CPH(centroids,clusterAssment):
    CPHlist=np.zeros((centroids.shape[0]),float)
    for n in range(centroids.shape[0]):
        count=np.sum(clusterAssment[:,0]==n)
        print(count)
        CPHlist[n]=float(count)/float(clusterAssment.shape[0])

    return CPHlist

def clusterforreal(Redataset,centroids):
    #centriods are from Ti cluster
    m = np.shape(RedataSet)[0]
    ReclusterAssment = np.mat(np.zeros((m,2)))
    for i in range(m):
        minDist = 100000.0
        minIndex = -1
        # 遍历所有数据找到距离每个点最近的质心,
        # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for j in range(k):
            # 计算数据点到质心的距离
            # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
            distance = distHamming(centroids[j,:],RedataSet[i,:])
            # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
            if distance < minDist:
                minDist = distance
                minIndex = j
        if ReclusterAssment[i,0] != minIndex:
            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
            ReclusterAssment[i,:] = minIndex,minDist
    return ReclusterAssment


def CHPTi(Ti,patternsize,k):#main programme
    TiDataset=Tidatacatch(Ti,patternsize)
    print('data catch done')
    Tic,TiclusterAssment=KMeans(TiDataset,k)
    print('kmean done')
    CPHlist=cluster2CPH(Tic,TiclusterAssment)
    print('CHPTi done')
    return Tic,TiclusterAssment,CPHlist
    


def CHPforRe(Re,Tic,patternsize):#main programme
    Dataset=datacatch(Re,patternsize)
    ReclusterAssment=clusterforreal(Redataset,centroids)
    CPHlist=cluster2CPH(Tic,ReclusterAssment)
    return CPHTilist
    

    
    
    

''' 
dataSet = loadDataSet("./cc.txt")
k =3
centroids,clusterAssment = KMeans(dataSet,k)
print(centroids)
print(clusterAssment)
showCluster(dataSet,k,centroids,clusterAssment)
print(cluster2CPH(centroids,clusterAssment))
'''  

