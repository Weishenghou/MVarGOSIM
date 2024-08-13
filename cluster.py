#!/usr/bin/env python
# coding: utf-8

# In[3]:



from numpy import *
from editdistance import *
import random
from math import sqrt

#########################################层级聚类####################################
def yezi(clust):
    if clust.left == None and clust.right == None :
        return [clust.id]
    return yezi(clust.left) + yezi(clust.right)
    
#计算两个向量的距离
def cluster_distance(vecA, vecB):
    ss=vecA.shape[0]*vecA.shape[1]*vecA.shape[2]
    drill1=vecA.reshape(ss,1)
    drill2=vecB.reshape(ss,1)
    d=hamming_distance(drill1,drill2)
    return d


'''
class bicluster:
    def __init__(self, vec, left=None,right=None,distance=0.0,id=None):
        self.left = left
        self.right = right  #每次聚类都是一对数据，left保存其中一个数据，right保存另一个
        self.vec = vec      #保存两个数据聚类后形成新的中心
        self.id = id     
        self.distance = distance
        
def hcluster(blogwords,n,name) :
    biclusters = [ bicluster(vec = blogwords[i], id = i ) for i in range(len(blogwords)) ]
    distances = {}
    flag = None;
    currentclusted = -1
    while(len(biclusters) > n) : #假设聚成n个类
        min_val = 999999999999; #Python的无穷大应该是inf
        biclusters_len = len(biclusters)
        for i in range(biclusters_len-1) :
            for j in range(i + 1, biclusters_len) :
                if distances.get((biclusters[i].id,biclusters[j].id)) == None:

                    #print i,j
                    #print biclusters[j].vec
                    distances[(biclusters[i].id,biclusters[j].id)] = cluster_distance(biclusters[i].vec,biclusters[j].vec)
                d = distances[(biclusters[i].id,biclusters[j].id)] 
                if d < min_val :
                    min_val = d
                    flag = (i,j)
        bic1,bic2 = flag #解包bic1 = i , bic2 = j
        r=int(random.randint(0,1))
        if r==0:
            newvec=biclusters[bic1].vec
        else:
            newvec=biclusters[bic2].vec
        #newvec = [(biclusters[bic1].vec[i] + biclusters[bic2].vec[i])/2 for i in range(len(biclusters[bic1].vec))] #形成新的类中心，平均
        newbic = bicluster(newvec, left=biclusters[bic1], right=biclusters[bic2], distance=min_val, id = currentclusted) #二合一
        currentclusted -= 1
        del biclusters[bic2] #删除聚成一起的两个数据，由于这两个数据要聚成一起
        del biclusters[bic1]
        biclusters.append(newbic)#补回新聚类中心
        clusters = [yezi(biclusters[i]) for i in range(len(biclusters))] #深度优先搜索叶子节点，用于输出显示
        print ('run done')
    #np.save('./database/bicluster.npy',biclusters)
    np.save('./database/clusters'+str(name)+'.npy',clusters)
    #return biclusters,clusters
    return clusters




def initialpatterndatabase(cdatabase,N):
    print('start')
    for n in range(len(cdatabase)): 
        p = multiprocessing.Process(target=hcluster, args=(cdatabase[n],N,n)) 
        print('process start') 
        p.start() 
    p.join()
    Cdatabase=[]
    for i in range(len(cdatabase)):
        Cdatabase.append(np.load('./database/clusters'+str(i)+'.npy'))
    np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase

###################################################################################################
'''
#####################################无监督聚类######################################################
#test code
def Simplecluster(database,U,name):#简单聚类方法
    #U为阈值
    Cdatabase=[]
    d=[]
    c=[]
    for n in range(len(database)):
        if n not in c:
            d=[]
            for m in range(n,len(database)):
                if cluster_distance(database[n], database[m])<=U:
                    d.append(m)
                    c.append(m)
            Cdatabase.append(d)
    #print len(c)
    np.save('./database/clusters'+str(name)+'.npy',Cdatabase)
    return Cdatabase

def Simplecluster2(database,U):#简单聚类方法
    #U为阈值
    Cdatabase=[]
    d=[]
    c=[]
    for n in range(len(database)):
        if n not in c:
            d=[]
            for m in range(n,len(database)):
                if cluster_distance(database[n], database[m])<=U:
                    d.append(m)
                    c.append(m)
            Cdatabase.append(d)
    #print len(c)
    return Cdatabase

def FinitialpatterndatabaseSimple(cdatabase,U):
    print('start')
    U1=U*2/3

    p1= multiprocessing.Process(target=Simplecluster, args=(cdatabase[0],U1,0)) 
    print('process start') 
    p1.start()

    p2= multiprocessing.Process(target=Simplecluster, args=(cdatabase[1],U1,1)) 
    print('process start') 
    p2.start()

    p3= multiprocessing.Process(target=Simplecluster, args=(cdatabase[2],U1,2)) 
    print('process start') 
    p3.start()

    p4= multiprocessing.Process(target=Simplecluster, args=(cdatabase[3],U1,3)) 
    print('process start') 
    p4.start()

    p5= multiprocessing.Process(target=Simplecluster, args=(cdatabase[4],U1,4)) 
    print('process start') 
    p5.start()

    p6= multiprocessing.Process(target=Simplecluster, args=(cdatabase[5],U1,5)) 
    print('process start') 
    p6.start()

    p7= multiprocessing.Process(target=Simplecluster, args=(cdatabase[6],U1,6)) 
    print('process start') 
    p7.start()

    p8= multiprocessing.Process(target=Simplecluster, args=(cdatabase[7],U1,7)) 
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
    return True
    
def initialpatterndatabaseSimple(cdatabase,U):#构建基于重叠区分类数据库
    FinitialpatterndatabaseSimple(cdatabase,U)
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
    np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase
###################################################################################################改良版
def FinitialpatterndatabaseSimple2(cdatabase,U):
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

    

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    
    print('process end')
    return True
    
def initialpatterndatabaseSimple2(cdatabase,U):#构建基于重叠区分类数据库
    FinitialpatterndatabaseSimple2(cdatabase,U)
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
    
    Cdatabase.append(cc1)
    Cdatabase.append(cc2)
    Cdatabase.append(cc3)
    Cdatabase.append(cc4)
    Cdatabase.append(cc5)
    Cdatabase.append(cc6)
    Cdatabase.append(cc7)
    Cdatabase.append(cc8)
    Cdatabase.append(cc9)
    
    np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase



###################################################################################################改良版ver3.0
def FinitialpatterndatabaseSimple3(cdatabase,U):
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

 

    

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    
    
    print('process end')
    return True
    
def initialpatterndatabaseSimple3(cdatabase,U):#构建基于重叠区分类数据库
    FinitialpatterndatabaseSimple3(cdatabase,U)
    Cdatabase=[]
    cc1=np.load('./database/clusters0.npy')
    cc2=np.load('./database/clusters1.npy')
    cc3=np.load('./database/clusters2.npy')
    cc4=np.load('./database/clusters3.npy')
    cc5=np.load('./database/clusters4.npy')

    Cdatabase.append(cc1)
    Cdatabase.append(cc2)
    Cdatabase.append(cc3)
    Cdatabase.append(cc4)
    Cdatabase.append(cc5)
    
    
    np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase


def FinitialpatterndatabaseSimplezhu(cdatabase,U):
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


    p1.join()
    p2.join()
    p3.join()
    p4.join()
    
    
    print('process end')
    return True

def initialpatterndatabaseSimplezhu(cdatabase,U):#构建基于重叠区分类柱状数据库
    FinitialpatterndatabaseSimplezhu(cdatabase,U)
    Cdatabase=[]
    cc1=np.load('./database/clusters0.npy')
    cc2=np.load('./database/clusters1.npy')
    cc3=np.load('./database/clusters2.npy')
    cc4=np.load('./database/clusters3.npy')


    Cdatabase.append(cc1)
    Cdatabase.append(cc2)
    Cdatabase.append(cc3)
    Cdatabase.append(cc4)

    
    
    np.save('./database/Cdatabase.npy',Cdatabase)
    return Cdatabase
'''
np.set_printoptions(threshold='nan')#打印数组全部元素
time_start=time.time()#计时开始

c=np.load('da.npy')

print len(c[0])
#print c
#initialpatterndatabase(c,10)
#Cdatabase=Simplecluster(c[0],4)
Cdatabase=initialpatterndatabaseSimple(c,5)

#print Cdatabase[1]

time_end=time.time()
print('timecost:')
print(time_end-time_start)


# In[5]:


print len(Cdatabase[0])
print Cdatabase[0][29]
#print Cdatabase[14]
#print cluster_distance(c[0][Cdatabase[1][120]], c[0][Cdatabase[14][0]])
'''
