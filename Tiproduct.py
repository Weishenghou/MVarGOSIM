#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2
import numpy as np
import random
from zuobiao import*
from AIinitial import*
from glob import glob
############全自动生成训练模型与训练图像#########
def vtktrans(m,path):
    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, path) 

def extendTimodels(m,template_h,template_x,template_y):#全自动拓展插入硬数据的待模拟网格
    lag=max(template_h,template_x,template_y)//2
    m2=np.pad(m,lag,'edge')
    d=[]
    for h in range(lag,m2.shape[0]-lag):
            for x in range(lag,m2.shape[1]-lag):
                for y in range(lag,m2.shape[2]-lag):
                    d.append((h,x,y))
    
    for cc in range(lag):
        random.shuffle(d)
        flag=0
        for n in range(len(d)):
            h=d[n][0]
            x=d[n][1]
            y=d[n][2]
            if m2[h,x,y]==-1:
                value=extend2dAI(m2,h,x,y)
                flag=1
                if value!=-1:
                    #print value
                    m[h-lag,x-lag,y-lag]=value
                
                else:
                    if cc==lag-1:
                        m[h-lag,x-lag,y-lag]=value
        if flag==0:
            break
        m2=np.pad(m,lag,'edge')
        #填充为1的 
    return m

def sectionloadandgrid(id,H,X,Y,num,nn):
    #num为随机次数
    path='./Ti/Ti1/'+str(id)+'.bmp'
    section=cv2.imread(path,0)
    c=max(X,Y)
    
    
    g=-np.ones((H,X,Y),int)
    x1=0
    y1=0
    x2=X-x1-1
    y2=Y-1
    g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
    Ti1=g
    path='./TiforGAN/'+str(nn)
    np.save(path,Ti1)
    vtktrans(Ti1,path)
    nn=nn+1
    
    for x1 in range(1,X-1):
        g=-np.ones((H,X,Y),int)
        y1=0
        x2=X-x1-1
        y2=Y-1
        g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
        Ti=extendTimodels(g,c,c,c)
        path='./TiforGAN/'+str(nn)
        np.save(path,Ti)
        vtktrans(Ti,path)
        nn=nn+1
        
    g=-np.ones((H,X,Y),int)
    x1=X-1
    y1=0
    x2=X-x1-1
    y2=Y-1
    g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
    Ti2=g
    path='./TiforGAN/'+str(nn)
    np.save(path,Ti)
    vtktrans(Ti2,path)
    nn=nn+1
    
    for n in range(num):
        x1=random.randint(0,X)
        x2=random.randint(0,X)
        y1=random.randint(0,Y)
        y2=random.randint(0,Y)
        if ((x1-x2)!=0)&((y1-y2)!=0)&(((x1-x2)*(y1-y2))>=(H*Y/16)):
            dis=[]
            disy=[]
            if x1>x2:
                t=x1
                x1=x2
                x2=t
            if y1>y2:
                t=y1
                y1=y2
                y2=t
            for n12 in range(x1,x2+1):
                dis.append(n12)
            for n34 in range(y1,y2+1):
                disy.append(n34)
            Tit=Ti1[:,dis,:]
            Ti=Tit[:,:,disy]
            path='./TiforGAN/'+str(nn)
            np.save(path,Ti)
            vtktrans(Ti1,path)
            nn=nn+1
    for n in range(num):
        x1=random.randint(0,X)
        x2=random.randint(0,X)
        y1=random.randint(0,Y)
        y2=random.randint(0,Y)
        if ((x1-x2)!=0)&((y1-y2)!=0)&(((x1-x2)*(y1-y2))>=(H*Y/16)):
            dis=[]
            disy=[]
            if x1>x2:
                t=x1
                x1=x2
                x2=t
            if y1>y2:
                t=y1
                y1=y2
                y2=t
            for n12 in range(x1,x2+1):
                dis.append(n12)
            for n34 in range(y1,y2+1):
                disy.append(n34)
            Tit=Ti2[:,dis,:]
            Ti=Tit[:,:,disy]
            path='./TiforGAN/'+str(nn)
            np.save(path,Ti)
            vtktrans(Ti,path)
            nn=nn+1
    tem=Y
    Y=X
    X=tem
    
    g=-np.ones((H,X,Y),int)
    x1=0
    y1=0
    x2=X-x1-1
    y2=Y-1
    g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
    Ti3=g
    path='./TiforGAN/'+str(nn)
    np.save(path,Ti3)
    vtktrans(Ti3,path)
    nn=nn+1
    
    for x1 in range(1,X-1):
        g=-np.ones((H,X,Y),int)
        y1=0
        x2=X-x1-1
        y2=Y-1
        g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
        Ti=extendTimodels(g,c,c,c)
        path='./TiforGAN/'+str(nn)
        np.save(path,Ti)
        vtktrans(Ti,path)
        nn=nn+1
        
    g=-np.ones((H,X,Y),int)
    x1=X-1
    y1=0
    x2=X-x1-1
    y2=Y-1
    g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
    Ti4=g
    path='./TiforGAN/'+str(nn)
    np.save(path,Ti4)
    vtktrans(Ti4,path)
    nn=nn+1
    
    for n in range(num):
        x1=random.randint(0,X)
        x2=random.randint(0,X)
        y1=random.randint(0,Y)
        y2=random.randint(0,Y)
        if ((x1-x2)!=0)&((y1-y2)!=0)&(((x1-x2)*(y1-y2))>=(H*Y/16)):
            dis=[]
            disy=[]
            if x1>x2:
                t=x1
                x1=x2
                x2=t
            if y1>y2:
                t=y1
                y1=y2
                y2=t
            for n12 in range(x1,x2+1):
                dis.append(n12)
            for n34 in range(y1,y2+1):
                disy.append(n34)
            Tit=Ti3[:,dis,:]
            Ti=Tit[:,:,disy]
            path='./TiforGAN/'+str(nn)
            np.save(path,Ti)
            vtktrans(Ti,path)
            nn=nn+1
    for n in range(num):
        x1=random.randint(0,X)
        x2=random.randint(0,X)
        y1=random.randint(0,Y)
        y2=random.randint(0,Y)
        if ((x1-x2)!=0)&((y1-y2)!=0)&(((x1-x2)*(y1-y2))>=(H*Y/16)):
            dis=[]
            disy=[]
            if x1>x2:
                t=x1
                x1=x2
                x2=t
            if y1>y2:
                t=y1
                y1=y2
                y2=t
            for n12 in range(x1,x2+1):
                dis.append(n12)
            for n34 in range(y1,y2+1):
                disy.append(n34)
            Tit=Ti4[:,dis,:]
            Ti=Tit[:,:,disy]
            path='./TiforGAN/'+str(nn)
            np.save(path,Ti)
            vtktrans(Ti,path)
            nn=nn+1
            
def gridTiproduction(H,X,Y,Tinum,num):
    for n in range(1,Tinum+1):
        sectionloadandgrid(n,H,X,Y,num)
    H=1/2*H
    X=1/2*X
    Y=1/2*Y
    nn=0
    for n in range(1,Tinum+1):
        sectionloadandgrid(Tinum+n,H,X,Y,num,nn)
        
        
def sectionloadandgrid2(id,c,num,nn):
    #num为随机次数
    path='./Ti/Ti1/'+str(id)+'.bmp'
    section=cv2.imread(path,0)

    
    
    g=-np.ones((3*c,3*c,3*c),int)
    x1=0
    y1=0
    x2=c-x1-1
    y2=c-1
    g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
    Ti1=g


    
    for x1 in range(c):
        g=-np.ones((c,c,c),int)
        y1=0
        x2=c-x1-1
        y2=c-1
        g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
        Ti=extendTimodels(g,c,c,c)
        path='./TiforGAN/'+str(nn)
        np.save(path,Ti)
        vtktrans(Ti,path)
        nn=nn+1
        
    g=-np.ones((3*c,3*c,3*c),int)
    x1=c-1
    y1=0
    x2=0
    y2=c-1
    g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
    Ti2=g

    
    for n in range(num):
        x1=random.randint(c-1,2*c-1)
        x2=x1+c
        y1=random.randint(c-1,2*c-1)
        y2=y1+c
        dis=[]
        disy=[]
        for n12 in range(x1,x2+1):
            dis.append(n12)
        for n34 in range(y1,y2+1):
            disy.append(n34)
        Tit=Ti1[:,dis,:]
        Ti=Tit[:,:,disy]
        g=extendTimodels(g,c,c,c)
        path='./TiforGAN/'+str(nn)
        np.save(path,g)
        vtktrans(g,path)
        nn=nn+1
    for n in range(num):
        x1=random.randint(c-1,2*c-1)
        x2=x1+c
        y1=random.randint(c-1,2*c-1)
        y2=y1+c
        dis=[]
        disy=[]
        for n12 in range(x1,x2+1):
            dis.append(n12)
        for n34 in range(y1,y2+1):
            disy.append(n34)
        Tit=Ti2[:,dis,:]
        Ti=Tit[:,:,disy]
        g=extendTimodels(g,c,c,c)
        path='./TiforGAN/'+str(nn)
        np.save(path,g)
        vtktrans(g,path)
        nn=nn+1

    
    g=-np.ones((3*c,3*c,3*c),int)
    x2=0
    y2=0
    x1=c-x2-1
    y1=c-1
    g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
    Ti3=g

    
    for x1 in range(c):
        g=-np.ones((c,c,c),int)
        y1=0
        x2=X-x1-1
        y2=Y-1
        g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
        Ti=extendTimodels(g,c,c,c)
        path='./TiforGAN/'+str(nn)
        np.save(path,Ti)
        vtktrans(Ti,path)
        nn=nn+1
        
    g=-np.ones((3*c,3*c,3*c),int)
    x2=c-1
    y2=0
    x1=0
    y1=c-1
    g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
    Ti4=g
    
    
    for n in range(num):
        x1=random.randint(c-1,2*c-1)
        x2=x1+c
        y1=random.randint(c-1,2*c-1)
        y2=y1+c
        dis=[]
        disy=[]
        for n12 in range(x1,x2+1):
            dis.append(n12)
        for n34 in range(y1,y2+1):
            disy.append(n34)
        Tit=Ti3[:,dis,:]
        Ti=Tit[:,:,disy]
        g=extendTimodels(g,c,c,c)
        path='./TiforGAN/'+str(nn)
        np.save(path,g)
        vtktrans(g,path)
        nn=nn+1
    for n in range(num):
        x1=random.randint(c-1,2*c-1)
        x2=x1+c
        y1=random.randint(c-1,2*c-1)
        y2=y1+c
        dis=[]
        disy=[]
        for n12 in range(x1,x2+1):
            dis.append(n12)
        for n34 in range(y1,y2+1):
            disy.append(n34)
        Tit=Ti4[:,dis,:]
        Ti=Tit[:,:,disy]
        g=extendTimodels(g,c,c,c)
        path='./TiforGAN/'+str(nn)
        np.save(path,g)
        vtktrans(g,path)
        nn=nn+1
            
            
def gridTiproduction2(c,Tinum,num):
    nn=0
    for n in range(1,Tinum+1):
        sectionloadandgrid2(n,c,num,nn)
        nn=nn+1

def gridTiproduction3(c,num):#c为训练数据要求长宽高，num为延伸个数
    path = glob('./datasets/2dTI/*' )
    nn=0
    print(path)
    print('mission start')
    for img in path:
        section=cv2.imread(img,0)
        g1=-np.ones((c,3*c,3*c),int)
        x1=0
        y1=0
        x2=3*c-x1-1
        y2=3*c-1
        g1=sectionload_x(g1,section,x1,y1,x2,y2)#载入剖面
        Ti1=extendTimodels(g1,c,3*c,3*c)
        np.save('./datasets/ti1',Ti1)


        '''
        for x1 in range(c):
            g=np.ones((c,c,c),int)
            y1=0
            x2=c-x1-1
            y2=c-1
            g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
            Ti=extendTimodels(g,c,c,c)
            path='./datasets/3dTI/'+str(nn)
            np.save(path,Ti)
            vtktrans(Ti,path)
            nn=nn+1
        '''    
        g2=-np.ones((c,3*c,3*c),int)
        x1=3*c-1
        y1=0
        x2=0
        y2=3*c-1
        g2=sectionload_x(g2,section,x1,y1,x2,y2)#载入剖面
        Ti2=extendTimodels(g2,c,3*c,3*c)
        np.save('./datasets/ti2',Ti2)
    
        for n in range(num):
            x1=random.randint(0,2*c-1)
            x2=x1+c
            y1=random.randint(0,2*c-1)
            y2=y1+c
            print(x1,y1)
            g=Ti1[:,x1:x2,y1:y2]
            #g=extendTimodels(g,c,c,c)
            path='./datasets/3dTI/'+str(nn)
            np.save(path,g)
            vtktrans(g,path)
            nn=nn+1
        for n in range(num):
            x1=random.randint(0,2*c-1)
            x2=x1+c
            y1=random.randint(0,2*c-1)
            y2=y1+c
            g=Ti2[:,x1:x2,y1:y2]
            #g=extendTimodels(g,c,c,c)
            path='./datasets/3dTI/'+str(nn)
            np.save(path,g)
            vtktrans(g,path)
            nn=nn+1

    
        g3=-np.ones((c,3*c,3*c),int)
        x2=0
        y2=0
        x1=3*c-x2-1
        y1=3*c-1
        g3=sectionload_x(g3,section,x1,y1,x2,y2)#载入剖面
        Ti3=extendTimodels(g3,c,3*c,3*c)
        np.save('./datasets/ti3',Ti3)
        '''
        for x1 in range(c):
            g=np.ones((c,c,c),int)
            y1=0
            x2=c-x1-1
            y2=c-1
            g=sectionload_x(g,section,x1,y1,x2,y2)#载入剖面
            Ti=extendTimodels(g,c,c,c)
            path='./datasets/3dTI/'+str(nn)
            np.save(path,Ti)
            vtktrans(Ti,path)
            nn=nn+1
        '''   
        g4=-np.ones((c,3*c,3*c),int)
        x2=3*c-1
        y2=0
        x1=0
        y1=3*c-1
        g4=sectionload_x(g4,section,x1,y1,x2,y2)#载入剖面
        Ti4=extendTimodels(g4,c,3*c,3*c)
        np.save('./datasets/ti4',Ti4)
    
        for n in range(num):
            x1=random.randint(0,2*c-1)
            x2=x1+c
            y1=random.randint(0,2*c-1)
            y2=y1+c
            g=Ti3[:,x1:x2,y1:y2]
            path='./datasets/3dTI/'+str(nn)
            np.save(path,g)
            vtktrans(g,path)
            nn=nn+1
        for n in range(num):
            x1=random.randint(0,2*c-1)
            x2=x1+c
            y1=random.randint(0,2*c-1)
            y2=y1+c
            g=Ti4[:,x1:x2,y1:y2]
            #g=extendTimodels(g,c,c,c)
            path='./datasets/3dTI/'+str(nn)
            np.save(path,g)
            vtktrans(g,path)
            nn=nn+1
        print('one done!')
   
gridTiproduction3(64,30)


