#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from Fault import*
from PIL import Image
import os
def Ttrans(surface):#将所有非-1值转换为1
    surfacet=surface.copy()
    for x in range(surface.shape[0]):
        for y in range(surface.shape[1]):
            if surface[x,y]!=-1:
                surfacet[x,y]=1
    return surfacet
                    
def testF(npz):#检测是否是端点
    blist=[]
    b=np.array([[-1,-1,-1],
            [-1,1,-1],
            [-1,1,-1]])
    blist.append(b)
    b=np.array([[-1,-1,-1],
            [-1,1,-1],
            [1,-1,-1]])
    blist.append(b)
    b=np.array([[-1,-1,-1],
            [1,1,-1],
            [-1,-1,-1]])
    blist.append(b)
    b=np.array([[1,-1,-1],
            [-1,1,-1],
            [-1,-1,-1]])
    blist.append(b)
    b=np.array([[-1,1,-1],
            [-1,1,-1],
            [-1,-1,-1]])
    blist.append(b)
    b=np.array([[-1,-1,1],
            [-1,1,-1],
            [-1,-1,-1]])
    blist.append(b)
    b=np.array([[-1,-1,-1],
            [-1,1,1],
            [-1,-1,-1]])
    blist.append(b)
    b=np.array([[-1,-1,-1],
            [-1,1,-1],
            [-1,-1,1]])
    blist.append(b)
    
    b=np.array([[-1,-1,-1],
            [-1,1,1],
            [-1,1,1]])
    blist.append(b)
    b=np.array([[-1,-1,-1],
            [1,1,-1],
            [1,1,-1]])
    blist.append(b)
    b=np.array([[1,1,-1],
            [1,1,-1],
            [-1,-1,-1]])
    blist.append(b)
    b=np.array([[-1,1,1],
            [-1,1,1],
            [-1,-1,-1]])
    blist.append(b)
    
    b=np.array([[-1,-1,-1],
            [-1,1,-1],
            [-1,1,1]])
    blist.append(b)
    b=np.array([[-1,-1,-1],
            [-1,1,-1],
            [1,1,-1]])
    blist.append(b)
    b=np.array([[1,1,-1],
            [-1,1,-1],
            [-1,-1,-1]])
    blist.append(b)
    b=np.array([[-1,1,1],
            [-1,1,-1],
            [-1,-1,-1]])
    blist.append(b)
    
    b=np.array([[-1,-1,-1],
            [-1,1,1],
            [-1,-1,1]])
    blist.append(b)
    b=np.array([[-1,-1,-1],
            [1,1,-1],
            [1,-1,-1]])
    blist.append(b)
    b=np.array([[1,-1,-1],
            [1,1,-1],
            [-1,-1,-1]])
    blist.append(b)
    b=np.array([[-1,1,-1],
            [-1,1,1],
            [-1,-1,-1]])
    blist.append(b)
    


    b=np.array([[-1,-1,-1],
            [-1,1,-1],
            [-1,-1,-1]])
    blist.append(b)

    for n in range(len(blist)):
        if (npz == blist[n]).all():
            return True
    return False
    
    
    
def Lofound(surface):
    m=np.pad(surface,((1,1),(1,1)),'constant',constant_values = ((-1,-1),(-1,-1)))
    #print(m)
    m2=Ttrans(m)
    zuobiaoA=[]
    lista=[]
    for x in range(1,surface.shape[0]+1):
        for y in range(1,surface.shape[1]+1):
            if  testF(m2[x-1:x+2,y-1:y+2]):
                #print('one')
                zuobiaoA.append((x-1,y-1))
    return zuobiaoA

def buildsoft(surface):
    zuobiaoA=Lofound(surface)
    for n in range(len(zuobiaoA)):
        for m in range(len(zuobiaoA)):
            x=(zuobiaoA[n][0]+zuobiaoA[m][0])//2
            y=(zuobiaoA[n][1]+zuobiaoA[m][1])//2
            if surface[x,y]==-1:
                #print('one',[x,y],(surface[zuobiaoA[n]]+surface[zuobiaoA[m]])//2)
                surface[x,y]=(surface[zuobiaoA[n]]+surface[zuobiaoA[m]])//2
    return surface

def twodfaultsurfaceS(m,value):#fault trans to 二维俯视面.value为模拟值
    L=m.shape[1]
    W=m.shape[2]
    mm=np.pad(m,((1,1),(0,0),(0,0)),'constant', constant_values=-1)
    top=-np.ones((L,W),int)
    bottom=-np.ones((L,W),int)
    for x in range(L):
        for y in range(W):
            count=0
            for h in range(1,m.shape[0]+1):
                if mm[h,x,y]!=-1:
                   if mm[h,x,y]==value:
                      if mm[h-1,x,y]!=value:
                         top[x,y]=h
  
                      if mm[h+1,x,y]!=value:
                         bottom[x,y]=h
                      count=1
                   if mm[h,x,y]!=value and count==0:
                      top[x,y]=0
                      bottom[x,y]=0
    return top,bottom

def imgloaderforkeras2(kfc,flag0):
    L=kfc.shape[0]
    W=kfc.shape[1]
    x_data=[]
    xlist=[]
    ylist=[]
    y_data=[]
    z_data=[]
    inputdata=[]
    count=0
    xcount=0
    for x in range(L):
        for y in range(W):
            if kfc[x,y]!=-1:#当非空值时
                xlist.append(float(x))
                ylist.append(float(y))  

    

    if flag0==0:   
       x1=int(min(xlist))
       x2=int(max(xlist))
       y1=int(min(ylist))
       y2=int(max(ylist))
    else:
       x1=0
       x2=L-1
       y1=0
       y2=W-1
    
    for x in range(x1,x2+1):
        for y in range(y1,y2+1):
            if kfc[x,y]!=-1:#当非空值时
                x_data.append([float(x)/float(L),float(y)/float(W)])
     
                y_data.append(float(kfc[x,y]))
                count=count+1
            else:
                inputdata.append([float(x)/float(L),float(y)/float(W)])
                xcount=xcount+1
    x_data=np.array(x_data).astype(np.float32)
    x_data=x_data.reshape(count,2)
    inputdata=np.array(inputdata).astype(np.float32)
    inputdata=inputdata.reshape(xcount,2)
    y_data=np.array(y_data).astype(np.float32)
    y_data=y_data.reshape(count,1)



 


    #print x_data,y_data,z_data
    return x_data,y_data,inputdata

def buildfaultkerasS3(m,value,epoch,name,flag0):#重构地层体
    
    bottom,top=twodfaultsurface(m,value)
    
    #for n in range(1):
    #####    top=buildsoft(top)
    #    bottom=buildsoft(bottom)
    
    #top=buildsoft(top)
    #bottom=buildsoft(bottom)
    
    x_data,y_data,inputdata=imgloaderforkeras2(top,flag0)
    maxh=max(y_data)
    y_data=y_data/m.shape[0]

    #print(x_data)
    '''
    ydatacount=[]
    for n in range(len(y_data)):
        if y_data[n] not in ydatacount:
           ydatacount.append(y_data[n])
    
    s=[]
    if len(ydatacount)>1:
       s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    else:
       for n in range(len(inputdata)):
           s.append(y_data[0])
    '''
    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    s=s*m.shape[0]
    for h in range(len(inputdata)):
        #print(inputdata[h])
        top[int(inputdata[h][0]*float(m.shape[1])+0.5),int(inputdata[h][1]*float(m.shape[2])+0.5)]=int(s[h]+0.5)





    x_data,y_data,inputdata=imgloaderforkeras2(bottom,flag0)
    minh=min(y_data)
    y_data=y_data/m.shape[0]
    '''
    ydatacount=[]
    for n in range(len(y_data)):
        if y_data[n] not in ydatacount:
           ydatacount.append(y_data[n])
    s=[]
    if len(ydatacount)>1:
       s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    else:
       for n in range(len(inputdata)):
           s.append(y_data[0])
    '''
    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    s=s*m.shape[0]
    for h in range(len(inputdata)):
        #print(inputdata[h])
        bottom[int(inputdata[h][0]*float(m.shape[1])+0.5),int(inputdata[h][1]*float(m.shape[2])+0.5)]=int(s[h]+0.5)



    path='./database/init'+str(name)+'.npy'
    np.save(path,top)
    #top2=np.zeros((top.shape[0],top.shape[1],3),int)
    #for x in range(top.shape[0]):
    #    for y in range(top.shape[1]):
    #        top2[x,y,:]=top[x,y]
    #top2=Image.fromarray(top.astype(np.uint8))
    #top2.show()
    path='./database/inib'+str(name)+'.npy'
    np.save(path,bottom)

    c=np.zeros((2),int)
    c[0]=maxh
    c[1]=minh
    path='./database/c'+str(name)+'.npy'
    np.save(path,c)
    #bottom2=np.zeros((top.shape[0],top.shape[1],3),int)
    #for x in range(top.shape[0]):
    #    for y in range(top.shape[1]):
    #        bottom2[x,y,:]=bottom[x,y]
    #bottom2=Image.fromarray(bottom.astype(np.uint8))
    #bottom2.show()

def buildfaultkerasM3(m,valuelist,flaglist,epoch):#重构地层体multiprocess,flaglist
    processes=list()
    for n in range(len(valuelist)):
        print('process')
    
        s=multiprocessing.Process(target=buildfaultkerasS3,args=(m,valuelist[n],epoch,n,flaglist[n]))
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    for n in range(len(valuelist)):
        path='./database/init'+str(n)+'.npy'
        top=np.load(path)
        path='./database/inib'+str(n)+'.npy'
        bottom=np.load(path)
        path='./database/c'+str(n)+'.npy'
        c=np.load(path)
        
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                print(x,y)
                print(top.shape[0],top.shape[1])
                if top[x,y]!=-1:
                   if (top[x,y]>bottom[x,y]):
      
                      #for h in range(max(0,bottom[x,y]),min(m.shape[0],top[x,y])):
                      for h in range(max(0,min(top[x,y],bottom[x,y])),min(m.shape[0],max(bottom[x,y],top[x,y]))):
                          if h<=c[0] and h>=c[1]:
                             if m[h,x,y]==-1:
                                m[h,x,y]=valuelist[n]
                   elif abs(top[x,y]-bottom[x,y])==0:
                      for h in range(max(c[1],top[x,y]),min(c[0],top[x,y])):
                          if h<=c[0] and h>=c[1]:
                             if m[h,x,y]==-1:
                                m[h,x,y]=valuelist[n]





    for n in range(len(valuelist)):
        path='./database/init'+str(n)+'.npy'
        os.remove(path)
        path='./database/inib'+str(n)+'.npy'
        os.remove(path)
        path='./database/c'+str(n)+'.npy'
        os.remove(path)
        
    return m


def buildfaultkerasM4(m,valuelist,flaglist,epoch):#重构地层体multiprocess,flaglist,分别重构，自行调整版本
    processes=list()
    for n in range(len(valuelist)):
        print('process')
    
        s=multiprocessing.Process(target=buildfaultkerasS3,args=(m,valuelist[n],epoch,n,flaglist[n]))
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    for n in range(len(valuelist)):
        path='./database/init'+str(n)+'.npy'
        top=np.load(path)
        path='./database/inib'+str(n)+'.npy'
        bottom=np.load(path)
        path='./database/c'+str(n)+'.npy'
        c=np.load(path)
        mm=m.copy()
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                
                print(x,y)
                print(top.shape[0],top.shape[1])
                if top[x,y]!=-1:

                   if (bottom[x,y]-bottom[x,y]<=3 and bottom[x,y]-bottom[x,y]>0) or (top[x,y]>bottom[x,y]):
      
                      #for h in range(max(0,bottom[x,y]),min(m.shape[0],top[x,y])):
                      for h in range(max(0,min(top[x,y],bottom[x,y])),min(m.shape[0],max(bottom[x,y],top[x,y])+1)):
                          if h<=c[0] and h>=c[1]:
                             if mm[h,x,y]==-1:
                                mm[h,x,y]=valuelist[n]
                   elif abs(top[x,y]-bottom[x,y])==0:
                      for h in range(max(c[1],top[x,y]-1),min(c[0],top[x,y]+2)):
                          if h<=c[0] and h>=c[1]:
                             if mm[h,x,y]==-1:
                                mm[h,x,y]=valuelist[n]

        path='./output/pythia'+str(valuelist[n])+'.npy'
        mm=np.save(path,mm)
        path2='./output/'+str(valuelist[n])+'.vtk'
        data=m.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, path2) 

    for n in range(len(valuelist)):
        path='./database/init'+str(n)+'.npy'
        os.remove(path)
        path='./database/inib'+str(n)+'.npy'
        os.remove(path)
        path='./database/c'+str(n)+'.npy'
        os.remove(path)
    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/drill.vtk') 
    
    return m
def imgloaderforkeras3(kfc,flag0):
    print(kfc)



    lista=[]

    for h in range(kfc.shape[0]):
        for y in range(kfc.shape[1]):
            if kfc[h,y] not in lista:
               lista.append(kfc[h,y])
    print(lista)
    L=kfc.shape[0]
    W=kfc.shape[1]
    x_data=[]
    xlist=[]
    ylist=[]
    y_data=[]
    z_data=[]
    inputdata=[]
    count=0
    xcount=0
    for x in range(L):
        for y in range(W):
            if kfc[x,y]!=-1 and kfc[x,y]!=-2:#当非空值时
                xlist.append(float(x))
                ylist.append(float(y))  

    

    if flag0==0:   
       x1=int(min(xlist))
       x2=int(max(xlist))
       y1=int(min(ylist))
       y2=int(max(ylist))
    else:
       x1=0
       x2=L-1
       y1=0
       y2=W-1
    
    for x in range(x1,x2+1):
        for y in range(y1,y2+1):
            if kfc[x,y]!=-1 and kfc[x,y]!=-2:#当非空值时
                x_data.append([float(x)/float(L),float(y)/float(W)])
                
                y_data.append(float(kfc[x,y]))
                count=count+1
            elif kfc[x,y]==-1:
                inputdata.append([float(x)/float(L),float(y)/float(W)])
                xcount=xcount+1
    print(len(x_data))
    print(len(inputdata))
    x_data=np.array(x_data).astype(np.float32)
    x_data=x_data.reshape(count,2)
    inputdata=np.array(inputdata).astype(np.float32)
    inputdata=inputdata.reshape(xcount,2)
    y_data=np.array(y_data).astype(np.float32)
    y_data=y_data.reshape(count,1)



 


    #print x_data,y_data,z_data
    return x_data,y_data,inputdata
def buildfaultkerasS5(m,value,epoch,name,flag0,bottom,top):#重构地层体
    

    
    #for n in range(1):
    #####    top=buildsoft(top)
    #    bottom=buildsoft(bottom)
    
    #top=buildsoft(top)
    #bottom=buildsoft(bottom)
    print(top,bottom)
    x_data,y_data,inputdata=imgloaderforkeras3(top,flag0)
    maxh=max(y_data)
    y_data=y_data/m.shape[0]

    #print(x_data)

    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    s=s*m.shape[0]
    for h in range(len(inputdata)):
        #print(inputdata[h])
        top[int(inputdata[h][0]*float(m.shape[1])+0.5),int(inputdata[h][1]*float(m.shape[2])+0.5)]=int(s[h]+0.5)





    x_data,y_data,inputdata=imgloaderforkeras3(bottom,flag0)
    minh=min(y_data)
    y_data=y_data/m.shape[0]

    s= faultmodelbuild(x_data,y_data,inputdata,epoch)
    s=s*m.shape[0]
    for h in range(len(inputdata)):
        #print(inputdata[h])
        bottom[int(inputdata[h][0]*float(m.shape[1])+0.5),int(inputdata[h][1]*float(m.shape[2])+0.5)]=int(s[h]+0.5)



    path='./database/init'+str(name)+'.npy'
    np.save(path,top)
    #top2=np.zeros((top.shape[0],top.shape[1],3),int)
    #for x in range(top.shape[0]):
    #    for y in range(top.shape[1]):
    #        top2[x,y,:]=top[x,y]
    #top2=Image.fromarray(top.astype(np.uint8))
    #top2.show()
    path='./database/inib'+str(name)+'.npy'
    np.save(path,bottom)

    c=np.zeros((2),int)
    c[0]=maxh
    c[1]=minh
    path='./database/c'+str(name)+'.npy'
    np.save(path,c)
    #bottom2=np.zeros((top.shape[0],top.shape[1],3),int)
    #for x in range(top.shape[0]):
    #    for y in range(top.shape[1]):
    #        bottom2[x,y,:]=bottom[x,y]
    #bottom2=Image.fromarray(bottom.astype(np.uint8))
    #bottom2.show()




def buildfaultkerasM5(m,valuelist,flaglist,epoch,Allusinglist,startornot):#重构地层体multiprocess,flaglist,分别重构，自行调整版本ver 5.0
# 添加了将顶底面输出为图片并可以自己修改的功能,Allusinglist为usinglist的集合，usinglist代表当前待模拟值用到的剖面数目
#startornot代表是否使用已经准备好的top和bottom图片的集合，0默认不使用1使用
    processes=list()

    for n in range(len(valuelist)):
        if startornot[n]!=1:
           bottom,top=twodfaultsurfacedelta(m,valuelist[n],Allusinglist[n],0)
           cv2.imwrite('./output/'+str(valuelist[n])+'/top.bmp',top)
           cv2.imwrite('./output/'+str(valuelist[n])+'/bottom.bmp',bottom)

        

    print('顶底面图构建完毕')

    for n in range(len(valuelist)):
        print('process')
        top=cv2.imread('./output/'+str(valuelist[n])+'/top.bmp',0)
        top=top.astype(np.int32)
        lista=[]
        for h in range(top.shape[0]):
            for y in range(top.shape[1]):
                if top[h,y] not in lista:
                   lista.append(top[h,y])
        print(lista)
        bottom=cv2.imread('./output/'+str(valuelist[n])+'/bottom.bmp',0)
        bottom=bottom.astype(np.int32)
        for x in range(top.shape[0]):
            for y in range(top.shape[1]):
                if top[x,y]==0:
                   top[x,y]=-1
                   #print('-1')
                if top[x,y]==199:
                   top[x,y]=-2
                   #print('-2')
                if bottom[x,y]==0:
                   bottom[x,y]=-1
                   #print('-1')
                if bottom[x,y]==199:
                   bottom[x,y]=-2
                   #print('-2')
        print(top,bottom)
        s=multiprocessing.Process(target=buildfaultkerasS5,args=(m,valuelist[n],epoch,n,flaglist[n],bottom,top))
        s.start()
        processes.append(s)
    for s in processes:
        s.join()
    for n in range(len(valuelist)):
        path='./database/init'+str(n)+'.npy'
        top=np.load(path)
        path='./database/inib'+str(n)+'.npy'
        bottom=np.load(path)
        path='./database/c'+str(n)+'.npy'
        c=np.load(path)
        mm=m.copy()
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                
                print(x,y)
                print(top.shape[0],top.shape[1])
                if top[x,y]!=-1 and top[x,y]!=-2 and bottom[x,y]!=-1 and bottom[x,y]!=-2:

                   if (top[x,y]>bottom[x,y]):
      
                      #for h in range(max(0,bottom[x,y]),min(m.shape[0],top[x,y])):
                      for h in range(max(0,min(top[x,y],bottom[x,y])),min(m.shape[0],max(bottom[x,y],top[x,y])+1)):
                          if h<=c[0] and h>=c[1]:
                             if mm[h,x,y]==-1:
                                mm[h,x,y]=valuelist[n]
                   elif abs(top[x,y]-bottom[x,y])==0:
                      for h in range(max(c[1],top[x,y]-1),min(c[0],top[x,y]+2)):
                          if h<=c[0] and h>=c[1]:
                             if mm[h,x,y]==-1:
                                mm[h,x,y]=valuelist[n]

        path='./output/pythia'+str(valuelist[n])+'.npy'
        np.save(path,mm)
        path2='./output/'+str(valuelist[n])+'.vtk'
        data=mm.transpose(-1,-2,0)#转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
        grid.point_data.scalars = np.ravel(data,order='F') 
        grid.point_data.scalars.name = 'lithology' 
        write_data(grid, path2) 

    for n in range(len(valuelist)):
        path='./database/init'+str(n)+'.npy'
        os.remove(path)
        path='./database/inib'+str(n)+'.npy'
        os.remove(path)
        path='./database/c'+str(n)+'.npy'
        os.remove(path)
    data=m.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, './output/drill.vtk') 
    
    return m












###########################################################多层次翻转版本






def twodfaultsurfacedeltaM(hard,value,usinglist,nanvalue,highlist):#转换为二维俯视面#有多层翻转的用这个,highlist为记录每个剖面分割高程的列表，为0时不分割
    jvalue=value
    L=hard.shape[1]
    W=hard.shape[2]
    top=[]
    bottom=[]
    valuecount=1
    for n in range(2):
        top.append(255*np.ones((L,W),int))
    
        bottom.append(255*np.ones((L,W),int))
        if highlist[n]!=0:
            valuecount=2

    for nss in range(len(usinglist)):
        guding=[]
        mm=-np.ones((hard.shape[0],L,W),int)
        
        file1=open('./Ti/Tiparameter.txt')
        for nb in range(6*(usinglist[nss]-1)+1):
            content=file1.readline()
        top1=[]
        bottom1=[]
        for nc in range(2):
            top1.append(255*np.ones((L,W),int))
    
            bottom1.append(255*np.ones((L,W),int)) 

        for aa in range(6):
            content=file1.readline()
            string1=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string1))
            guding.append(xx)
        file1.close()
        path='./Ti/'+str(usinglist[nss])+'.bmp'
        section=cv2.imread(path,0)
        #print(guding)

        #top1=-np.ones((L,W),int)
        #bottom1=-np.ones((L,W),int)
        #print(guding[0],guding[1],guding[2],guding[3])
        mm=sectionload_xG2(mm,section,guding[0],guding[1],guding[2],guding[3],guding[4],guding[5],jvalue)#载入剖面	

        mm=np.pad(mm,((1,1),(0,0),(0,0)),'constant', constant_values=-1)

        if highlist[nss]==0:
            for x in range(L):
                for y in range(W):
                    count=0
                    for h in range(1,hard.shape[0]+1):
                        if mm[h,x,y]==value:
                            if mm[h-1,x,y]!=value and count==0:
                                top1[0][x,y]=h

                            if mm[h+1,x,y]!=value:
                                bottom1[0][x,y]=h
                                count=1
        elif highlist[nss]!=0:
            for x in range(L):
                for y in range(W):
                    count=0
                    for h in range(1,highlist[nss]):
                        if mm[h,x,y]==value:
                           
                            if mm[h-1,x,y]!=value and count==0:
                                top1[0][x,y]=h
                                
                            if mm[h+1,x,y]!=value:
                                bottom1[0][x,y]=h
                                count=1
                    count=0
                    for h in range(highlist[nss],hard.shape[0]+1):
                        if mm[h,x,y]==value:
                            if mm[h-1,x,y]!=value and count==0:
                                top1[1][x,y]=h
                            if mm[h+1,x,y]!=value:
                                bottom1[1][x,y]=h
                                count=1

        for n in range(2):
            for x in range(L):
                for y in range(W):
                    if top1[n][x,y]!=255:
                        top[n][x,y]=top1[n][x,y]
                    if bottom1[n][x,y]!=255:
                        bottom[n][x,y]=bottom1[n][x,y]


   

    for n in range(2):
        cv2.imwrite('./output/'+str(value)+'/top'+str(n)+'.bmp',bottom[n])
        cv2.imwrite('./output/'+str(value)+'/bottom'+str(n)+'.bmp',top[n])

    return valuecount









def Faultbuilder(m,value,flag,epoch,Allusinglist,startornot,highlist):#单独重构断层,flag,分别重构，自行调整版本ver 1.0
# 添加了将顶底面输出为图片并可以自己修改的功能,Allusinglist为usinglist的集合，usinglist代表当前待模拟值用到的剖面数目
#startornot代表是否使用已经准备好的top和bottom图片的集合，0默认不使用1使用
    processes=list()





    if startornot!=1:
        valuecount=twodfaultsurfacedeltaM(m,value,Allusinglist,0,highlist)
        #获取顶底层面以及有多少个
        print('顶底面图构建完毕')
    print('process')
    if valuecount==1:
        top=cv2.imread('./output/'+str(value)+'/top0.bmp',0)
        top=top.astype(np.int32)
        #lista=[]

        bottom=cv2.imread('./output/'+str(value)+'/bottom0.bmp',0)
        bottom=bottom.astype(np.int32)
        for x in range(top.shape[0]):
            for y in range(top.shape[1]):
                if top[x,y]==255:
                    top[x,y]=-1
                    #print('-1')
                if top[x,y]==199:
                    top[x,y]=-2
                    #print('-2')
                if bottom[x,y]==255:
                    bottom[x,y]=-1
                    #print('-1')
                if bottom[x,y]==199:
                    bottom[x,y]=-2
                    #print('-2')
        print(top,bottom)
        s=multiprocessing.Process(target=buildfaultkerasS5,args=(m,value,epoch,0,flag,bottom,top))
        s.start()
        processes.append(s)

    else:
         for tom in range(valuecount):
            top=cv2.imread('./output/'+str(value)+'/top'+str(tom)+'.bmp',0)
            top=top.astype(np.int32)
             #lista=[]

            bottom=cv2.imread('./output/'+str(value)+'/bottom'+str(tom)+'.bmp',0)
            bottom=bottom.astype(np.int32)
            for x in range(top.shape[0]):
                for y in range(top.shape[1]):
                    if top[x,y]==255:
                        top[x,y]=-1
                         #print('-1')
                    if top[x,y]==199:
                        top[x,y]=-2
                        #print('-2')
                    if bottom[x,y]==255:
                        bottom[x,y]=-1
                        #print('-1')
                    if bottom[x,y]==199:
                        bottom[x,y]=-2
                        #print('-2')
            print(top,bottom)
            s=multiprocessing.Process(target=buildfaultkerasS5,args=(m,value,epoch,tom ,flag,bottom,top))
            s.start()
            processes.append(s)
    for s in processes:
        s.join()





    if valuecount==1:
        path='./database/init0.npy'
        top=np.load(path)
        path='./database/inib0.npy'
        bottom=np.load(path)
        path='./database/c0.npy'
        c=np.load(path)
        mm=m.copy()
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                
                print(x,y)
                print(top.shape[0],top.shape[1])
                if top[x,y]!=-1 and top[x,y]!=-2 and bottom[x,y]!=-1 and bottom[x,y]!=-2:
                   if (top[x,y]>bottom[x,y]):
      
                       #for h in range(max(0,bottom[x,y]),min(m.shape[0],top[x,y])):
                       for h in range(max(0,min(top[x,y],bottom[x,y])),min(m.shape[0],max(bottom[x,y],top[x,y])+1)):
                           if h<=c[0] and h>=c[1]:
                              if mm[h,x,y]==-1:
                                 mm[h,x,y]=value
                   elif abs(top[x,y]-bottom[x,y])==0:
                       for h in range(max(c[1],top[x,y]-1),min(c[0],top[x,y]+2)):
                           if h<=c[0] and h>=c[1]:
                              if mm[h,x,y]==-1:
                                 mm[h,x,y]=value

    
    if valuecount!=1:
       mm=m.copy()
       for stu in range(valuecount):
           
            path='./database/init'+str(stu)+'.npy'
            top=np.load(path)
            path='./database/inib'+str(stu)+'.npy'
            bottom=np.load(path)
            path='./database/c'+str(stu)+'.npy'
            c=np.load(path)
            print(c)
            print(stu)
            for x in range(m.shape[1]):
                for y in range(m.shape[2]):
                
                    #print(x,y)
                    #print(top.shape[0],top.shape[1])
                    if top[x,y]!=-1 and top[x,y]!=-2 and bottom[x,y]!=-1 and bottom[x,y]!=-2:
                        if (top[x,y]>bottom[x,y]):
          
                            #for h in range(max(0,bottom[x,y]),min(m.shape[0],top[x,y])):
                            for h in range(max(0,min(top[x,y],bottom[x,y])),min(m.shape[0],max(bottom[x,y],top[x,y])+1)):
                                if h<=c[0] and h>=c[1]:
                                    if mm[h,x,y]==-1:
                                        mm[h,x,y]=value
                        elif abs(top[x,y]-bottom[x,y])==0:
                            for h in range(max(c[1],top[x,y]-1),min(c[0],top[x,y]+2)):
                                if h<=c[0] and h>=c[1]:
                                    if mm[h,x,y]==-1:
                                       mm[h,x,y]=value








    path='./output/pythia'+str(value)+'.npy'
    np.save(path,mm)
    path2='./output/'+str(value)+'.vtk'
    data=mm.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, path2) 




    
    return m


def Faultbuilder2(m,value,flag,epoch,Allusinglist,startornot,highlist):#单独重构断层,flag,分别重构，自行调整版本ver 1.0
# 添加了将顶底面输出为图片并可以自己修改的功能,Allusinglist为usinglist的集合，usinglist代表当前待模拟值用到的剖面数目
#startornot代表是否使用已经准备好的top和bottom图片的集合，0默认不使用1使用
    processes=list()





    if startornot!=1:
        valuecount=twodfaultsurfacedeltaM(m,value,Allusinglist,0,highlist)
        #获取顶底层面以及有多少个
        print('顶底面图构建完毕')
    print('process')
    last=cv2.imread('./output/'+str(value)+'/last.bmp',0)
    if valuecount==1:
        top=cv2.imread('./output/'+str(value)+'/top0.bmp',0)
        top=top.astype(np.int32)
        #lista=[]

        bottom=cv2.imread('./output/'+str(value)+'/bottom0.bmp',0)
        bottom=bottom.astype(np.int32)
        for x in range(top.shape[0]):
            for y in range(top.shape[1]):
                if top[x,y]==255:
                    top[x,y]=-1
                    #print('-1')
    

                if bottom[x,y]==255:
                    bottom[x,y]=-1
                    #print('-1')


        print(top,bottom)
        s=multiprocessing.Process(target=buildfaultkerasS5,args=(m,value,epoch,0,flag,bottom,top))
        s.start()
        processes.append(s)

    else:
         for tom in range(valuecount):
            top=cv2.imread('./output/'+str(value)+'/top'+str(tom)+'.bmp',0)
            top=top.astype(np.int32)
             #lista=[]

            bottom=cv2.imread('./output/'+str(value)+'/bottom'+str(tom)+'.bmp',0)
            bottom=bottom.astype(np.int32)
            for x in range(top.shape[0]):
                for y in range(top.shape[1]):
                    if top[x,y]==255:
                        top[x,y]=-1
                         #print('-1')
                    if top[x,y]==199:
                        top[x,y]=-2
                        #print('-2')
                    if bottom[x,y]==255:
                        bottom[x,y]=-1
                        #print('-1')
                    if bottom[x,y]==199:
                        bottom[x,y]=-2
                        #print('-2')
            print(top,bottom)
            s=multiprocessing.Process(target=buildfaultkerasS5,args=(m,value,epoch,tom ,flag,bottom,top))
            s.start()
            processes.append(s)
    for s in processes:
        s.join()





    if valuecount==1:
        path='./database/init0.npy'
        top=np.load(path)
        path='./database/inib0.npy'
        bottom=np.load(path)
        path='./database/c'+str(n)+'.npy'
        c=np.load(path)
        mm=m.copy()
        for x in range(m.shape[1]):
            for y in range(m.shape[2]):
                
                print(x,y)
                print(top.shape[0],top.shape[1])
                if top[x,y]!=-1 and bottom[x,y]!=-1 and last[x,y]!=199:
                   if (top[x,y]>bottom[x,y]):
      
                       #for h in range(max(0,bottom[x,y]),min(m.shape[0],top[x,y])):
                       for h in range(max(0,min(top[x,y],bottom[x,y])),min(m.shape[0],max(bottom[x,y],top[x,y])+1)):
                           if h<=c[0] and h>=c[1]:
                              if mm[h,x,y]==-1:
                                 mm[h,x,y]=value
                   elif abs(top[x,y]-bottom[x,y])==0:
                       for h in range(max(c[1],top[x,y]-1),min(c[0],top[x,y]+2)):
                           if h<=c[0] and h>=c[1]:
                              if mm[h,x,y]==-1:
                                 mm[h,x,y]=value


    if valuecount!=1:
       mm=m.copy()
       for stu in range(valuecount):
           
            path='./database/init'+str(stu)+'.npy'
            top=np.load(path)
            path='./database/inib'+str(stu)+'.npy'
            bottom=np.load(path)
            path='./database/c'+str(stu)+'.npy'
            c=np.load(path)
            print(c)
            print(stu)
            for x in range(m.shape[1]):
                for y in range(m.shape[2]):
                
                    #print(x,y)
                    #print(top.shape[0],top.shape[1])
                    if top[x,y]!=-1 and bottom[x,y]!=-1 and last[x,y]!=199:
                        if (top[x,y]>bottom[x,y]):
          
                            #for h in range(max(0,bottom[x,y]),min(m.shape[0],top[x,y])):
                            for h in range(max(0,min(top[x,y],bottom[x,y])),min(m.shape[0],max(bottom[x,y],top[x,y])+1)):
                                if h<=c[0] and h>=c[1]:
                                    mm[h,x,y]=value
                                    #if mm[h,x,y]==-1:
                                        #mm[h,x,y]=value







    path='./output/pythia'+str(value)+'.npy'
    np.save(path,mm)
    path2='./output/'+str(value)+'.vtk'
    data=mm.transpose(-1,-2,0)#转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), 
             dimensions=data.shape) 
    grid.point_data.scalars = np.ravel(data,order='F') 
    grid.point_data.scalars.name = 'lithology' 
    write_data(grid, path2) 




    
    return mm




def Lossafa(y_true, y_pred):
    return K.mean(K.square(y_pred[0] - y_true[0]), axis=-1)+K.mean(K.square(y_pred[1] - y_true[1]), axis=-1)


def faultmodelbuildFORALL(X, Y, inputdata, epoch):
    model = Sequential()
    rate = 0
    model.add(Dense(50, input_dim=2, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(rate))

    model.add(Dense(300, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate))

    model.add(Dense(2))
    # model.add(Dense(1,activation='linear'))
    model.compile(loss='Lossafa', optimizer='Adamax')

    model.summary()
    Mycallback = [
        earlystopcc(monitor='loss', value=0.00005, verbose=0)  # 0.00005
        # keras.callbacks.EarlyStopping(monitor='loss',patience=100,verbose=0,mode='auto',min_delta=100)
    ]

    history = model.fit(X, Y, epochs=epoch, verbose=1, callbacks=Mycallback)

    output = model.predict(inputdata)
    return output