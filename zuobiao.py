
# coding: utf-8

# In[51]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab
import numpy as np
import time
from PIL import Image
from pathlib2 import Path
#from pathlib import Path
import os


########################坐标导入##############################################
def gridproduct(x1,y1,x4,y4,h1,h2,beilvh,beilvx,beilvy):
    #x1,y1（对应零点）,x4,y4为两个对角顶点的真实坐标，h1(顶)，h2（底，地下则为负数）是建模的深度，beilv为网格对真实值的比例，
    #如10表示10个网格对应一个单位，
    H=int((h1-h2)*beilvh)+1
    L=int(abs((x4-x1)*beilvx))+1
    W=int(abs((y4-y1)*beilvy))+1
    m=-np.ones((H,L,W),int)
    m=255*m
    print('高，长，宽')
    print(H,L,W)
    return m



def zhenshi2xiangdui(h0,x0,y0,x1,y1,h1,beilvh,beilvx,beilvy):
    #x1,y1（对应零点）,h1(顶)是建模的深度，h0,x0,y0为真实坐标，
    #计算相对坐标
    hz=int((h1-h0)*beilvh)
    xz=int(abs((x0-x1))*beilvx)
    yz=int(abs((y0-y1))*beilvy)
    return hz,xz,yz


def zhenshi2xiangdui_1(m,h0,x0,y0,x1,y1,x4,y4,h1,h2):
    #x1,y1（对应零点）,x4,y4为两个对角顶点的真实坐标，h1(顶)，h2（底，地下则为负数）是建模的深度，h0,x0,y0为真实坐标，需要计算的步骤更多
    h=m.shape[0]-1
    l=m.shape[1]-1
    w=m.shape[2]-1
    hv=float((h1-h2))/h#计算倍率
    xv=float((x4-x1))/l
    yv=float((y4-y1))/w
    #计算相对坐标
    hz=int((h1-h0)/hv+0.5)
    xz=int((x0-x1)/xv+0.5)
    yz=int((y0-y1)/yv+0.5)
    return hz,xz,yz

def zhenshi2xiangduixy_1(m,x0,y0,x1,y1,x4,y4):
    #绝对坐标转相对坐标
    l=m.shape[1]-1
    w=m.shape[2]-1
    xv=float((x4-x1))/l
    yv=float((y4-y1))/w
    #计算相对坐标
    xz=int((x0-x1)/xv+0.5)
    yz=int((y0-y1)/yv+0.5)
    return xz,yz

def zhenshi2xiangduixy(x0,y0,x1,y1,beilvx,beilvy):
    #x1,y1（对应零点）,x0,y0为真实坐标，
    #计算相对坐标
    xz=int(abs((x0-x1))*beilvx)
    yz=int(abs((y0-y1))*beilvy)
    return xz,yz


def xiangdui2zhenshi(m,hz,xz,yz,x1,y1,x4,y4,h1,h2):
    #x1,y1（对应零点）,x4,y4为两个对角顶点的真实坐标，h1(顶)，h2（底，地下则为负数）是建模的深度，hz,xz,yz为相对坐标
    h=m.shape[0]-1
    l=m.shape[1]-1
    w=m.shape[2]-1
    hv=float((h1-h2))/h#计算倍率
    xv=float((x4-x1))/l
    yv=float((y4-y1))/w
    #计算相对坐标
    h0=(h1-(hz*hv))
    x0=(x1+(xz*xv))
    y0=(y1+(yz*yv))
    return h0,x0,y0

def xiangdui2zhenshixy(m,xz,yz,x1,y1,x4,y4):
    #x1,y1（对应零点）,x4,y4为两个对角顶点的真实坐标，xz,yz为绝对坐标
    l=m.shape[1]-1
    w=m.shape[2]-1
    xv=float((x4-x1))/l
    yv=float((y4-y1))/w
    #计算相对坐标
    x0=(x1+(xz*xv))
    y0=(y1+(yz*yv))
    return x0,y0

def drilload_j(m,drill,x0,y0,x1,y1,beilvx,beilvy):#绝对坐标下，钻孔导入,drill必须为数组,x0,y0为绝对坐标
    xz,yz=zhenshi2xiangduixy(x0,y0,x1,y1,beilvx,beilvy)
    m[:,xz,yz]=drill
    return m

def drilload_x(m,drill,xz,yz):#相对坐标的钻孔导入,drill必须为数组,xz,yz为相对坐标
    m[:,xz,yz]=drill
    return m
'''        
def sectionload_x(m,section,xz,yz,xz2,yz2):#相对坐标的剖面导入,section为数组,xz,yz分别为剖面两端点的相对坐标
#斜剖面导入后需要扩充才正确
    ns=section.shape[1]
    if (ns==m.shape[1])&(yz==yz2):#当剖面为竖时 
        m[:,:,yz]=section
    elif (ns==m.shape[2])&(xz==xz2):#当剖面为横时
        m[:,xz,:]=section
    else: #斜剖面
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
        if ns!=lv:
            section=sectionex(section,m.shape[0],lv)
        for n in range(lv):
            m[:,x1,y1]=section[:,n]      
            #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
            x1=int(xz+(n+1)*xlv+0.5)#四舍五入
            y1=int(yz+(n+1)*ylv+0.5)
    return m
'''
def sectionload_x(m,section,xz,yz,xz2,yz2):#相对坐标的剖面导入,section为数组,xz,yz分别为剖面两端点的相对坐标
#斜剖面导入后需要扩充才正确
    ns=section.shape[1]
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
    h=m.shape[0]
    section=sectionex(section,h,lv)
    #print h
    #print section.shape[0],section.shape[1],lv,m.shape[0],m.shape[1],m.shape[2]
    for n in range(lv):
       m[:,x1,y1]=section[:,n]      
       #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
       x1=int(xz+(n+1)*xlv+0.5)#四舍五入
       y1=int(yz+(n+1)*ylv+0.5)
    return m

def sectionread_x(m,xz,yz,xz2,yz2):#相对坐标的剖面读取,section为数组,xz,yz分别为剖面两端点的相对坐标

    if (abs(xz-xz2)==m.shape[1])&(yz==yz2):#当剖面为竖时 
        section=m[:,:,yz]
    elif (abs(yz-yz2)==m.shape[2])&(xz==xz2):#当剖面为横时
        section=m[:,xz,:]
    else: #斜剖面
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
        section=np.zeros((m.shape[0],lv),int)

        for n in range(lv):
            section[:,n]=m[:,x1,y1]      
            #print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
            x1=int(xz+(n+1)*xlv+0.5)#四舍五入
            y1=int(yz+(n+1)*ylv+0.5)
    return section

def sectionload_j(m,section,x0,y0,x02,y02,x1,y1,beilvx,beilvy):#绝对坐标的剖面导入,section为数组,xz,yz分别为剖面两端点的相对坐标
    xz,yz=zhenshi2xiangduixy(x0,y0,x1,y1,beilvx,beilvy)
    xz2,yz2=zhenshi2xiangduixy(x02,y02,x1,y1,beilvx,beilvy)
    m=sectionload_x(m,section,xz,yz,xz2,yz2)
    return m
    
def sectionex(section,height,length):#剖面的缩放
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
                section_new[int(n*kksk),:]=section[n,:]
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
                section_new2[:,int(n*kksk)]=section[:,n]
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

''' 
def simgridex(m,beilv):#beilv为放大或者缩小的倍率
    H=m.shape[0]*beilv
    W=m.shape[1]
    L=m.shape[2]*beilv
    m2=np.zeros((H,W,L),int)
    for n in range(W):
        section=m[:,n,:]
        sections=sectionex(section,H,L)
        m2[:,n,:]=sections
    W=m.shape[1]*beilv
    m3=np.zeros((H,W,L),int)
    for n in range(L):
        section=m2[:,:,n]
        sections=sectionex(section,H,W)
        m3[:,:,n]=sections
    return m3
   
def simgridex2(m,H,W,L):#beilv为放大或者缩小的倍率

    m2=np.zeros((H,m.shape[1],L),int)
    for n in range(m.shape[1]):
        section=m[:,n,:]
        sections=sectionex(section,H,L)
        m2[:,n,:]=sections
    m3=np.zeros((H,W,L),int)
    for n in range(L):
        section=m2[:,:,n]
        sections=sectionex(section,H,W)
        m3[:,:,n]=sections
    return m3
'''
def simgridex(m,beilv):
    H0=m.shape[0]
    W0=m.shape[1]
    L0=m.shape[2]

    dstHeight = int(H0*beilv)
    dstWidth = int(W0*beilv)
    dstLength= int(L0*beilv)
    
    dst = np.zeros((dstHeight, dstWidth,dstLength), int)
    for i in range(0, dstHeight):
        for j in range(0, dstWidth):
            for k in range(0, dstLength):
                iNew = int(i*(H0*1.0 / dstHeight))
                jNew = int(j*(W0*1.0/dstWidth))
                kNew = int(k*(L0*1.0/dstLength))
                print(dst.shape,iNew,jNew,kNew)
                dst[i, j,k] = m[iNew, jNew,kNew]
    return dst

def simgridex2(m,H,W,L):
    H0=m.shape[0]
    W0=m.shape[1]
    L0=m.shape[2]

    dstHeight = int(H)
    dstWidth = int(W)
    dstLength= int(L)
    
    dst = np.zeros((dstHeight, dstWidth,dstLength), int)
    for i in range(0, dstHeight):
        for j in range(0, dstWidth):
            for k in range(0, dstLength):
                iNew = int(i*(H0*1.0 / dstHeight))
                jNew = int(j*(W0*1.0/dstWidth))
                kNew = int(k*(L0*1.0/dstLength))
                dst[i, j,k] = m[iNew, jNew,kNew]
    return dst
########################坐标导入##############################################     
