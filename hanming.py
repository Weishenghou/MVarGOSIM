
import numpy as np
import matplotlib.pylab as plt
import cv2
import os
from zuobiao import*
from tvtk.api import tvtk, write_data 

##提取钻钻孔地层底面
'''
data=np.load('./output/reconstruction.npy')
x=[150,250,150,250,150,250,150,250]
y=[50,50,150,150,250,250,350,350]
[a,b,c]=data.shape

data=data[:160,y[7],x[7]]
b=0
for i in range(160):
  if data[i]!=data[i-1]:
    print(data[i-1],i-b)
    b=i
print("161",160-b) 

'''

##提取钻孔转vtk
'''
data=np.load('./output/reconstruction.npy')
x=[150,250,150,250,150,250,150,250]
y=[50,50,150,150,250,250,350,350]
[a,b,c]=data.shape

path2=np.zeros((a,b,c))
for z in range(len(x)):
  for i in range(2):
    for j in range(2):
      path2[:,y[z]+i,x[z]+j]=data[:,y[z],x[z]]

np.save('./output/zk.npy',path2)  
data=path2.transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape) 
grid.point_data.scalars = np.ravel(data,order='F') 
grid.point_data.scalars.name = 'lithology' 
write_data(grid, './output/zk.vtk') 

'''

#实虚钻孔对比#
def s(b,c):
 for i in range(c):
  a.append(b)

data=np.load('./output/reconstruction.npy')
x=[50,50,150,150,250,250,350,350]
y=[150,250,150,250,150,250,150,250]

#MKZ2-A099
a=[]
s(20,8)
s(50,5)
s(60,8)
s(70,8)
s(80,4)
s(90,16)
s(100,2)
s(90,9)
s(100,3)
s(110,14)
s(120,15)
s(100,10)
s(110,14)
s(120,14)
s(161,31)

Sum=0
for i in range(160):
 if a[i]!=data[i,x[0],y[0]]:
  Sum+=1
print(Sum)

#MKZ3-QFC-25
a=[]
s(20,8)
s(40,14)
s(50,18)
s(60,7)
s(70,7)
s(80,4)
s(90,7)
s(60,4)
s(70,8)
s(80,4)
s(90,16)
s(100,4)
s(110,15)
s(120,13)
s(161,31)

Sum=0
for i in range(160):
 if a[i]!=data[i,x[1],y[1]]:
  Sum+=1
print(Sum)

#MKZ3-QFC-56
a=[]
s(20,10)
s(50,3)
s(60,8)
s(70,7)
s(80,5)
s(90,29)
s(100,4)
s(110,13)
s(120,6)
s(90,12)
s(100,4)
s(110,14)
s(120,14)
s(161,31)

Sum=0
for i in range(160):
 if a[i]!=data[i,x[2],y[2]]:
  Sum+=1
print(Sum)

#MKZ3-QFC-S38
a=[]
s(20,7)
s(40,6)
s(50,20)
s(60,7)
s(70,10)
s(80,4)
s(50,8)
s(60,7)
s(70,7)
s(80,5)
s(90,16)
s(100,4)
s(110,14)
s(120,14)
s(161,31)

Sum=0
for i in range(160):
 if a[i]!=data[i,x[3],y[3]]:
  Sum+=1
print(Sum)

#MKZ3-QFC-S44（观1）
a=[]
s(20,10)
s(50,4)
s(60,7)
s(70,8)
s(80,4)
s(90,13)
s(70,6)
s(80,4)
s(90,17)
s(100,5)
s(90,17)
s(100,4)
s(110,13)
s(120,16)
s(161,33)

Sum=0
for i in range(160):
 if a[i]!=data[i,x[4],y[4]]:
  Sum+=1
print(Sum)

#MKZ3-QFC-S44（观2）
a=[]
s(20,10)
s(40,12)
s(50,17)
s(60,8)
s(50,14)
s(60,8)
s(70,7)
s(80,4)
s(90,17)
s(100,4)
s(110,14)
s(120,14)
s(161,31)

Sum=0
for i in range(160):
 if a[i]!=data[i,x[5],y[5]]:
  Sum+=1
print(Sum)

#MKZ3-WF-B07
a=[]
s(20,8)
s(50,16)
s(60,7)
s(70,8)
s(80,4)
s(50,15)
s(60,8)
s(70,8)
s(80,4)
s(90,17)
s(100,4)
s(110,15)
s(120,14)
s(161,32)

Sum=0
for i in range(160):
 if a[i]!=data[i,x[6],y[6]]:
  Sum+=1
print(Sum)

#WFZ-02
a=[]
s(20,7)
s(50,7)
s(31,7)
s(40,16)
s(50,16)
s(60,8)
s(70,8)
s(80,4)
s(90,16)
s(100,4)
s(110,22)
s(120,14)
s(161,31)

Sum=0
for i in range(160):
 if a[i]!=data[i,x[7],y[7]]:
  Sum+=1
print(Sum)


