import numpy as np
import matplotlib.pylab as plt
import cv2
import os
from tvtk.api import tvtk, write_data 

##检查模型奇异值
'''
path1=np.load('./output/soft.npy')

[a,b,c]=path1.shape
s=0
for z in range(a):
  for i in range(b):
    for j in range(c):
      if path1[z,i,j]==210:
        print(z,i,j)
        s+=1
print(s)        

'''

##提取钻钻孔地层底面
'''
data=np.load('./zk.npy')
x=np.array([24250,24268,24274,24280])
y=np.array([40639,40637,40657,40640])
x=(x-24236)*2
y=(y-40628)*2


data=data[:120,x[2],y[2]]
b=0
for i in range(120):
  if data[i]!=data[i-1]:
    print(data[i-1],i-b)
    b=i
print("60",120-b) 

'''

##改奇异值
'''
path=np.load('./output/soft.npy')

[a,b,c]=path.shape

for z in range(a):
  for i in range(b):
    for j in range(c):
      if path[z,i,j]==185:
        path[z,i,j]=60
      if path[z,i,j]==210:
        path[z,i,j]=45
      if path[z,i,j]==220:
        path[z,i,j]=60
np.save('./output/soft.npy',path)      
   '''

def zb(m,sx):
  s=m.shape[0]*m.shape[1]*m.shape[2]
  Sum=0
  for z in range(m.shape[0]):
    for x in range(m.shape[1]):
      for y in range(m.shape[2]):
        if m[z,x,y]==sx:
          Sum=Sum+1
  return (Sum*100)/s

def zb1(m,sx):
  s=m.shape[0]*m.shape[1]
  Sum=0
  for x in range(m.shape[0]):
    for y in range(m.shape[1]):
      if m[x,y]==sx:
        Sum=Sum+1
  return (Sum*100)/s
           
##占比##
'''
ls=[195,100,120,70,40,141,161,45,60,240,80]
m=np.load('./reconstruction1.npy')
for i in range(len(ls)):
  print(zb(m,ls[i]))

for n in range(10):
  print(" ")
  path='./Ti1/'+str(n+1)+'.bmp'
  section=cv2.imread(path,0)          
  for i in range(len(ls)):
    print(zb1(section,ls[i]))
'''


##连通函数##
m=np.load('./output/reconstruction.npy')
#m=np.load('./output/outputinitial.npy')
print(m.shape)
#dx=[50,100,150,200,250,300,350,400]
#dy=[50,100,150,200,250,300,350,400]
dz=[2,5,10]
d=dz
for i in range(len(d)):
  Sum=0
  s=0
  for z in range(m.shape[0]-d[i]):
    for y in range(m.shape[1]):
      for x in range(m.shape[2]):
        if m[z+d[i],y,x]==60 and m[z,y,x]==m[z,y,x]:
          s=s+1
        if m[z+d[i],y,x]!=-1 and m[z,y,x]!=-1:
          Sum=Sum+1        
  print("%.2f" %(s/Sum))     


'''
##变差函数##
m=np.load('./output/reconstruction.npy')
#m=np.load('./output/outputinitial.npy')
print(m.shape)
#dx=[2]
#dy=[2,5,50,100,150,200,250,300,350,400]
dz=[5,20,40,60,80,100,120,140,160]
d=dz
for i in range(len(d)):
  Sum=0
  s=0
  for z in range(m.shape[0]-d[i]):
    for y in range(m.shape[1]):
      for x in range(m.shape[2]):
        if m[z,y,x]!=-1 and m[z+d[i],y,x]!=-1:
          Sum=Sum+1 
          a= (m[z+d[i],y,x]-m[z,y,x]) *(m[z+d[i],y,x]-m[z,y,x])*0.5 
          s=s+a     
  print("%.2f" %(s/Sum))  
'''         
 
