import numpy as np 
from tvtk.api import tvtk, write_data 
from zuobiao import*
from AIinitial import *


def extendTimodel(m,template_h,template_x,template_y):#全自动拓展插入硬数据的待模拟网格
    lag=max(template_h,template_x,template_y)//2
    #lag=max(template_h,template_x,template_y)//2
    m2=np.pad(m,lag,'edge')
    d=[]
    for h in range(lag,m2.shape[0]-lag):
            for x in range(lag,m2.shape[1]-lag):
                for y in range(lag,m2.shape[2]-lag):
                    d.append((h,x,y))
    
    for cc in range(lag):
        #random.shuffle(d)
        flag=0
        for n in range(len(d)):
            h=d[n][0]
            x=d[n][1]
            y=d[n][2]
            if m2[h,x,y]==-1:
                value=extend2dAI(m2,h,x,y)
                flag=-1
                if value!=-1 and value!=0:
                    #print value
                    m[h-lag,x-lag,y-lag]=value

        if flag==0:
            break
        m2=np.pad(m,lag,'edge')
        #填充为1的    
    return m
c=np.load('./output/output1219.npy')
data = np.load("./output/reconstruction.npy")

print(c.shape,data.shape)

#data=simgridex(data,2)

print('done')
flag=0
while flag==0:
    for x in range(data.shape[1]):
        for y in range(data.shape[2]):
            for h in range(1,data.shape[0]):
                if c[h,x,y]!=0 and data[h,x,y]==0:
                   data[h,x,y]=-1
    extendTimodel(data,3,3,3)
    flag=1   
    print('extend done')
    count=0
    for x in range(data.shape[1]):
        for y in range(data.shape[2]):
            for h in range(1,data.shape[0]):
                if data[h,x,y]==-1:
                   flag=0
                   count=count+1
                   print(count)
#

'''
for x in range(data.shape[1]):
    for y in range(data.shape[2]):
        for h in range(1,data.shape[0]):

            if data[h,x,y]==255:
               data[h,x,y]=-1
'''
#extendTimodel(data,40,40,40)



for x in range(data.shape[1]):
    for y in range(data.shape[2]):
        for h in range(data.shape[0]):

            if c[h,x,y]==0:
               data[h,x,y]=0
            if c[h,x,y]==170:
               data[h,x,y]=255
np.save('./output/st.npy',data)
print('done')

data=data.transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape) 

grid.point_data.scalars = np.ravel(data,order='F') 
grid.point_data.scalars.name = 'lithology' 

# Writes legacy ".vtk" format if filename ends with "vtk", otherwise 
# this will write data using the newer xml-based format. 
write_data(grid, './output/stt.vtk') 

