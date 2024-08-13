#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pylab as plt
import os
from tvtk.api import tvtk, write_data 




        


m=np.load('./output/study_area.npy')
m1=np.load('./output/outputinitial.npy')
for z in range(m.shape[0]):
  for y in range(m.shape[1]):
    for x in range(m.shape[2]):
      if m1[z,y,x]==-1:
        m[z,y,x]=-1   


print(m.shape)        
np.save('./output/initial_Vp.npy',m)    
data=m.transpose(-1,-2,0)#转置坐标系
grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0), dimensions=data.shape) 
grid.point_data.scalars = np.ravel(data,order='F') 
grid.point_data.scalars.name = 'lithology' 
write_data(grid, './output/initial_Vp.vtk') 



