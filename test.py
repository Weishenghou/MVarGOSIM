#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

m2=np.arange(1000)
m2=m2.reshape((10,10,10))
for z in range(3, m2.shape[0] - 3):
    print(z)