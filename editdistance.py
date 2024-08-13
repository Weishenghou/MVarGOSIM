#!/usr/bin/env python
# coding: utf-8

# In[2]:


import Levenshtein
from scipy.cluster.vq import *
#from scipy.misc import imresize
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
import random
import os
import sys
from tvtk.api import tvtk, write_data 

import time
import multiprocessing
import matplotlib.pyplot as plt

def edit_distance(drill1,drill2):
    np.set_printoptions(threshold=np.inf)  
    len1 = drill1.shape[0]
    len2 = drill2.shape[0]
    dp = np.zeros((len1 + 1,len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i    
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if drill1[i-1] == drill2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]

def hamming_distance(drill1,drill2):#计算汉明距离
    vector1 = np.mat(drill1)
    vector2 = np.mat(drill2)

    vector3 = vector1-vector2

    #print("vector3 = vector1-vector2",vector3)

    smstr = np.nonzero(vector1-vector2);
    return np.shape(smstr[0])[0]

