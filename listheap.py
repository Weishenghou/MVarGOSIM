#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import heapq
def getListMaxNumIndex(num_list,topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
    return list(max_num_index)

def getListMinNumIndex(num_list,topk=3):
    '''
    获取列表中最小的前n个数值的位置索引
    '''
    min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
    return list(min_num_index)

