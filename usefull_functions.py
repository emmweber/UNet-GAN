#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os



# In[2]:


class optimization():
    def __init__(self,learning_rate=False):
        if learning_rate==True:
            self.hyper_learning_optimization()

    def hyper_learning_optimization(self,number,threshold=0,scale_factor=4, transposition_factor = 0):
            # scale factor==> from [0;1) to scale_factor*[0,1)
            # transposition factor ==>  (x-x0) with x0=trasposition factor ==> set max value that is achievable

            values = scale_factor*(np.random.rand(number))+transposition_factor

            learning_rates = np.power(10., -values)

            mask = learning_rates<threshold
            learning_rates=learning_rates[mask]


            return learning_rates



# In[3]:


class commonfunctions:

    def __init__(self):
        self.test=0

    def list_directory(self,directory):
        listdir = []
        for file in os.listdir(directory):
            direct = directory+'/'+file
            if os.path.isdir(direct):
                listdir.append(direct)
        return listdir


# In[ ]:




