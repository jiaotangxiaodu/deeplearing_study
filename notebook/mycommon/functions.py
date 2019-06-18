#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def _gradient_vector(f,x,h):
    g = np.zero_like(x)
    for i,feature in enumerate(x):
        x[i] = feature + h
        y_high = f(x)
        x[i] = feature - h
        y_low = f(x)
        x[i] = feature
        g[i] = (y_high - y_low) / (2*h)
    return g


# In[7]:


def _gradient_matrix(f,x,h):
    return np.array([_gradient_vector(f,row,h) for row in x])


# In[16]:


def gradient(f,x,h=1e-6):
    x = np.array(x,dtype='float')
    if x.ndim == 1:
        return _gradient_vector(f,x,h)
    return _gradient_matrix(f,X,h)


# In[22]:


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


# In[57]:


def cross_entropy_error(y,t,delta=1e-8):
    return - t.dot(np.log(y+delta))


# In[ ]:




