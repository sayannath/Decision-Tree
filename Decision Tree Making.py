#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


print(os.listdir())


# In[3]:


# importing required libraries
import numpy as np
import pandas as pd


# In[4]:


# Load the Data Set from the csv file
dataSet = pd.read_csv('groupStudy.csv')


# In[5]:


dataSet


# In[6]:


# selecting the row and columns to be used
hours = dataSet['Hours of study']
marks = dataSet['Marks scored']


# In[7]:


# reshaping the matrix
X = np.array(hours).reshape(-1, 1)
y = np.array(marks).reshape(-1, 1)


# In[8]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)
y_predicted  = regressor.predict(np.array(5).reshape(-1, 1))


# In[9]:


y_predicted

