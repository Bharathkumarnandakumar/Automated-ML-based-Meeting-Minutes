#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.datasets import get_data
import pandas as pd


# In[2]:


import os


# In[3]:


with open("C:/Users/bhara/Desktop/a1/case.text",encoding="utf8") as f:
    lines = f.readlines()


# In[4]:


data= pd.DataFrame(lines)


# In[5]:


data=data.rename(columns={0:'discussion'})


# In[6]:


data


# In[7]:


import spacy


# In[8]:


from pycaret.nlp import *


# In[9]:


nlp1=setup(data,target='discussion')


# In[11]:


hdp=create_model('hdp')


# In[12]:


print(hdp)


# In[13]:


df_hdp=assign_model(hdp)


# In[15]:


df_hdp


# In[19]:


plot_model(hdp,plot='wordcloud',topic_num='Topic 3')


# In[17]:


evaluate_model(hdp)


# In[18]:


tuned_lda = tune_model(model = 'hdp')


# In[ ]:




