#!/usr/bin/env python
# coding: utf-8

# In[22]:


#pycaret-automl
from pycaret.datasets import get_data
import pandas as pd


# In[23]:


import os


# In[26]:


#loading the data
with open("C:/Users/bhara/Desktop/a1/case.text",encoding="utf8") as f:
    lines = f.readlines()


# In[27]:


#data preprocessing
data= pd.DataFrame(lines)


# In[28]:


data=data.rename(columns={0:'discussion'})


# In[29]:


data


# In[30]:


#creating the model
import spacy


# In[31]:


from pycaret.nlp import *


# In[32]:


nlp1=setup(data,target='discussion')


# In[50]:


lda=create_model('lda')


# In[34]:


print(lda)


# In[35]:


df_lda=assign_model(lda)


# In[36]:


df_lda


# In[57]:


plot_model(lda,plot='wordcloud',topic_num='Topic 0')


# In[58]:


plot_model(lda,plot='topic_model')


# In[47]:



plot_model(lda,plot='frequency',topic_num='Topic 2')


# In[54]:



plot_model(lda,plot='tsne',topic_num='Topic 2')


# In[49]:



plot_model(lda,plot='umap',topic_num='Topic 2')


# In[42]:


evaluate_model(lda)


# tuned_lda = tune_model(model = 'lda')

# In[59]:


tuned_lda = tune_model(model = 'lda')


# In[ ]:




