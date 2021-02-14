#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pymongo
from pymongo import MongoClient

clien=pymongo.MongoClient("mongodb+srv://bharath_23:nanda8189N23@cluster0-l5wpk.mongodb.net/test?retryWrites=true&w=majority")


# In[9]:


mydb = clien["meeting_text"]
col = mydb["audio_text"]


# In[10]:


f = open("C:/Users/bhara/Desktop/convertedtext.txt")
text = f.read()
doc = {
"file_name": "C:/Users/bhara/Desktop/convertedtext.txt",
"contents" : text }
col.insert_one(doc)


# In[ ]:





# In[ ]:




