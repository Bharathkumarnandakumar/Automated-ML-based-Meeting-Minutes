#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import IPython.display as ipd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa


# In[3]:


filelist = os.listdir("C:/Users/bhara/Desktop/data team") 
data = pd.DataFrame(filelist)
data['Class']=data
data=data.rename(columns={0:'ID'})
data['Class'][0:100]='bharath'
data['Class'][100:200]='chaithra'
data['Class'][200:300]='joshua'
data['Class'][300:400]='tanvi'


# df_male

# In[136]:


data


# In[137]:


data.to_csv(r'C:/Users/bhara/Desktop/data team.csv', index = False)


# In[138]:


data=pd.read_csv('C:/Users/bhara/Desktop/data team.csv')


# In[139]:


data


# In[101]:


mfc=[]
chr=[]
me=[]
ton=[]
lab=[]
for i in tqdm(range(len(data))):
    f_name="C:/Users/bhara/Desktop/data team/"+str(data.ID[i])
    X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
    mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T,axis=0)
    mfc.append(mf)
    l=data.Class[i]
    lab.append(l)
    try:
        t =    np.mean(librosa.feature.tonnetz(
                       y=librosa.effects.harmonic(X),
                       sr=s_rate).T,axis=0)
        ton.append(t)
    except:
        print(f_name)  
    m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T,axis=0)
    me.append(m)
    s = np.abs(librosa.stft(X))
    c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T,axis=0)
    chr.append(c)


# In[140]:


mfcc = pd.DataFrame(mfc)
mfcc.to_csv('C:/Users/bhara/Desktop/SAP Industrial Case Study Project/mfc.csv', index=False)
chrr = pd.DataFrame(chr)
chrr.to_csv('C:/Users/bhara/Desktop/SAP Industrial Case Study Project/chr.csv', index=False)
mee = pd.DataFrame(me)
mee.to_csv('C:/Users/bhara/Desktop/SAP Industrial Case Study Project/me.csv', index=False)
tonn = pd.DataFrame(ton)
tonn.to_csv('C:/Users/bhara/Desktop/SAP Industrial Case Study Project/ton.csv', index=False)
la = pd.DataFrame(lab)
la.to_csv('C:/Users/bhara/Desktop/SAP Industrial Case Study Project/labels.csv', index=False)


# In[103]:


features = []
for i in range(len(ton)):
    features.append(np.concatenate((me[i], mfc[i], 
                ton[i], chr[i]), axis=0))
features[:5]    


# In[104]:


la = pd.get_dummies(lab)
label_columns=la.columns #To get the classes
target = la.to_numpy() #Convert labels to numpy array


# In[105]:


tran = StandardScaler()
features_train = tran.fit_transform(features)
len(features_train)


# In[106]:


feat_train=features_train[:360]
target_train=target[:360]
y_train=features_train[360:380]
y_val=target[360:380]
test_data=features_train[380:]
test_label=target[380:]


# In[107]:


print("Training",feat_train.shape)
print(target_train.shape)
print("Validation",y_train.shape)
print(y_val.shape)
print("Test",test_data.shape)
print(test_label.shape)


# In[116]:


model = Sequential()

model.add(Dense(166, input_shape=(166,), activation = 'relu'))

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.6))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[117]:


history = model.fit(feat_train, target_train, batch_size=64, epochs=30, 
                    validation_data=(y_train, y_val))


# In[129]:



train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Set figure size.
plt.figure(figsize=(10, 7))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_acc, label='Training', color='black')
plt.plot(val_acc, label='Validation', color='red')

# Set title
plt.title('Training and Validation Accuracy', fontsize = 15)
plt.xlabel('Epoch', fontsize = 15)
plt.legend(fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.xticks(range(0,30,5), range(0,30,5));


# In[127]:


predict = model.predict_classes(test_data)


# In[120]:



# Output
predict


# In[121]:


label_columns


# In[122]:



# To match the labels
prediction=[]
for i in predict:
  j=label_columns[i]
  prediction.append(j)


# In[123]:


# Predicted Labels of test data
prediction


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




