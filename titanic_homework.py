#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy 
import pandas as pd
from sklearn import preprocessing


# In[31]:


get_ipython().system('pip install sklearn')


# In[32]:


df=pd.read_csv('titanic3_3.csv')
df


# In[33]:


cols=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
df_titanic=df[cols]


# In[34]:


mskk=numpy.random.rand(len(df_titanic))<0.80  # 隨機產生0~1的數
mskk


# In[35]:


train_data = df_titanic[mskk]
len(train_data)


# In[36]:


test_data = df_titanic[~mskk]
len(test_data)


# In[37]:


~mskk


# In[38]:


train_data


# In[39]:


train_data.isnull().sum()


# In[40]:


def PreprocessData(raw_df):
    raw_df=raw_df.drop(['name'],axis=1) 
    raw_df=raw_df.drop(['parch'],axis=1) #
    age_mean=raw_df['age'].mean()        
    raw_df['age']=raw_df['age'].fillna(age_mean)
    fare_mean=raw_df['fare'].mean()     
    raw_df['fare']=raw_df['fare'].fillna(fare_mean)
    raw_df['sex']=raw_df['sex'].map({'female':0,'male':1})   
    raw_df=pd.get_dummies(data=raw_df,columns=["embarked"])
    
    array_titanic=raw_df.values 
    
    label=array_titanic[:,0]   
    feature=array_titanic[:,1:]
    
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))  
    scaledFeature = minmax_scale.fit_transform(feature)
    
    return scaledFeature,label


# In[41]:


train_feature,train_label = PreprocessData(train_data)


# In[42]:


train_feature


# In[69]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[70]:


model = Sequential()
model.add(Dense(units = 128,
                input_dim = 8,
                activation = 'relu'))


# In[71]:


model.add(Dropout(0.7))


# In[72]:


model.add(Dense(units = 60,
                activation = 'relu'))


# In[73]:


model.add(Dense(units = 1,
                activation = 'sigmoid'))


# In[74]:


model.summary()


# In[75]:


model.compile(loss='binary_crossentropy', 
             optimizer='adam',
             metrics=['accuracy'])


# In[76]:


train_history = model.fit(x=train_feature,
                         y=train_label,
                         validation_split=0.1,
                         epochs=1000,
                         batch_size=50)


# In[79]:


import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train_History')
  plt.ylabel('tain')
  plt.xlabel('Epoch')
  
show_train_history(train_history,'acc','val_acc')  #藍色:acc 橘色:val_acc


# In[80]:


show_train_history(train_history,'loss','val_loss')


# In[81]:


test_feature,test_label = PreprocessData(test_data)


# In[82]:


scores=model.evaluate(x = test_feature,y=test_label)
scores[1]


# In[83]:


pred=model.predict_classes(test_feature)
print(pred[:10])
print(test_label[:10])


# In[84]:


pred=model.predict(test_feature)
print(pred[:10])


# # 加入Jack & Rose

# In[ ]:


Jack=[0,'Jack',3,'male',23,1,0,5.0000,'S']
Rose=[1,'Rose',1,'female',20,1,0,100.0000,'S']


# In[ ]:


JR_df = pd.DataFrame([Jack,Rose],
                    columns=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked'])


# In[ ]:


df_titanic_JR = test_data.append(JR_df, ignore_index=True)
df_titanic_JR.isnull().sum()


# In[ ]:


df_titanic_JR


# In[ ]:


all_feature,all_label = PreprocessData(df_titanic_JR)


# In[ ]:


all_probability=model.predict(all_feature)
all_probability


# In[ ]:


df_titanic_JR.insert(len(df_titanic_JR.columns),
                  'probability',all_probability)


# In[ ]:


df_titanic_JR


# In[ ]:




