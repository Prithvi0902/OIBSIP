#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris=load_iris()


# In[3]:


iris.data


# In[4]:


iris.target_names


# In[5]:


iris.feature_names


# In[6]:


x=iris.data


# In[7]:


y=iris.target


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=4)


# In[10]:


x_train


# In[11]:


x_test


# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# In[13]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[14]:


knn.fit(x_train,y_train)


# In[15]:


y_prd=knn.predict(x_test)
y_prd


# In[16]:


y_test


# In[17]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_prd)


# In[ ]:




