#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('insurance.csv')


# # EDA

# In[3]:


df.head()


# In[4]:


df.duplicated().sum()


# In[5]:


df[df.duplicated(keep=False)]


# In[6]:


df.isna().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.dtypes


# In[10]:


df.info


# In[11]:


df.info()


# In[12]:


df[df['age']<20]


# In[13]:


df[df['bmi']>28]


# In[14]:


df[df['bmi']<23]


# In[15]:


df1=df.drop(['sex','smoker','region'],axis=1)
df1


# In[16]:


df.pivot_table(index='sex',values='expenses',aggfunc=['sum','count','mean','min','max'])


# #  DATA VISUALIZATION

# In[17]:


plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x='sex',y='expenses',hue='sex')


# In[18]:


plt.figure(figsize=(8,6))
sns.barplot(data=df,x='sex',y='expenses',hue='sex')


# In[19]:


plt.figure(figsize=(8,6))
sns.boxplot(data=df,x='age',y='expenses')
plt.title('age')


# In[20]:


plt.figure(figsize=(6,5))
sns.boxplot(data=df,y='age')


# In[21]:


plt.figure(figsize=(8,8))
sns.boxplot(data=df,x='sex',y='expenses')
plt.title('sex vs expenses')


# In[22]:


plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x='age',y='expenses',hue='sex')


# In[23]:


f,ax=s=plt.subplots(1,1,figsize=(8,6))
ax=sns.histplot(data=df,x='expenses',hue='sex',kde=True)


# In[24]:


df.head().loc[:,['sex','expenses']]


# In[25]:


df.pivot_table(index='age',values='expenses',aggfunc=['sum','count','mean','min','max'])


# In[26]:


df.pivot_table(index='bmi',values='expenses',aggfunc=['sum','count','mean','min','max'])


# In[27]:


df.pivot_table(index='sex',values='expenses',aggfunc=['sum','count','mean','min','max'])


# In[28]:


f,ax=plt.subplots(1,1,figsize=(8,6))
ax=sns.scatterplot(data=df,x='bmi',y='expenses',hue='sex')
plt.title('bmi wrt expenses')
plt.show()


# In[29]:



ax=plt.subplots(1,1,figsize=(8,8))
ax=sns.distplot(df['expenses'],kde=True,color='c')


# In[30]:


expenses=df['expenses'].groupby(df.sex).sum().sort_values(ascending=True)
f,ax=plt.subplots(1,1,figsize=(6,6))
ax=sns.barplot(expenses.head(),expenses.head().index)


# In[31]:


f,ax=plt.subplots(1,1,figsize=(8,6))
ax=sns.barplot(x='region',y='expenses',hue='sex',data=df,palette='cool')


# In[32]:


f,ax=plt.subplots(1,1,figsize=(8,6))
ax=sns.barplot(x='region',y='expenses',hue='smoker',data=df)


# In[33]:


plt.figure(figsize=(8,6))
sns.barplot(data=df,x='region',y='expenses',hue='smoker')


# In[34]:


f,ax=plt.subplots(1,1,figsize=(8,6))
ax=sns.barplot(x='region',y='expenses',hue='children',data=df,palette='Set1')


# In[35]:


plt.figure(figsize=(8,6))
sns.barplot(x='region',y='expenses',hue='children',data=df,palette='Set1')


# In[36]:


ax=sns.lmplot(data=df,x='age',y='expenses',hue='smoker',palette='Set1')

ax=sns.lmplot(data=df,x='bmi',y='expenses',hue='smoker',palette='Set2')

ax=sns.lmplot(data=df,x='children',y='expenses',hue='smoker',palette='Set3')


# In[37]:


f,ax=plt.subplots(1,1,figsize=(10,8))
ax=sns.violinplot(x='children',y='expenses',data=df,orient='v',hue='smoker',palette='inferno')


# In[38]:


#converting objects label into categorical

df[['sex','smoker','region']]=df[['sex','smoker','region']].astype('category')
df.dtypes


# In[39]:


#converting category labels into numerical laabels using LabelEncoder


# In[40]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()

label.fit(df.sex.drop_duplicates())
df.sex=label.transform(df.sex)

label.fit(df.smoker.drop_duplicates())
df.smoker=label.transform(df.smoker)

label.fit(df.region.drop_duplicates())
df.region=label.transform(df.region)
df.dtypes


# In[41]:


f,ax=plt.subplots(1,1,figsize=(10,10))
ax=sns.heatmap(df.corr(),annot=True,cmap='cool')


# # Linear Regression

# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[43]:


x=df.drop(['expenses'],axis=1)
y=df['expenses']


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[45]:


x_train.head()


# In[46]:


x_test.head()


# In[47]:


y_train.head()


# In[48]:


y_test.head()


# In[49]:


x_train.shape


# In[50]:


x_test.shape


# In[51]:


y_train.shape,y_test.shape


# In[52]:


lr=LinearRegression()


# In[53]:


lr.fit(x_train,y_train)


# In[54]:


y_pred=lr.predict(x_test)


# In[55]:


y_pred[:5]


# In[56]:


lr.intercept_


# In[57]:


lr.coef_


# In[58]:


lr.score(x_test,y_test)


# In[59]:


y_test[:5]


# In[60]:


df1=pd.DataFrame({'Actual':y_test,'Predict':y_pred,'dif':y_test-y_pred})
df1


# # Ridge Regression

# In[61]:


from sklearn.linear_model import Ridge
Ridge=Ridge(alpha=0.5)

Ridge.fit(x_train,y_train)


# In[62]:


Ridge.intercept_


# In[63]:


Ridge.coef_


# In[64]:


Ridge.score(x_test,y_test)


# In[65]:


y_pred_Ridge=Ridge.predict(x_test)


# In[66]:


y_pred_Ridge[:5]


# In[67]:


y_test[:5]


# # Lasso Regression

# In[68]:


from sklearn.linear_model import Lasso
Lasso=Lasso(alpha=0.2,fit_intercept=True,normalize=False,precompute=False,max_iter=1000,tol=0.0001,warm_start=False,
            positive=False,random_state=None,selection='cyclic')


# In[69]:


Lasso.fit(x_train,y_train)


# In[70]:


Lasso.intercept_


# In[71]:


Lasso.coef_


# In[72]:


Lasso.score(x_test,y_test)


# # Random Forest Regressor

# In[73]:


from sklearn.ensemble import RandomForestRegressor as rfr


# In[74]:


x=df.drop(['expenses'],axis=1)


# In[75]:


y=df.expenses


# In[76]:


Rfr=rfr(n_estimators=100,criterion='mse',random_state=1,n_jobs=-1)


# In[77]:


Rfr.fit(x_train,y_train)


# In[78]:


x_train_pred=Rfr.predict(x_train)


# In[79]:


x_train_pred[:5]


# In[80]:


y_train[:5]


# In[81]:


x_test_pred=Rfr.predict(x_test)


# In[82]:


x_test_pred[:5]


# In[83]:


y_test[:5]


# In[84]:


df3=pd.DataFrame({'Actual':y_test,'Predict':x_test_pred,'difference':y_test-x_test_pred})
df3


# In[85]:


('MSE train data :%.3f,MSE test data:%.3f'%
(metrics.mean_squared_error(x_train_pred,y_train),
metrics.mean_squared_error(x_test_pred,y_test)))


# In[86]:


('R2 train data:%.3f,R2 test data:%.3f'%
(metrics.r2_score(y_train,x_train_pred,y_train),
metrics.r2_score(y_test,x_test_pred,y_test)))


# In[ ]:




