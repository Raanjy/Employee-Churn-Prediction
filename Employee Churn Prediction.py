#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.system('pip freeze > requirements.txt')




import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns


# In[2]:


hr = pd.read_csv("Hr_dataset.csv")


# In[3]:


hr.head(5)


# In[4]:


hr.tail()


# In[5]:


hr.shape


# In[6]:


hr.describe().T


# In[7]:


hr.isnull().sum()


# In[8]:


hr.duplicated().value_counts()


# In[9]:


hr.info()


# In[10]:


## converting Object into Int64


# In[11]:


hr.columns=hr.columns.str.replace(" ","_")
hr.columns


# In[12]:


hr.columns


# In[13]:


hr_dummies = pd.get_dummies(hr)
hr_dummies.head()


# In[14]:


##hr_dummies = pd.get_dummies(hr['salary'])
##hr_dummies.head()


# In[15]:


hr_dummies.head()


# In[16]:


hr_dummies.drop_duplicates().shape


# In[17]:


hr=hr_dummies.drop_duplicates()


# In[18]:


hr.shape


# In[19]:


#Exploratory Data Analysis


# In[20]:


##sns.histplot(hr)


# In[21]:


hr.corr()


# In[22]:


plt.rcParams["figure.figsize"] = (20,9)


# In[23]:


sns.heatmap(hr.corr(), annot=True,linewidths=2, linecolor='Black')


# In[24]:


plt.scatter(hr['number_project'], hr['average_montly_hours'])
plt.xlabel("No of Projects")
plt.ylabel("Avg Monthly Hours")


# In[25]:


plt.scatter(hr['number_project'], hr['last_evaluation'])
plt.xlabel("No of Projects")
plt.ylabel("Last Evaluation")


# In[26]:


plt.scatter(hr['salary_low'], hr['salary_medium'])
plt.xlabel("Salary Low")
plt.ylabel("Salary Medium")


# In[27]:


sns.regplot(x='number_project', y='average_montly_hours',data = hr)


# In[28]:


## Outlier Treatment


# In[29]:


sns.boxplot(hr, x='average_montly_hours', orient= 'v')


# In[30]:


sns.boxplot(hr, x='number_project', orient= 'V')


# In[31]:


## Finding Outliers


# In[32]:


##sns.scatterplot(hr)


# In[33]:


##sns.pairplot(hr)


# In[34]:


##Standardization


# In[35]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[36]:


##hr_scaled


# In[37]:


##hr_scaled


# In[38]:


## drop and Pop


# In[39]:


x= hr.drop(["left"], axis=1)


# In[40]:


y=hr.pop('left')


# In[41]:


##Splitting Dataset


# In[42]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3, random_state=0 )


# In[43]:


x.shape


# In[44]:


y.shape


# In[45]:


xtrain.head(5)


# In[46]:


xtest.head(5)


# In[47]:


ytrain.head(5)


# In[48]:


ytest.head(5)


# In[49]:


## Standardizing the Data 


# In[50]:


hr_scale= StandardScaler()
hr_scaled_xtrain = hr_scale.fit_transform(xtrain)
hr_scaled_xtrain


# In[51]:


hr_scaled_xtest = hr_scale.transform(xtest)
hr_scaled_xtest


# In[52]:


## Model Building


# In[53]:


from sklearn.neural_network import MLPClassifier


# In[54]:


##hr_mlp = MLPClassifier()


# In[55]:


##hr_fit = hr_mlp.fit(xtrain, ytrain)


# In[56]:


##hr_fit


# In[57]:


from sklearn.linear_model import LogisticRegression


# In[58]:


hr_reg = LogisticRegression()


# In[59]:


hr_fit = hr_reg.fit(xtrain, ytrain)


# In[60]:


## Parameters


# In[61]:


hr_fit.get_params()


# In[62]:


##Prediction


# In[63]:


hr_pred = hr_fit.predict(xtest)
hr_pred


# In[64]:


##plot a scatter plot for the prediction


# In[65]:


plt.scatter(ytest,hr_pred)


# In[66]:


## residuals


# In[67]:


residuals = ytest-hr_pred
residuals


# In[68]:


##plot residuals vs 


# In[69]:


sns.distplot(residuals, kde = True)


# In[70]:


## scatter plot with predictions and residuals


# In[71]:


plt.scatter(hr_pred, residuals)


# In[72]:


## Performance Metrics


# In[73]:


##from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[74]:


##print(mean_absolute_error(ytest,hr_pred))
##print(mean_squared_error(ytest,hr_pred))
##print(np.sqrt(mean_squared_error(ytest,hr_pred)))


# In[75]:


# R Square and Adjusted Square - For Regression Model


# In[ ]:





# In[76]:


## Accuracy scores 


# In[77]:


from sklearn.metrics import accuracy_score, roc_auc_score,auc, classification_report,confusion_matrix


# In[78]:


hr_acuracy = accuracy_score(ytest, hr_pred)
hr_acuracy 


# In[79]:


hr_matrix = confusion_matrix(ytest, hr_pred)
hr_matrix 


# In[80]:


hr_creport= classification_report(ytest,hr_pred)
hr_creport


# In[81]:


## ROC Score and AUC curve


# In[82]:


hr_auc = roc_auc_score(ytest,hr_pred)
hr_auc


# In[83]:


hr_roc = roc_curve(ytest,hr_pred)
hr_roc 


# In[84]:


##Save


# In[85]:


import pickle


# In[86]:


with open("Final_model.pkl", 'wb') as f:
    pickle.dump(hr_fit,f)


# In[87]:


with open("Final_model.pkl", 'rb') as f:
    f_model = pickle.load(f)


# In[88]:


f_model


# In[89]:


## Predict


# In[90]:


#newarray = np.array(f_model)
#newarray
#newarray = newarray.reshape(1,1,1)
#newarray


# In[91]:


print(f_model.predict([1,0.6,0.7,0.3,0.4,0.5,1,0.5,0.6,0.7,0.3,0.4,0.5,1,0.5,0.7,0,1,0.3]))


# In[92]:


pip freeze > requirements.txt


# In[93]:


conda install scikit-learn


# In[94]:


with open("hr.pkl", 'wb') as f:
    pickle.dump(hr,f)


# In[95]:


with open("hr.pkl", 'rb') as f:
    data = pickle.load(f)


# In[96]:


data.head()

