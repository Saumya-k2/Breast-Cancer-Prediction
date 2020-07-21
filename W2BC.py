#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
df=pd.read_csv(r"C:\Users\Saumya\Desktop\BC\datasets_180_408_data.csv")
print(df)


# In[ ]:





# In[13]:


df.head()


# Dependent Variable: 1 (diagnosis) and
# Independent Varible: all other column id,radius_worst,texture_mean,......
# 

# statistical parameters: 
#     Mean value,Standard deviation,perimeter,area,Standard Error,Extreme value/worst, 
#     smoothness,compactness,concavity,symmetry,fractal_dimension
#     

# In[11]:


df.shape


# In[20]:


df.isna().sum()


# In[21]:


df["diagnosis"].value_counts()


# In[25]:


from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.head()


# In[ ]:


import seaborn as sns
sns.pairplot(df,vars=["radius_mean","texture_mean","perimeter_mean","area_mean"],hue="perimeter_mean")


# In[33]:


df.corr(method="pearson")


# strong corelation between as texture_mean is increasing , smoothness_mean is decreasesing
# weak correlation between smoothness_mean and texture_mean bcz I can decrement of texture mean with decrement of smoothness_mean.
# same with Fractal_dimension and area/perimeter .

# In[24]:


sns.heatmap(df.corr() ,annot=True,fmt="0.0%")


# In[ ]:




