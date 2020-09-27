#!/usr/bin/env python
# coding: utf-8

# ## Fit the Final Model 

# In[15]:


import sys
sys.path.insert(0, './scripts')

from data import Data 
from featuregenerator import FeatureGenerator

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import seaborn as sns


# ### Data

# In[2]:


train_feature_file = "raw_data/train_features.csv"
train_target_file = "raw_data/train_salaries.csv"
test_file = "raw_data/test_features.csv"
cat_cols = ['companyId', 'jobType', 'degree', 'major', 'industry']
num_cols = ['yearsExperience', 'milesFromMetropolis']
target_col = 'salary'

data = Data(train_feature_file, train_target_file, test_file, cat_cols, num_cols, target_col, label_encode=True)


# ### Feature Generation 

# In[3]:


featuregenerator = FeatureGenerator(data, target_col)
featuregenerator.add_group_stats()


# ### Train Data 

# In[4]:


train_X, train_y = data.train_df[data.feature_cols], data.train_df[target_col]


# ### Create Model Object 

# In[5]:


regressor = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=6)


# ### Fit Final Model 

# In[6]:


regressor.fit(train_X, train_y)


# ### Score Test Dataset  

# In[7]:


data.test_df.head()


# In[8]:


test_X = data.test_df[data.feature_cols]


# In[9]:


test_X.shape


# In[10]:


y_pred = regressor.predict(test_X)


# ### Feature Importance 

# In[17]:


importances = regressor.feature_importances_

feature_importances = pd.DataFrame({'feature' : data.feature_cols, 'importance' : importances})
feature_importances.sort_values(by='importance', inplace=True, ascending=False)
feature_importances.set_index('feature', inplace=True, drop=True)


# In[20]:


print(feature_importances)
feature_importances.plot.bar()

