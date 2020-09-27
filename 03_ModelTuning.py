#!/usr/bin/env python
# coding: utf-8

# ## Model Tuning
# 
# After trials with other algorithms, it's been decided and let's tune the hyperparameters of the GBR model.

# In[1]:


import sys
sys.path.insert(0 , './scripts')

from data import Data
from featuregenerator import FeatureGenerator

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# In[2]:


train_feature_file = "raw_data/train_features.csv"
train_target_file = "raw_data/train_salaries.csv"
test_file = "raw_data/test_features.csv"
cat_cols = ['companyId', 'jobType', 'degree', 'major', 'industry']
num_cols = ['yearsExperience', 'milesFromMetropolis']
target_col = 'salary'


# In[3]:


data = Data(train_feature_file, train_target_file, test_file, cat_cols, num_cols, label_encode=True)

X = data.train_df[data.feature_cols]
y = data.train_df[target_col]


# In[8]:


gbr = GradientBoostingRegressor(learning_rate=0.1)

grid_search_results = GridSearchCV(gbr, param_grid={'n_estimators' : [70,80,90,100], 'max_depth' : [4,5,6]}, n_jobs=2, verbose=5, cv=2)
grid_search_results.fit(X, y)


# In[9]:


grid_search_results.best_params_


# Final Parameters
# So, the final parameters we have settled on are:
# 
#     n_estimators  = 100
#     learning_rate = 0.1
#     max_depth     = 6
#     loss          = 'ls'
#     subsample     =  1

# In[5]:


model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=6)
results = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=2, n_jobs=2, verbose=5, error_score='raise')
print("MSE : %.3f (%.3f)" % (results.mean(), results.std()))


# ## FeatureEngineering 

# In[4]:


featuregenerator = FeatureGenerator(data, target_col)
featuregenerator.add_group_stats()


# In[8]:


X = data.train_df[data.feature_cols]
y = data.train_df[target_col]


# In[7]:


model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=6)
results = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=2, n_jobs=2, verbose=5, error_score='raise')
print("MSE : %.3f (%.3f)" % (results.mean(), results.std()))


# - With no feature engineering, the MSE was about __~-358__.
# - So now it looks pretty good improvement in model performance by about __13% decrease in MSE to ~-310__ with __feature engineering__ involved.

# Let's move ahead and fit and save the final model in the next notebook.
