import pandas as pd
from numpy import percentile
from sklearn.preprocessing import LabelEncoder

"""
Data class for Salary prediction case study

"""

class Data:
    
    def __init__(self, train_feature_file, train_target_file, test_file, cat_cols, num_cols, target_col, label_encode=True):
        
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.target_col = target_col
        self.feature_cols = cat_cols + num_cols
        self.train_df = self._create_train_df(train_feature_file, train_target_file, label_encode)
        self.test_df = self._create_test_df(test_file, label_encode)
        
    def _create_train_df(self, features, target, label_encode=True):
    
        """ Creates and returns the training dataframe """
        
        train_feature_df = self._load_data(features)
        train_target_df = self._load_data(target)
        train_df = self._merge_df(train_feature_df, train_target_df)
        train_df = self._clean_data(train_df)
        
        if label_encode:
            self._labelEncode(train_df, self.cat_cols)
        return train_df
    
    def _create_test_df(self, test, label_encode=True):
        
        """ Creates and returns the testing dataframe """
        
        test_df = self._load_data(test)
        if label_encode:
            self._labelEncode(test_df, self.cat_cols)
        return test_df
    
    
    def _load_data(self, filename):
        return pd.read_csv(filename)
    
    
    def _merge_df(self, df1, df2):        
        return df1.merge(df2, on='jobId', how='inner')
    
    
    def _clean_data(self, df):
        df = df.drop_duplicates(subset="jobId")
        #df = df.drop("jobId", axis=1)
        df = self._remove_outliers(df)    
        return df
    
    
    def _remove_outliers(self, df):
        quartiles = percentile(df['salary'], [25, 75])
        Q1 = quartiles[0]
        Q3 = quartiles[1]
        
        IQR = Q3 - Q1
        # We aren't removing the outliers above upperbound as decided while exploring in 02_DataExploration notebook
        df = df[df.salary > (Q1 - (1.5 * IQR))]
        return df
    

    def _labelEncode(self, df, cols):
        for col in cols:
            df[col] = LabelEncoder().fit_transform(df[col])
            
    def _reverseLabelEncode(self, df, cols):
        for col in cols:
            df[col] = LabelEncoder().inverse_transform(df[col])


