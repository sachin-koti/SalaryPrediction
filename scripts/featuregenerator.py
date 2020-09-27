import pandas as pd


"""

FeatureGenerator class for Salary prediction case study

"""

class FeatureGenerator:
    
    def __init__(self, data, target_col):
        
        self.data = data
        self.cat_cols = data.cat_cols
        self.target_col = target_col
        self.groups = data.train_df.groupby(by=list(self.cat_cols))
        
    def add_group_stats(self):
        
        groups_stats_df = self._get_groupdata_stats()
        
        self.data.train_df = self._merge_new_cols(self.data.train_df, groups_stats_df, self.cat_cols, fillna=True)
        self.data.test_df = self._merge_new_cols(self.data.test_df, groups_stats_df, self.cat_cols, fillna=True)
        
        self._extend_feature_cols_list(self.data, cols=list(groups_stats_df.columns))
        
    def _get_groupdata_stats(self):
        
        data_stats = pd.DataFrame({'group_mean' : self.groups[self.target_col].mean()})
        data_stats['group_max'] = self.groups[self.target_col].max()
        data_stats['group_min'] = self.groups[self.target_col].min()
        data_stats['group_std'] = self.groups[self.target_col].std()
        data_stats['group_median'] = self.groups[self.target_col].median()
        
        return data_stats
        
        
    def _merge_new_cols(self, df1, df2, keys, fillna):
        
        df = df1.merge(df2, on=keys, how='left')
        if fillna:
            df.fillna(0, inplace=True)
        return df
    
    def _extend_feature_cols_list(self, data, cols):
        data.feature_cols.extend(cols)
        