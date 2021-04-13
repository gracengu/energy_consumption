import os
import random
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, LeaveOneGroupOut
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler

from ec.config import Config

ANALYSIS_DIRECTORY = "ec/data/analysis/"

class Imputation(Config):

    def __init__(self, train, test, impute_method=None):
        self.train          = train
        self.test           = test
        self.impute_method  = impute_method or self.IMPUTE_CONFIG["IMPUTE_METHOD"]
        
    def impute(self, data):
        """impute missing data from the selection of imputer methods"""
        imputer = {
            "interpolate": self.interpolate(data, method="linear", limit_direction="forward"),
            "knn": self.knn_impute(data, k=5, weights="uniform", metric="nan_euclidean"),
            "randomForest" : self.random_forest_impute(data, iteration=10, no_estimators=100),
            # "mice"
            # "ffill"
            # "clustering"
        }
        
        return imputer.get(self.impute_method, "Invalid impute method")


    def time_series_plot(self, group_df, building_id, y, impute=False):

        fig = plt.figure(figsize=(20,2))
        group_df['timestamp'] = pd.to_datetime(group_df['timestamp'])

        if impute: 

            original_data = self.train.loc[self.train["series_id"]== building_id,:]

            missing_dates = list(pd.to_datetime(original_data.loc[original_data[y].isnull(), "timestamp"]))

            imputed = group_df.sort_values(by = "timestamp")
            not_imputed = group_df.sort_values(by = "timestamp")

            imputed.loc[~imputed["timestamp"].isin(missing_dates), y] = np.nan
            not_imputed.loc[not_imputed["timestamp"].isin(missing_dates),y] = np.nan
            
            plt.plot(imputed["timestamp"], imputed[y], 'g')
            plt.plot(not_imputed["timestamp"], not_imputed[y], 'b')
            plt.title(str(building_id), fontsize=15)
            st.pyplot(fig) 

        else: 
            
            is_day_off = group_df.loc[:, group_df.columns.isin(["timestamp","is_day_off", \
                "session",y])].sort_values(by = "timestamp")
            is_not_day_off = group_df.loc[:, group_df.columns.isin(["timestamp","is_day_off",\
                "session",y])].sort_values(by = "timestamp")
            is_night = group_df.loc[:, group_df.columns.isin(["timestamp","is_day_off",\
                "session",y])].sort_values(by = "timestamp")

            is_day_off.loc[is_day_off['is_day_off'] == 0, y] = np.nan
            is_night.loc[~is_day_off['session'].isin(["Night", "Late Night"]),y] = np.nan

            plt.plot(is_not_day_off["timestamp"], is_not_day_off[y], 'g')
            plt.plot(is_day_off["timestamp"], is_day_off[y],  'r')
            plt.plot(is_night["timestamp"], is_night[y],  'b')
            plt.title(str(building_id), fontsize=15)
            st.pyplot(fig)

    def interpolate(self, df, method="slinear", limit_direction=None):
        
        if method == 'linear':
            df_new = df.interpolate(method=method, limit_direction=limit_direction)
        else:
            df_new = df.interpolate(method=method, order=self.IMPUTE_CONFIG["INTERPOLATE_ORDER"], \
                                limit_direction=self.IMPUTE_CONFIG["INTERPOLATE_DIRECTION"])

        return df_new

    def knn_impute(self, df, k=None, weights="uniform", metric="nan_euclidean"):
        
        if k == None:
            k = self.MODELLING_CONFIG["KNN_NEIGHBOUR"]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df.values)
        knn_imputer = KNNImputer(n_neighbors=k)
        scaled_data = knn_imputer.fit_transform(scaled_data)
        scaled_data = scaler.inverse_transform(scaled_data)

        st.write(scaled_data)

        return scaled_data


    def random_forest_impute(self, df, iteration=10, no_estimators=100):
        rf_imputer = MissForest(max_iter=iteration, n_estimators=no_estimators)
        impute_data = rf_imputer.fit_transform(df)
        impute_data = pd.DataFrame(impute_data, columns=impute_data.columns)

        return impute_data

    @st.cache(allow_output_mutation=True)
    def temperature_imputation(self, df, var, method, feature_list):

        # Linear Interpolation (Missing < 10%)
        grouped_linear = df.groupby(["series_id"]).filter(lambda x: x[var].isnull().sum()*100/len(x[var]) < 10 \
            and x[var].isnull().sum()*100/len(x[var]) > 0)
        grouped_linear[var] = grouped_linear[var].transform(\
            lambda group: group.interpolate(method='linear', axis=0).ffill().bfill())
        
        # kNN or MICE
        grouped_series = df.groupby(["series_id"]).filter(lambda x: x[var].isnull().sum()*100/len(x[var]) > 10)
        features_data = grouped_series.loc[:,grouped_series.columns.isin(feature_list)]

        if method == "kNN":
            imputer = KNNImputer(n_neighbors=Config.IMPUTE_CONFIG["KNN_NEIGHBOUR"], weights='uniform', \
                metric='nan_euclidean')
        if method == "MICE":
            imputer = IterativeImputer(max_iter=Config.IMPUTE_CONFIG["MICE_MAXITER"], random_state=126)

        imputer.fit(features_data)
        Xtrans = pd.DataFrame(imputer.transform(features_data))
        grouped_series[var] = Xtrans[0].tolist()

        # For complete data
        complete_data = df.groupby(["series_id"]).filter(lambda x: x[var].isnull().sum()*100/len(x[var]) == 0)
        
        # combined row
        if len(complete_data.columns) == len(grouped_linear.columns) == len(grouped_series.columns):
            imputed_df = pd.concat([complete_data.reset_index(drop=True), grouped_linear.reset_index(drop=True), \
                                    grouped_series.reset_index(drop=True)], axis = 0)

        return imputed_df



