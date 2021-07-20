import os
import random
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pickle
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from ec.base import Logger
from ec.config import Config
from ec.analysis.feature_engineering import FeatureEngineering

num_cores = multiprocessing.cpu_count()


class Train(Config):

    def __init__(self):
        self._logger = Logger().logger
        self.series_id = list()
        self.train_rmse = list()
        self.train_mae = list()
        self.test_rmse = list()
        self.test_mae = list()
        self.baseline_metrics = pd.DataFrame()
        self.train_metrics = pd.DataFrame()

    def rmse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    def mape(self, y_true, y_pred): 
        return mean_absolute_percentage_error(y_true, y_pred) * 100

    def mae(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def metrics_per_id (self, df_with_prediction, pred_col = 'consumption_pred'):

        """Generates accuracy metrics for each series id.

        Returns:
            dataframe: dataframe consisting of series id, both train and test rmse and mae
        """        
        
        for series_id in df_with_prediction.series_id.unique():
        
            subset_df = df_with_prediction[df_with_prediction.series_id == series_id].copy() 
            subset_df = subset_df.dropna(subset = [pred_col]) 

            try: 
                # train 
                train_rmse = self.rmse(subset_df[subset_df.Set=='Train'].consumption.values,
                                subset_df[subset_df.Set=='Train'][pred_col].values)
                train_mae = self.mae(subset_df[subset_df.Set=='Train'].consumption.values,
                                subset_df[subset_df.Set=='Train'][pred_col].values)

                # test
                test_rmse = self.rmse(subset_df[subset_df.Set=='Test'].consumption.values,
                                subset_df[subset_df.Set=='Test'][pred_col].values)
                test_mae = self.mae(subset_df[subset_df.Set=='Test'].consumption.values,
                                subset_df[subset_df.Set=='Test'][pred_col].values)

                self.series_id.append(series_id)
                self.train_rmse.append(train_rmse)
                self.train_mae.append(train_mae)
                self.test_rmse.append(test_rmse)
                self.test_mae.append(test_mae)
                
            except ValueError: 
                print("Building:{} doesn't have consumption value in either train or test dataset".format(series_id))
                
                
        metrics_df = pd.DataFrame({'series_id'      : self.series_id,
                                    'train_rmse'    : self.train_rmse,
                                    'train_mae'     : self.train_mae,
                                    'test_rmse'     : self.test_rmse,
                                    'test_mae'      : self.test_mae})
        return metrics_df

    def baseline_model (self, original_df, method):

        """Generates baseline model for blind comparison (before answer is given).

        Hourly model: Take consumption from last week at same hour as prediction
        Daily model: Take consumption from last week at same day as prediction
        Weekly model: Take consumption from last week as prediction

        Returns:
            dataframe: evaluation metrics dataframe and the prediction dataframe. 
        """        
        
        if method == 'Hourly':
            pred_point = 24*7 # last week same hour
        elif method == 'Daily':
            pred_point = 7 # last week same day
        elif method == 'Weekly':
            pred_point = 1 # last week


        df = original_df.copy()
        df['consumption_pred'] = df['consumption']
        df['consumption_pred'] = df.groupby(['series_id'])['consumption_pred'].transform(lambda x:x.shift(pred_point))
        
        self.baseline_metrics = self.metrics_per_id(df)
        metrics_summary_df = pd.DataFrame({'model':['baseline'],
                                        'train_rmse' : [self.baseline_metrics.train_rmse.mean()],
                                        'train_mae'  : [self.baseline_metrics.train_mae.mean()],
                                        'test_rmse'  : [self.baseline_metrics.test_rmse.mean()],
                                        'test_mae'   : [self.baseline_metrics.test_mae.mean()]})

        self._logger.info("[BaseModelEvaluation] Train RMSE (Avg by series_id): {0:,.2f}".format( \
            self.baseline_metrics.train_rmse.mean()))
        self._logger.info("[BaseModelEvaluation] Train MAE  (Avg by series_id): {0:,.2f}".format( \
            self.baseline_metrics.train_mae.mean()))
        self._logger.info("[BaseModelEvaluation] Test RMSE  (Avg by series_id): {0:,.2f}".format(  
            self.baseline_metrics.test_rmse.mean()))
        self._logger.info("[BaseModelEvaluation] Test RMSE  (Avg by series_id): {0:,.2f}".format( \
            self.baseline_metrics.test_mae.mean()))

        return metrics_summary_df, df

        
    def plot_actual_pred(self, df_with_prediction, pred_col = 'consumption_pred', num_plt = 5, randomized=True, app=True):

        """plot random 5 series_id for visualization
        """  
        
        if randomized:
            rand_series_id = random.sample(set(df_with_prediction.series_id.unique()), num_plt)
            df_plot = df_with_prediction[df_with_prediction.series_id.isin(rand_series_id)][['timestamp','series_id',\
                    'Set','consumption']+[pred_col]].copy()           
        else: 
            df_plot = df_with_prediction[['timestamp','series_id','Set','consumption']+[pred_col]].copy()
            df_plot = df_plot.groupby(["series_id", "Set"]).apply(lambda x: x.sort_values(by="timestamp",ascending=True))
            df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
            
        df_plot = pd.melt(df_plot,['timestamp','series_id','Set'])
        df_plot['hue_col'] = df_plot['variable']+'_'+ df_plot['Set']

        fig = sns.FacetGrid(df_plot, col="series_id", hue='hue_col', height=2, aspect=6, col_wrap=1, sharey = False, \
            sharex = False, legend_out=False)
        fig.map(sns.lineplot, "timestamp", "value")
        fig.tight_layout()
        fig.add_legend(loc='upper left')

        if app:
            st.pyplot(fig)
        else:
            plt.show()

        return None


    def split_train_test_point(self, df, method):
        """split train test based on prediction points.

        Args:
            df (dataframe): dataframe to be splitted into train and test
            method (string): forecast type i.e. hourly/daily/weekly

        Returns:
            dataframe: data labelled as train or test. 
        """        

        out_df = pd.DataFrame()
        num_row = self.PREDICTION_CONFIG[method]

        for series_id in df.series_id.unique():
            subset_df = df[df.series_id == series_id]
            subset_df = subset_df.reset_index(drop=True)
            index_test = subset_df.shape[0] - num_row
            subset_df.loc[:index_test,'Set'] = 'Train'
            subset_df.loc[index_test:,'Set'] = 'Test'
            out_df = out_df.append(subset_df,ignore_index=True)

        return out_df

    def model_training (self, df, train_df, features, models, method):

        """trains model

        Args:
            df (dataframe): 
            train_df (dataframe): 
            features (list):
            models (list):
            method (string): forecast type i.e. hourly/daily/weekly

        Returns:
            model: trained model output
            dataframe: detail evaluation metrics 
        """        

        overall_metrics_detail_df = pd.DataFrame()
        overall_metrics_df = pd.DataFrame()
        model_output = dict()
        pred_df = df.copy()

        for name, model in models.items():

            # train model
            self._logger.debug("[{}] Fitting Model ...".format(model.__class__.__name__+"Model"))
            reg = model.fit(train_df[features].values, train_df['consumption'].values.reshape(-1,1))

            # get prediction
            self._logger.debug("[{}] Performing Prediction ...".format(model.__class__.__name__+"Model"))
            pred = reg.predict(pred_df[features])
            pred_df = pred_df.assign(consumption_pred = pred)

            # get evaluation
            self._logger.debug("[{}] Evaluating Accuracy ...".format(model.__class__.__name__+"Model"))
            self.train_metrics = self.metrics_per_id(pred_df)
            self.train_metrics['model'] = name
            overall_metrics_detail_df = overall_metrics_detail_df.append(self.train_metrics)
            metrics_summary_df = pd.DataFrame({'model':[name],
                                            'train_rmse' : [self.train_metrics.train_rmse.mean()],
                                            'train_mae'  : [self.train_metrics.train_mae.mean()],
                                            'test_rmse'  : [self.train_metrics.test_rmse.mean()],
                                            'test_mae'   : [self.train_metrics.test_mae.mean()]})
 
            overall_metrics_df = overall_metrics_df.append(metrics_summary_df)
            self._logger.info("[TrainModelEvaluation] Train RMSE (Avg by series_id): {0:,.2f}".format(\
                self.train_metrics.train_rmse.mean()))
            self._logger.info("[TrainModelEvaluation] Train MAE  (Avg by series_id): {0:,.2f}".format(\
                self.train_metrics.train_mae.mean()))
            self._logger.info("[TrainModelEvaluation] Test RMSE  (Avg by series_id): {0:,.2f}".format(\
                self.train_metrics.test_rmse.mean()))
            self._logger.info("[TrainModelEvaluation] Test RMSE  (Avg by series_id): {0:,.2f}".format(\
                self.train_metrics.test_mae.mean()))
            model_output[name] = pred_df

            with open('./ec/models/mp/'+ method +'_'+ name +'.pkl', 'wb') as f:
                pickle.dump(reg, f)

            self._logger.debug("[{}] Model Saved ...".format(model.__class__.__name__+"Model"))
                
        return model_output, overall_metrics_detail_df, overall_metrics_df
            

    def recursive_pred(self, df, series_id, model, features, method):

        """recursively adding prediction as part of validation data

        1st iter: f(actual1,actual2,actual3) = pred1
        2nd iter: f(actual2,actual3,pred1) = pred2

        Returns:
            dataframe: recursed dataframe
        """        

        freq_df = df[df.series_id==series_id].copy()
        freq_df['consumption_actual'] = freq_df['consumption']
        num_row = self.PREDICTION_CONFIG[method]
        freq_df = freq_df.sort_values('timestamp')
        freq_df = freq_df.reset_index(drop=True)
        
        # subset df to have historical data for lag_features
        for i in range(1, num_row+1, 1)[::-1]:
                        
            end_index = freq_df.index[-i]
                        
            if method == 'Hourly':
                feature_lag_max = 24*7 
            elif method == 'Daily':
                feature_lag_max = 7
            elif method == 'Weekly':
                feature_lag_max = 2
            
            start_index = end_index - feature_lag_max
            pred_df = freq_df.iloc[start_index:end_index+1]
            
            # build lag feature
            df_metadata_raw = pd.read_csv(Config.FILES["METADATA_DIR"])
            feat = FeatureEngineering(pred_df, df_metadata_raw)
            pred_df = feat.lag_features(method)
            
            # predict
            try:
                pred = model.predict(pred_df[features].iloc[-1:])
            except ValueError:
                print(series_id)
            
            # make prediction the actual consumption
            freq_df["consumption"].iloc[-i,] = float(pred)   
            
        return freq_df