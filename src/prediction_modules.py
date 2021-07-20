import os
import numpy as np
import pandas as pd

import random
from joblib import Parallel, delayed
import multiprocessing
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from ec.base import Logger
from ec.config import Config
from ec.analysis import Analysis
from ec.analysis.feature_engineering import FeatureEngineering
from ec.train.modelling_regression import Train

num_cores = multiprocessing.cpu_count()

if __name__ == '__main__':

    _logger = Logger().logger

    prediction_points = ['Hourly', 'Daily', 'Weekly']

    for points in prediction_points:

        # if points == "Hourly":

        #     df_path = Config.FILES["MERGED_{}_DIR".format(points.upper())]
        #     df = pd.read_csv(df_path)

        #     df = df.loc[df.Set.str.contains("^Test_2|^prediction"),:]
        #     df.loc[df.Set.str.contains("^Test"),"Set"] = "Test"

        #     df['timestamp']= pd.to_datetime(df['timestamp']) 
        #     selected_buildings = df[df.forecast==points].series_id.unique()
        #     df = df[df.series_id.isin(selected_buildings)]

        if points in ["Daily", "Weekly"]:

            _logger.info("Start {} prediction ...".format(points))

            # Data Preparation
            hourly_path = Config.FILES["MERGED_HOURLY_DIR"]
            hourly_df = pd.read_csv(hourly_path)
            data_object = Analysis(hourly_df)
            hourly_df = data_object.datatype_conversion()

            hourly_df = hourly_df.loc[hourly_df.Set.str.contains("^Test_2|^prediction"),:]
            hourly_df.loc[hourly_df.Set.str.contains("^Test"),"Set"] = "Test"

            # resampling to change temperature from resampling by sum to resampling by mean
            if points == "Daily":
                resampling_method = "1D"
            elif points == "Weekly":
                resampling_method = "1W"
            else: 
                resampling_method = None
            hourly_df['timestamp']= pd.to_datetime(hourly_df['timestamp']) 
            if resampling_method is not None:
                hourly_df = hourly_df.groupby('series_id').resample(resampling_method, on='timestamp').mean()
                hourly_df.reset_index(inplace=True)

            # prepare scoring data
            df_path = Config.FILES["MERGED_{}_DIR".format(points.upper())]
            df = pd.read_csv(df_path)
            data_object = Analysis(df)
            df = data_object.datatype_conversion()
            historical_df = df[df.forecast=='Actual'].copy()
            historical_df.drop(columns=['temperature'],inplace=True)
            historical_df = pd.merge(historical_df, hourly_df[['series_id','timestamp','temperature']],on=[\
                'series_id','timestamp'], how='left')
            score_df = pd.concat([historical_df, df[df.forecast==points]])
            prediction = df[df.forecast==points][['series_id','year','month','day']]
            prediction['drop_indicator'] = 1
            score_df = pd.merge(score_df, prediction, on=['series_id','year','month','day'],how='left')
            score_df = score_df[~((score_df.forecast=='Actual') & (score_df.drop_indicator==1))]

            # load model
            with open(os.path.join(Config.FILES["REGRESSION_MODELS_DIR"], '{}_rf.pkl'.format(points)), 'rb') as f:
                model = pickle.load(f)

            # generate lag features
            df_metadata_raw = pd.read_csv(Config.FILES["METADATA_DIR"])
            feat = FeatureEngineering(score_df, df_metadata_raw)
            df_lag = feat.lag_features(method=points)

            # prepare data for prediction
            recursive_df = score_df.copy()
            train_object = Train()
            recursive_df = train_object.split_train_test_point(recursive_df, method=points)
            recursive_df = recursive_df[recursive_df.series_id.isin(Config.MODELLING_CONFIG["PREDICTION"].get(\
                "{}_BUILDINGS".format(points.upper())))]
            
            # perform prediction
            recursive_val_lst = []
            features = Config.FEATURE_DEFINITION[points]
            for series_id in recursive_df.series_id.unique():
                testing = train_object.recursive_pred(recursive_df, series_id, model, features, points)
                recursive_val_lst.append(testing)

            # clean prediction data
            recursive_val_df = pd.concat(recursive_val_lst, axis=0)
            recursive_val_df.rename(columns={'consumption_actual'   : 'consumption',
                                            'consumption'           : 'consumption_pred'}, inplace=True)

            # cross check answers
            answers_df = pd.read_csv(Config.FILES["ANSWERS_DIR"])
            answers_df.rename(columns={'consumption':'consumption_answers'}, inplace=True)
            answers_df['timestamp'] = pd.to_datetime(answers_df['timestamp'])
            answers_df['series_id'] = answers_df['series_id'].astype('str')
            final_df = pd.merge(recursive_val_df, answers_df, on=['series_id','timestamp'], how='left')

            # get evaluation
            seriesid_list = []
            prediction_rmse = []
            prediction_mae = []

            for series_id in final_df.series_id.unique():

                subset_df = final_df[final_df.series_id == series_id].copy() 
                temp_rmse = train_object.rmse(subset_df[subset_df.forecast!='Actual'].consumption_answers.values, \
                    subset_df[subset_df.forecast!='Actual'].consumption_pred.values)
                temp_mae = train_object.mae(subset_df[subset_df.forecast!='Actual'].consumption_answers.values, \
                    subset_df[subset_df.forecast!='Actual'].consumption_pred.values)
                seriesid_list.append(series_id)
                prediction_rmse.append(temp_rmse)
                prediction_mae.append(temp_mae)


            metrics_summary_df = pd.DataFrame({'model'          : model.__class__.__name__,
                                              'forecast'        : points,
                                              'series_id'       : seriesid_list,
                                              'prediction_rmse' : prediction_rmse,
                                              'prediction_mae'  : prediction_mae,
            })

            _logger.info("[PredictionEvaluation] Prediction RMSE (Avg by series_id): {0:,.2f}".format( \
                metrics_summary_df.prediction_rmse.mean()))
            _logger.info("[PredictionEvaluation] Prediction MAE  (Avg by series_id): {0:,.2f}".format( \
                metrics_summary_df.prediction_mae.mean()))


            final_df.to_csv(os.path.join(Config.FILES["OUTPUT_DIR"], 'prediction_df_{}.csv'.format(points)), \
                index=False)
            metrics_summary_df.to_csv(os.path.join(Config.FILES["OUTPUT_DIR"], 'evaluation_df_{}.csv'.format(points)), \
                index=False)



