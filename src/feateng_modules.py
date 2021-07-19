import os
import numpy as np
import pandas as pd

from ec.base import Logger
from ec.config import Config
from ec.analysis.feature_engineering import FeatureEngineering

# This script can be ran just by `python feateng_modules.py`

if __name__ == '__main__':

    _logger = Logger().logger

    df_metadata_raw = pd.read_csv(Config.FILES["METADATA_DIR"])
    df_train_raw = pd.read_csv(Config.FILES["TRAIN_DIR"])
    df_test_raw = pd.read_csv(Config.FILES["TEST_DIR"])
    df_prediction_raw = pd.read_csv(Config.FILES["PREDICTION_DIR"])

    prediction_points = ["Hourly", "Daily", "Weekly"]

    data_dict = dict({
        'metadata': df_metadata_raw, 
        'train': df_train_raw, # !: Train data should come before test and prediction for encoding purposes
        'test': df_test_raw,
        'prediction': df_prediction_raw}) 

    for points in prediction_points:

        _logger.info("Generating data for {}".format(points))

        complete_df = pd.DataFrame()

        if points == "Daily":
            resampling_method = "1D"
        elif points == "Weekly":
            resampling_method = "1W"
        else: 
            resampling_method = None

        for key, df in data_dict.items():

            _logger.info("Feature Engineering for {}".format(key))

            if key not in ["metadata", "prediction"]:

                df['timestamp']= pd.to_datetime(df['timestamp']) 
                if resampling_method is not None:
                    df = df.groupby('series_id').resample(resampling_method, on='timestamp').sum()
                    df = df.drop(columns=['series_id'])
                df.reset_index(inplace=True)

            if (points == "Hourly") or (points != "Hourly" and key != "metadata"):
                feature_object = FeatureEngineering(df, data_dict['metadata'])
                updated_df = feature_object.add_features(key)
            
            if key == "metadata" and points == "Hourly":
                data_dict['metadata'] = updated_df
            elif key != "metadata":
                if key == "prediction":
                    updated_df = updated_df.loc[updated_df["forecast"]==points, :]
                complete_df = pd.concat([complete_df, updated_df], axis=0, ignore_index=True)
                
        complete_df['forecast'] = complete_df['forecast'].fillna('Actual')
        complete_df.to_csv(Config.FILES["MERGED_{}_DIR".format(points.upper())], index=False)


