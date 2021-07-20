import streamlit as st
import numpy as np
import pandas as pd
import os, sys, inspect
from collections import OrderedDict

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings("ignore")

base_path, current_dir =  os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))

class Config:
    
    COLOR_PALETTE = {
    "Emerald green": "#00A19C",
    "Deep grey":  "#3D3935", 
    "Dark purple": "#432C5F",
    "Purple" :"#58478D",
    "Violet" :  "#615E9B",
    "Light blue": "#93BFEB",
    "Dark Green": "#395542",
    "Lime": "#CBD34C",
    "Gold": "#EFAA23",
    "Beige": "#CEB888",
    "Red": "#D81F2A",
    "Black": "#000000",
    "Night rider": "#333333",
    "Matterhorn": "#4d4d4d",
    "Dim grey":"#666666",
    "Grey":  "#777777",
    "Nobel very":  "#999999",
    "Light grey": "#CCCCCC"
    }

    PLOT_CONFIG = dict(
        sns_size            = (10,4),
        figure_style        = "whitegrid",
        plt_size            = 14,
        bxplt_rotation      = 45,
        cntplt_rotation     = 0,
        figure_labelsize    = 12
    )

    FILES = dict(
        METADATA_DIR                    = "./ec/data/analysis/Building Metadata.csv",
        TRAIN_DIR                       = "./ec/data/analysis/Training Data.csv",
        TEST_DIR                        = "./ec/data/analysis/Test Data.csv",
        PREDICTION_DIR                  = "./ec/data/analysis/Predictions.csv",
        MERGED_DIR                      = "./ec/data/analysis/merged_data_new.csv",
        MERGED_HOURLY_DIR               = "./ec/data/analysis/merged_data_Hourly.csv",
        MERGED_DAILY_DIR                = "./ec/data/analysis/merged_data_Daily.csv",
        MERGED_WEEKLY_DIR               = "./ec/data/analysis/merged_data_Weekly.csv",
        OUTPUT_DIR                      = "./ec/output",
        REGRESSION_MODELS_DIR           = "./ec/models",
        ANALYSIS_DIRECTORY              = "./ec/data/analysis/",
        TRAIN_DIRECTORY                 = "./ec/data/train/",
        TEMPFORECAST_DIRECTORY          = "./ec/data/forecast/temperature/",
        CONSUMPTIONFORECAST_DIRECTORY   = "./ec/data/forecast/consumption/",
        TEMPH_DIRECTORY                 = "./ec/models/time_series_temperature/hourly/",
        TEMPD_DIRECTORY                 = "./ec/models/time_series_temperature/daily/",
        TEMPW_DIRECTORY                 = "./ec/models/time_series_temperature/weekly/",
        ANSWERS_DIR                     = "./ec/data/analysis/Predictions_Answers.csv"

    )

    FEATURE_TYPES = dict(
        datetime_cols   = ['timestamp'],
        integer_cols    = ['hour', 'day', 'week', 'month', 'year', 'dayofweek', 'dayofyear'],
        float_cols      = ['consumption', 'temperature', 'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin',\
                           'dayofyear_cos', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos', \
                           'encoded_base_temperature', 'encoded_surface', 'encoded_session'],
        category_cols   = ['base_temperature', 'surface', 'day_off_count', 'session', 'day_name', \
                           'large', 'medium', 'small', 'x-large', 'x-small', 'xx-large', 'xx-small', 'high', 'low'],
        string_cols     = ['Set', 'series_id'],
        bool_cols       = ['is_night', 'is_day_off'],
    )

    FEATURE_DEFINITION = dict(
        Hourly = ['temperature', 'year', 'month',
                    'week', 'day', 'hour', 'dayofweek', 'dayofyear',
                    'is_day_off', 'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin',
                    'dayofyear_cos', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
                    'large', 'medium', 'small', 'x-large', 'x-small', 'xx-large',
                    'xx-small', 'high', 'low', 'last_week_consumption',
                    'last_day_consumption', 'last_hour_consumption'],
    
        Daily  = ['temperature', 'year', 'month',
                    'week', 'day', 'dayofweek', 'dayofyear',
                    'is_day_off', 'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin',
                    'dayofyear_cos', 'month_sin', 'month_cos',
                    'large', 'medium', 'small', 'x-large', 'x-small', 'xx-large',
                    'xx-small', 'high', 'low', 'last_week_consumption',
                    'last_day_consumption'],

        Weekly = ['temperature', 'year', 'month',
                    'week', 'month_sin', 'month_cos',
                    'large', 'medium', 'small', 'x-large', 'x-small', 'xx-large',
                    'xx-small', 'high', 'low', 'last_week_consumption',
                    'last_2week_consumption'],
    )

    CATEGORICAL_ORDERS = dict(
        Surface = ["xx-large", "x-large", "large", "medium", "small", "x-small", "xx-small"]
    )

    # Configurations for the training
    IMPUTE_CONFIG = dict(
        IMPUTATION_OPTIONS              = ['kNN', 'MICE'],
        DEFAULT_FEATURES_kNN            = ['encoded_hour', 'encoded_surface', 'encoded_base_temperature', \
                                            'encoded_is_day_off'],
        DEFAULT_FEATURES_MICE           = ['encoded_hour', 'encoded_surface', 'encoded_base_temperature', \
                                            'encoded_is_day_off', 'consumption'],
        KNN_NEIGHBOUR                   = 5,
        INTERPOLATE_DIRECTION           = "both",
        INTERPOLATE_ORDER               = 3,
        MICE_MAXITER                    = 100,
    )

    CLUSTER_CONFIG = dict(
        EXPLAINED_VARIANCETHRESHOLD     = 0.70,
        DEFAULT_PC                      = 3,
        MAX_CLUSTERS_ELBOW              = 20,
    )

    MODELLING_CONFIG = OrderedDict(

        VALIDATION = {
            'HOLDOUT_BUILDINGS': ["100807", "101079", "101132", "101208", "102214"]
        },

        REGRESSION_MODELS =  {
            'lr'    : LinearRegression(),
            'rf'    : RandomForestRegressor(),
            'lgbm'  : LGBMRegressor(),
        },

        PREDICTION = {
            'HOURLY_BUILDINGS': ["100191", "100489", "100540", "100548", "100566",
                "100567", "100626"], # "100021", "100046","100121",

            'DAILY_BUILDINGS': ["100017", "100036", "100137", "100481", "100719", "100755", "101003",
                "101010", "101357"], # "100948"

            'WEEKLY_BUILDINGS': ["100285", "100345", "100560", "100655", "100865", "101176", "101716",
                "101833", "102341"] # "100118"
        },

    )

    PREDICTION_CONFIG = dict(
        Hourly    = 25,
        Daily     = 8,
        Weekly    = 2,
    )


    




