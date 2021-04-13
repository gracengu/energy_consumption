import streamlit as st
import numpy as np
import pandas as pd
import os, sys, inspect

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
        METADATA_DIR   = "./ec/data/analysis/Building Metadata.csv",
        TRAIN_DIR      = "./ec/data/analysis/Training Data.csv",
        TEST_DIR       = "./ec/data/analysis/Test Data.csv",
        PREDICTION_DIR = "./ec/data/analysis/Predictions.csv",
    )

    FEATURE_DEFINITION = dict(
        datetime_cols   = ['timestamp', 'hour', 'day', 'weekday', 'week', 'month', 'year'],
        integer_cols    = ['series_id'],
        float_cols      = ['temperature', 'encoded_is_day_off', 'encoded_day_off_count', 'encoded_base_temperature', 
                           'encoded_surface', 'encoded_hour', 'encoded_month', 'encoded_session', 'encoded_is_night'],
        category_cols   = ['base_temperature', 'surface', 'is_day_off', 'day_off_count', 'session', 'is_night']
    )

    CATEGORICAL_ORDERS = dict(
        Surface = ["xx-large", "x-large", "large", "medium", "small", "x-small", "xx-small"]
    )

    # Configurations for the training
    IMPUTE_CONFIG = dict(
        IMPUTATION_OPTIONS              = ['kNN', 'MICE'],
        DEFAULT_FEATURES_kNN            = ['encoded_hour', 'encoded_surface', 'encoded_base_temperature', 'encoded_is_day_off'],
        DEFAULT_FEATURES_MICE           = ['encoded_hour', 'encoded_surface', 'encoded_base_temperature', 'encoded_is_day_off', 'consumption'],
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



    




