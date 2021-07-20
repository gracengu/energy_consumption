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

# This script can be ran just by `python training_modules.py`

if __name__ == '__main__':

    _logger = Logger().logger

    prediction_points = ['Hourly', 'Daily', 'Weekly']

    for points in prediction_points:

        _logger.info("Start {} data preparation ...".format(points))

        df_path = Config.FILES["MERGED_{}_DIR".format(points.upper())]
        main_df = pd.read_csv(df_path, index_col=False)
        main_df = main_df.loc[:, ~main_df.columns.str.contains('^Unnamed')]
        data_object = Analysis(main_df)
        main_df = data_object.datatype_conversion()

        # get training data only 
        df = main_df.loc[main_df.Set.str.contains("^Train|^Test_1"),:]
        df.loc[df.Set.str.contains("^Test"),"Set"] = "Test"
        df.pop("forecast")

        _logger.info("Start {} training ...".format(points))

        train_object = Train()
        metrics_df, pred_df = train_object.baseline_model(df, method=points)
        pred_df = pred_df[pred_df.series_id.isin(Config.MODELLING_CONFIG["VALIDATION"]\
            .get('HOLDOUT_BUILDINGS'))]
        pred_df.to_csv(os.path.join(Config.FILES["OUTPUT_DIR"], 'baseline_df_{}.csv'.format(points)), index=False)
        # train_object.plot_actual_pred(pred_df, num_plt=10)

        # create lag features
        df_metadata_raw = pd.read_csv(Config.FILES["METADATA_DIR"])
        feat = FeatureEngineering(df, df_metadata_raw)
        df_lag = feat.lag_features(method=points)

        # model training
        df_lag = df_lag.dropna()
        train_df = df_lag[df_lag.Set=='Train']
        features = Config.FEATURE_DEFINITION[points]
        models = Config.MODELLING_CONFIG["REGRESSION_MODELS"]
        model_output, overall_metrics_detail_df, overall_metrics_df = train_object.model_training(df_lag, train_df, \
            features, models, points)
        
        # save output
        final_df = pd.concat([metrics_df, overall_metrics_df], axis=0)
        # metrics_df.append(overall_metrics_df)
        final_df.to_csv(os.path.join(Config.FILES["OUTPUT_DIR"], 'overall_metrics_df_{}.csv'.format(points)),index=False)
        overall_metrics_detail_df.to_csv(os.path.join(Config.FILES["OUTPUT_DIR"], 'overall_metrics_detail_df_{}'.format(points)), \
            index=False)
        with open(os.path.join(Config.FILES["OUTPUT_DIR"], 'model_output_{}.pkl'.format(points)), 'wb') as f:
            pickle.dump(model_output, f)
            
        # random.seed(60)
        # train_object.plot_actual_pred(model_output['lr'], num_plt=10)
        
        # random.seed(60)
        # train_object.plot_actual_pred(model_output['rf'], num_plt=10)
        
        # random.seed(60)
        # train_object.plot_actual_pred(model_output['lgbm'], num_plt=10)
        
        # evaluate for recursive prediction
        with open(os.path.join(Config.FILES["REGRESSION_MODELS_DIR"], '{}_rf.pkl'.format(points)), 'rb') as f:
            model = pickle.load(f)
        
        # assign point of prediction to 'Test' in Set
        recursive_df = df.copy()
        recursive_df = recursive_df[recursive_df.series_id.isin(Config.MODELLING_CONFIG["VALIDATION"]\
            .get('HOLDOUT_BUILDINGS'))]
        recursive_df = train_object.split_train_test_point(recursive_df, method=points)

        # recursive_val_lst = Parallel(n_jobs=num_cores)(delayed(train_object.recursive_pred)(recursive_df, series_id, \
        #     model, features, points) for series_id in recursive_df.series_id.unique())

        recursive_val_lst = []
        for series_id in recursive_df.series_id.unique():
            testing = train_object.recursive_pred(recursive_df, series_id, model, features, points)
            recursive_val_lst.append(testing)

        recursive_val_df = pd.concat(recursive_val_lst, axis=0)
        recursive_val_df.rename(columns={'consumption_actual':'consumption',
                                        'consumption':'consumption_pred'}, inplace=True)
        recursive_val_df.to_csv(os.path.join(Config.FILES["OUTPUT_DIR"], 'recursive_val_df_{}.csv'.format(points)), \
            index=False)

        # random.seed(60)
        # train_object.plot_actual_pred(recursive_val_df, num_plt=10)


    