import os, yaml, logging
import itertools
import math
from math import sqrt
import fnmatch
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from collections import OrderedDict

import squarify
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

import statsmodels.api as sm
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM

from src.Config import Config


class Train(Config):
    data = {}

    def __init__(self, trend=["*"]):
        self.trend = trend
        self.data["metrics_df"] = pd.DataFrame()
    
    def evaluate_arima_model(self, X, arima_order):
        train_size = int(len(X) * 0.66)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        rmse = sqrt(mean_squared_error(test, predictions))
        return rmse
    
        
    def evaluate_models(self, dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        rmse = self.evaluate_arima_model(dataset, order)
                        if rmse < best_score:
                            best_score, best_cfg = rmse, order
    #                     print('ARIMA%s RMSE=%.3f' % (order,rmse))
                    except:
                        continue
        print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
        return best_cfg, best_score
        

    def sarima_grid_search(self, y,seasonal_period):
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
        mini = float('+inf')

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(y,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    if results.aic < mini:
                        mini = results.aic
                        param_mini = param
                        param_seasonal_mini = param_seasonal
    #                 print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                except:
                    continue
        print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))
        
        return param_mini, param_seasonal_mini, mini


    @staticmethod
    def root_mean_square_error(y_true, y_pred):
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return rmse


    def evaluate(self, actual, pred, model=None):
        r2score = r2_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        rmse = self.root_mean_square_error(actual, pred)
        
        metrics = dict(R2_Score=r2score,
                    MAE=mae,
                    RMSE=rmse)
        
        return metrics


    def split_data(self, df, ratio, forecasting=True):
        if forecasting == True:
            train_size = int(len(df) * ratio)
            self.data["train"], self.data["test"] = df.iloc[0: train_size], df.iloc[train_size: len(df)]
        else:
            X = df.iloc[:, 1:]
            Y = df.iloc[:, 0:1]
            self.data["X_train"], self.data["X_test"], self.data["y_train"], self.data["y_test"] = train_test_split(X, Y, train_size=0.7, random_state=42, shuffle=False)
        

    def trend_fit_metrics_score(self, df):
        model_2017_df = df.loc[df["date"]<="2017-10-01"]
        metrics_df = pd.DataFrame()

        for var in model_2017_df.columns.difference(["date", "year", "month"]):
            if fnmatch.fnmatch(var, "fit top *"):
                metrics = self.evaluate(model_2017_df.loc[model_2017_df["top 1"].notna()]["top 1"], 
                                model_2017_df.loc[model_2017_df[var].notna()][var])
                metrics.update({"trend": "top 1"})
            elif fnmatch.fnmatch(var, "fit2 top *"):
                metrics = self.evaluate(model_2017_df.loc[model_2017_df["top 2"].notna()]["top 2"], 
                                model_2017_df.loc[model_2017_df[var].notna()][var])
                metrics.update({"trend": "top 2"})
            elif fnmatch.fnmatch(var, "fit3 top *"):
                metrics = self.evaluate(model_2017_df.loc[model_2017_df["top 2"].notna()]["top 3"], 
                                model_2017_df.loc[model_2017_df[var].notna()][var])
                metrics.update({"trend": "top 3"})
            else:
                continue
            metrics.update({"fit": var})
            metrics_df = metrics_df.append([metrics])
            
        return metrics_df


    def barplot_metrics(self, data, metrics):
        fig, ax = plt.subplots(figsize=(15, 3))
        
        g = sns.barplot(y="trend", x=metrics, hue="fit", data=data, ax=ax, edgecolor="black", orient="h", palette="hls")
        ax.set(xlim=Config.METRICS_THRESHOLD_PLOT.get(metrics, None))
        ax.set_xlabel(metrics, fontsize=14, weight="bold")
        ax.set_ylabel("Trend", fontsize=14)
        
        g.legend(loc="center right", bbox_to_anchor=(1.15, 0.5), ncol=1)

        return fig


    def train_test_visualization(self, train, test, var, title=None):
        if ("date" in train.columns) or ("date" in test.columns):
            train = train.set_index("date")
            test = test.set_index("date")
        
        print("  Training dataset: {},   Testing dataset: {}".format(train.shape, test.shape))
        fig, ax = plt.subplots(figsize=(18,5))
        plt.plot(train[var], color='b')
        plt.plot(test[var], color='orange')
        plt.legend(["Train", "Test"])
        plt.title(title, fontsize=14, weight="bold")
        plt.xlabel("Date")
        plt.ylabel("Trend")
        
        return fig


    def check_stationarity(self, y, lags_plots=50, figsize=(22,8), title=None):
        y = pd.Series(y)
        fig = plt.figure()

        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((3, 3), (1, 0))
        ax3 = plt.subplot2grid((3, 3), (1, 1))
        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2)

        y.plot(ax=ax1, figsize=figsize, color="teal")
        ax1.set_title(title)
        plot_acf(y, lags=lags_plots, zero=False, ax=ax2, color="teal");
        plot_pacf(y, lags=lags_plots, zero=False, ax=ax3, method='ols', color="teal");
        sns.distplot(y, bins=int(sqrt(len(y))), ax=ax4, color='teal')
        ax4.set_title('Distribution')

        plt.tight_layout()
        
        print('Dickey-Fuller test results:')
        adfinput = adfuller(y)
        adftest = pd.Series(adfinput[0:4], index=['Statistical Test','P-Value','Used Lags','Observations Number'])
        adftest = round(adftest,4)
        
        for key, value in adfinput[4].items():
            adftest["Critical Values (%s)"%key] = value.round(4)
            
        print(adftest)

        return fig
        

    def generate_supervised(self, data, var):
        supervised_df = data.copy()
        
        for i in range(1,11):
            col = 'lag_' + str(i)
            supervised_df[col] = supervised_df[var].shift(i)
        
        supervised_df = supervised_df.dropna().reset_index(drop=True)
        
        return supervised_df
        
        
    def pred_line_plot(self, orig_df, orig_y, pred_df, pred_y, title=None):
        fig, ax = plt.subplots(figsize=(15,5))

        plt.plot(pd.to_datetime(orig_df.index), orig_y, data=orig_df, color='skyblue', label='observed')
        plt.plot(pd.to_datetime(pred_df.index), pred_y, data=pred_df, color='olive', label='predicted')
        ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(pred_df.index).min(), pd.to_datetime(pred_df.index).max(), alpha=.1, zorder=-1)
        
        plt.title("{} of {}".format(title, orig_y), fontsize=16, weight="bold")
        plt.ylabel("Trend", fontsize=14)
        plt.xlabel("Date", fontsize=14)
        plt.legend(fontsize="12")
        plt.show()
        
        return fig


    def run_model(self, df, var, model_name, forecasting=True):
        if forecasting == True:
            self.split_data(df=df, ratio=0.7, forecasting=True)
            self.data["train"] = self.data["train"][["date", var]].set_index("date")
            self.data["test"] = self.data["test"][["date", var]].set_index("date")
            
            if model_name == "ARIMA":
                p_values = self.MODELLING_CONFIG["P_RANGE"]
                d_values = self.MODELLING_CONFIG["D_RANGE"]
                q_values = self.MODELLING_CONFIG["Q_RANGE"]
                self.best_cfg, self.best_score = self.evaluate_models(self.data["train"].values, p_values, d_values, q_values)
                model = ARIMA(self.data["train"], order=(5,1,0))
                model_fit = model.fit(disp=0)
                pred_num = model_fit.predict(start=len(self.data["train"]), end=len(self.data["train"])+len(self.data["test"])-1)
                pred = pd.DataFrame(pred_num).rename(columns={0: var})
                pred = pred.set_index(pd.to_datetime(self.data["test"].index))
            elif model_name == "SARIMAX":
                self.param_mini, self.param_seasonal_mini, self.mini = self.sarima_grid_search(self.data["train"], 7)
                model = sm.tsa.statespace.SARIMAX(self.data["train"],
                                                order=self.param_mini,
                                                seasonal_order=self.param_seasonal_mini)
                model_fit = model.fit(disp=False)
                pref_fit = model_fit.get_prediction(start=len(self.data["train"]), end=len(self.data["train"])+len(self.data["test"])-1, dynamic=False)
                pred = pd.DataFrame(pref_fit.predicted_mean).rename(columns={0: var})
            elif model_name == "Exponential_Smooth":
                model = ExponentialSmoothing(self.data["train"], seasonal_periods=7, trend='add', seasonal='add')
                model_fit = model.fit(use_boxcox=False)
                pred_num = model_fit.predict(start=len(self.data["train"]), end=len(self.data["train"])+len(self.data["test"])-1)
                pred = pd.DataFrame(pred_num).rename(columns={0: var})
                pred = pred.set_index(pd.to_datetime(self.data["test"].index))
            else:
                print("Please only use models between: ARIMA, SARIMAX, Exponential_Smooth")
            
            metrics = self.evaluate(self.data["test"], pred, model=model_name)
        
        elif forecasting == False:
            self.data["supervised_df"] = df[["date", var]]
            self.data["supervised_df"] = self.generate_supervised(self.data["supervised_df"], var)
            self.data["supervised_df"].set_index("date", inplace=True)
            self.split_data(self.data["supervised_df"], ratio=0.7, forecasting=False)
            
            if model_name == "XGBoost":
                params_grid = {
                    'learning_rate': [0.05, 0.1, 0.3],
                    'min_child_weight': [1, 5, 10],
                    'max_depth': [2, 3, 5, 6],
                    'n_estimators': [20, 50, 200],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
                }
                n_folds = int(100 / (100*self.MODELLING_CONFIG["SPLIT_RATIO"])) + 1
                xgb = XGBRegressor()
                mod = GridSearchCV(estimator=xgb, param_grid=params_grid, scoring='r2', cv=n_folds, n_jobs=8)
                model_fit = mod.fit(self.data["X_train"], self.data["y_train"])
                print("    Best Params:", model_fit.best_params_)
                pred_fit = model_fit.predict(self.data["X_test"])
                pred = pd.DataFrame(pred_fit).rename(columns={0: var})
                pred.index = self.data["X_test"].index

            elif model_name == "LSTM":
                self.data["X_train"] = self.data["X_train"].values.reshape(self.data["X_train"].shape[0], 1, self.data["X_train"].shape[1])
                self.data["X_test"] = self.data["X_test"].values.reshape(self.data["X_test"].shape[0], 1, self.data["X_test"].shape[1])
            
                model = Sequential()
                model.add(LSTM(7, batch_input_shape=(1, self.data["X_train"].shape[1], self.data["X_train"].shape[2]), stateful=True))
                model.add(Dense(1))
                model.add(Dense(1))
                model.compile(loss="mean_squared_error", optimizer="adam")
                model.fit(self.data["X_train"], self.data["y_train"], epochs=200, batch_size=1, verbose=0, shuffle=False)
                model_fit = model.fit(self.data["X_train"], self.data["y_train"], epochs=200, batch_size=1, verbose=0, shuffle=False)
                pred = model.predict(self.data["X_test"], batch_size=1)
                pred = pd.DataFrame(pred).rename(columns={0: var})
                pred.index = self.data["y_test"].index
                
            metrics = self.evaluate(self.data["y_test"], pred, model=model_name)

        metrics["model"] = model_name
        metrics["trend"] = var
        self.data["metrics_df"] = self.data["metrics_df"].append(metrics, ignore_index=True)
        
        return model_fit, pred, metrics

    
    def plot_results(self, df):
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.lineplot("model", "RMSE", data=df, ax=ax, 
                    label='RMSE', color='mediumblue')
        sns.lineplot("model", "RMSE", data=df, ax=ax, 
                    label='MAE', color='Cyan')

        ax.set_title("Model Errors Comparison", fontsize=15, weight="bold")
        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Errors", fontsize=12)
        ax.grid(axis='y')
        
        return fig