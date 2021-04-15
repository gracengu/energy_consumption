import streamlit as st
import pandas as pd
import numpy as np
from numpy import log
import missingno as msno
import joblib
import glob
import os
import pmdarima as pm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse

from ec.config import Config

class Train(Config):

    data = {}

    def __init__(self):
        self.data["metrics_df"] = pd.DataFrame()

    def perform_adf_test(self, ts):
        result = adfuller(ts)
        print('ADF Statistics: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        if result[0] <result[4]["5%"]:
            print ("Reject Ho - Time Series is Stationary") # there is unit root
        else:
            print("Failed to Reject Ho - Time Series is Non-Stationary") # there is no unit root

    def split_train_validation(self, ts, perc):

        n = len(ts)
        train_df = ts[0:int(n*perc)]
        validation_df = ts[int(n*perc):]

        return train_df, validation_df

    def ts_histogram(self, ts):

        fig, ax = plt.subplots()
        ax.hist(ts)
        st.pyplot(fig)

    def normalization(self, ts):

        new_ts = log(ts.values)
        self.ts_histogram(new_ts)


    def temperature_forecast(self, mode, fgroup, id_name, prediction_df, y, fname):
    
        st.write("Currently forecasting for {}".format(id_name))

        fgroup.index = pd.to_datetime(fgroup.index)
        fgroup = fgroup.sort_index()
        # Preliminary checks
        # perform_adf_test(fgroup[y])
        # ts_histogram(fgroup[y])

        # Modelling
        if mode == 'hourly':
            fgroup = fgroup[y]
            ftrain, fval = self.split_train_validation(fgroup, 0.8)
            model = pm.auto_arima(ftrain,start_p=0,start_q=0,test='adf', 
                                m=12, d=0, max_order=4,
                                seasonal=True, start_P=0, D=1,maxiter=30,
                                trace=True, error_action='ignore', suppress_warnings=True,stepwise=True) 
        elif mode == 'daily':
            fgroup = fgroup[y].resample("D").mean()
            ftrain, fval = self.split_train_validation(fgroup, 0.6)
            model = pm.auto_arima(ftrain,start_p=0,start_q=0,test='adf', information_criterion='bic',
                            m=12, d=0, max_order=4,
                            seasonal=True, start_P=0, D=0,maxiter=50,
                            trace=True, error_action='ignore', suppress_warnings=True,stepwise=True) 
        elif mode == 'weekly':
            fgroup = fgroup[y].resample("W").mean()
            ftrain, fval = self.split_train_validation(fgroup, 0.6)
            model = pm.auto_arima(ftrain,start_p=0,start_q=0,test='adf', information_criterion='bic',
                            m=12, d=0, max_order=4,
                            seasonal=True, start_P=0, D=0,maxiter=50,
                            trace=True, error_action='ignore', suppress_warnings=True,stepwise=True) 
      
        # Save model
        model.fit(ftrain)
        if os.path.exists(fname):
            os.remove(fname)
        joblib.dump(model, fname)

        # Validation data
        forecast=model.predict(n_periods=len(fval), return_conf_int=True)
        forecast_df = pd.DataFrame(forecast[0],index = fval.index,columns=['Validation with Test'])
        st.pyplot(pd.concat([fgroup,forecast_df],axis=1).plot(figsize=(20,2)))
        error = rmse(fval.loc[fval.index.isin(forecast_df.index),], forecast_df)
        st.write(f'THE SARIMA(X) MODEL HAS RMSE OF {error}')
                                            
        # Forecast data
        forecast_df = prediction_df.loc[prediction_df["series_id"] == id_name,:].sort_index()
        if mode == 'hourly':

            forecast1=model.predict(n_periods=(len(forecast_df)+10), return_conf_int=True)
            forecast_range=pd.date_range(start=fgroup.index[-10], periods=(len(forecast_df)+10),freq='H')

        elif mode == 'daily':

            forecast1=model.predict(n_periods=(len(forecast_df)), return_conf_int=True)
            forecast_range=pd.date_range(start=forecast_df.index[0], periods=(len(forecast_df)),freq='D')

        elif mode == 'weekly':
        
            forecast1=model.predict(n_periods=(len(forecast_df)), return_conf_int=True)
            forecast_range=pd.date_range(start=fval.index[-1], periods=(len(forecast_df)),freq='W')
            
        forecast_df1 = pd.DataFrame(forecast1[0],index = forecast_range,columns=['Prediction'])
        st.pyplot(pd.concat([fgroup,forecast_df1],axis=1).plot(figsize=(20,2)))

        return forecast_df1

    def temperature_forecastload(self, mode, fgroup, id_name, prediction_df, y, fname):
    
        st.write("Currently loading model for {}".format(id_name))

        fgroup.index = pd.to_datetime(fgroup.index)
        fgroup = fgroup.sort_index()
        
        # Resample data
        if mode == 'hourly':
            fgroup = fgroup[y]
            
        elif mode == 'daily':
            fgroup = fgroup[y].resample("D").mean()
            
        elif mode == 'weekly':
            fgroup = fgroup[y].resample("W").mean()
            
        # Import model
        model = joblib.load(fname)
                                            
        # Forecast data
        forecast_df = prediction_df.loc[prediction_df["series_id"] == id_name,:].sort_index()
        if mode == 'hourly':

            forecast1=model.predict(n_periods=(len(forecast_df)+10), return_conf_int=True)
            forecast_range=pd.date_range(start=fgroup.index[-10], periods=(len(forecast_df)+10),freq='H')

        elif mode == 'daily':

            forecast1=model.predict(n_periods=(len(forecast_df)), return_conf_int=True)
            forecast_range=pd.date_range(start=fgroup.index[-1], periods=(len(forecast_df)),freq='D')

        elif mode == 'weekly':
        
            forecast1=model.predict(n_periods=(len(forecast_df)), return_conf_int=True)
            forecast_range=pd.date_range(start=fgroup.index[-1], periods=(len(forecast_df)),freq='W')
            
        forecast_df1 = pd.DataFrame(forecast1[0],index = forecast_range,columns=['Prediction'])
        st.pyplot(pd.concat([fgroup,forecast_df1],axis=1).plot(figsize=(20,2)))
        
        return forecast_df1

    

    