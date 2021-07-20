import streamlit as st
import numpy as np
import pandas as pd
import warnings
import time
import os
from glob import glob
import joblib
import matplotlib.pyplot as plt
import pickle
import random
from PIL import Image

from ec.config import Config
from ec.analysis import Analysis
from ec.train.modelling import Train
from ec.analysis.impute import Imputation
from ec.analysis.feature_engineering import FeatureEngineering
from ec.analysis.clustering import BuildingClustering

# Configurations
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.2f}".format
pd.options.display.max_rows = 200


IMPUTED_DATA = {}
for file_name in glob(os.path.join(Config.FILES["ANALYSIS_DIRECTORY"], 'imputed_data_*')):
    imputed_name = os.path.splitext(os.path.basename(file_name))[0]
    imputed_name = imputed_name.split("_")[2] + "_" + imputed_name.split("_")[3]
    IMPUTED_DATA[imputed_name] = pd.read_csv(file_name)


def main(): 

    # Original Dataset
    metadata_df = pd.read_csv(Config.FILES["METADATA_DIR"])
    raw_train_data = pd.read_csv(Config.FILES["TRAIN_DIR"])
    raw_test_data = pd.read_csv(Config.FILES["TEST_DIR"])
    train_test = pd.concat([raw_train_data, raw_test_data], axis=0)
    raw_prediction_data = pd.read_csv(Config.FILES["PREDICTION_DIR"])

    # Raw Data Preparation
    feature_object = FeatureEngineering(metadata_df, metadata_df)
    metadata_df = feature_object.add_features('metadata')
    metadata_object = Analysis(metadata_df) 
    raw_train = Analysis(raw_train_data)
    raw_test = Analysis(raw_test_data)
    raw_prediction = Analysis(raw_prediction_data)

    # Clean Dataset with Feature Engineered
    hourly_df = pd.read_csv(Config.FILES["MERGED_HOURLY_DIR"])
    daily_df = pd.read_csv(Config.FILES["MERGED_DAILY_DIR"])
    weekly_df = pd.read_csv(Config.FILES["MERGED_WEEKLY_DIR"])
    train_data = hourly_df.loc[hourly_df.Set=="Train",:]
    test_data = hourly_df.loc[hourly_df.Set=="Test_1",:]
    prediction_data = hourly_df.loc[hourly_df.Set=="prediction",:]

    # Clean Data Preparation
    train_object = Analysis(train_data)
    test_object = Analysis(test_data)
    prediction_object = Analysis(prediction_data)

    impute_object = Imputation(impute_method="interpolate", train=train_data, test=test_data)
    model_object = Train()

    if st.sidebar.checkbox('Home'):

        # image = Image.open('https://www.pexels.com/photo/540977/download/?search_query=&tracking_id=n0ja7c6z2v')
        st.image('https://www.pexels.com/photo/540977/download/?search_query=&tracking_id=n0ja7c6z2v', \
            caption="High rise buildings in Chicago - For display purpose only")

        st.subheader('Project Background:')

        st.markdown('''

            Building energy consumption varies across a year, especially for countries in the temperate zones of the \
            southern and northern hemispheres. With varying consumption comes varying demand, thus it becomes \
            important for building managers to make better decisions to reasonably control all kinds of equipment.\
            A well-managed, energy efficient building offers opportunities to reduce costs and reduce greenhouse gas \
            emissions. However, as a result of randomness and noisy disturbance, it is not an easy task to realize \
            accurate prediction of the building energy consumption. 

        ''')

        st.subheader('Objective:')

        st.markdown('''
        
        The objective of this task is to forecast energy consumption based on temperature and other building \
        information.

        Three (3) time horizons for predictions are defined:
        - Forecasting the consumption for each hour in the next day (24 predictions).
        - Forecasting the consumption for each day in the coming week (7 predictions).
        - Forecasting the consumption for each week in the coming two weeks (2 predictions).

        ''')

        st.subheader('Tags:')

        st.markdown('''

        Data Wrangling, EDA, Feature Engineering, Regression, Time-series Forecasting

        ''')

        st.subheader('Data:')

        st.markdown('''

        Train:

        Historical energy consumption and temperature data for different buildings.
        - series_id - An ID number for the time series, matches across datasets.
        - timestamp - The time of the measurement
        - consumption - Consumption (watt-hours) since the last measurement
        - temperature - Outdoor temperature (Celsius) during measurement from nearby weather stations, some values missing

        Test:

        Similar format as Training Data.csv. The consumption is expected to be forecasted.

        Metadata:

        - series_id - An ID number for the time series, matches across datasets
        - surface - The surface area of the building (ordinal)
        - base_temperature - The base temperature that the inside of the building is set  (ordinal)
        - monday_is_day_off - Whether or not the building is operational this day
        - tuesday_is_day_off - Whether or not the building is operational this day
        - wednesday_is_day_off - Whether or not the building is operational this day
        - thursday_is_day_off - Whether or not the building is operational this day
        - friday_is_day_off - Whether or not the building is operational this day
        - saturday_is_day_off - Whether or not the building is operational this day
        - sunday_is_day_off - Whether or not the building is operational this day

        Prediction:

        A continuation of Test Data.csv.
        - series_id - An ID number for the time series, matches across datasets.
        - timestamp - The time of the measurement
        - consumption – The output comes here!
        - temperature - Outdoor temperature (Celsius) during measurement from nearby weather stations, some values missing

        
        ''')

    if st.sidebar.checkbox('Data Missingness'):

        show = st.sidebar.radio("Select Data to View :",options=['Train Data','Test Data','Prediction Data'],\
            index= 0, key='Missingness')
        if show == 'Train Data':
            raw_train.missingness_stats()

        elif show == 'Test Data':
            raw_test.missingness_stats()

        elif show == 'Prediction Data':
            raw_prediction.missingness_stats()

    if st.sidebar.checkbox('Data Summary'):

        show = st.sidebar.radio("Select Data to View :", options=['Train Data','Test Data','Prediction Data'], \
            index= 0, key='Summary')

        if show == 'Train Data':

            st.write("Number of records:", len(train_data), '(', round(len(train_data)*100/len(train_test),2),'%)')
            st.write("Number of unique buildings:", train_data['series_id'].nunique())
            train_object.summary_stats()

        elif show == 'Test Data':

            st.write("Number of records:", len(test_data) , '(', round(len(test_data)*100/len(train_test),2),' %)')
            st.write("Number of unique buildings:", test_data['series_id'].nunique())
            test_object.summary_stats()

        elif show == 'Prediction Data':

            st.write("Number of records:", len(prediction_data))
            st.write("Number of unique buildings:", prediction_data['series_id'].nunique())  
            prediction_object.summary_stats()

    if st.sidebar.checkbox('EDA: Metadata'):
        metadata_object.eda_metadata()

    if st.sidebar.checkbox('EDA: Temperature Zones'):
        train_object.eda_temperate()

    if st.sidebar.checkbox('Time Series Plots'):

        st.write("Time series plots for training data (10 random buildings):")
        vars = ['Choose variable to view', 'base_temperature', 'surface']
        show = st.sidebar.selectbox("Select grouping variable :",options=vars, index= 0)

        for var in vars: 

            if show == 'Choose variable to view':
                st.write("On the right hand panel, please select which variable to group data:")
                st.stop()

            elif show == var: 

                modes = ['Choose mode to view'] + train_data[var].unique().tolist() 
                show = st.sidebar.selectbox("Select mode to view :",options=modes, index= 0)
                for mode in modes: 

                    if show == 'Choose mode to view':
                        st.write("On the right hand panel, please select which mode to view time series plot:")
                        st.stop()

                    elif show == mode:
                        filtered_df = train_data.loc[train_data[var]==mode,:]
                        random.seed(60)
                        rand_series_id = random.sample(set(filtered_df.series_id.unique()), 10)
                        filtered_df = filtered_df[filtered_df.series_id.isin(rand_series_id)].copy()  
                        group_list = filtered_df.groupby(["series_id"])
                        group_list = [(index, group) for index, group in group_list if len(group) > 0]
                        for name, group in group_list:
                            impute_object.time_series_plot(group, name, "consumption", "train")
                            #TODO: To include options to view other variables

    if st.sidebar.checkbox('Missing data imputation'):

        st.subheader("In this section, Missing Data Imputation with be demonstrated.")
        df_options = {'Train Data': train_data,'Test Data': test_data}
        df_options_show = st.sidebar.radio("Select Data to Perform Missing Data Imputation :", \
                                            options=list(df_options.keys()), index= 0)
        cluster_data = pd.read_csv(os.path.join(Config.FILES["TRAIN_DIRECTORY"], \
                        "clustering_data_{}.csv".format(df_options_show.lower().split(" ")[0])))


        imputed_options = Config.IMPUTE_CONFIG["IMPUTATION_OPTIONS"]
        method_show = st.sidebar.selectbox("Select imputed method:",options=imputed_options, index= 0)
        method_show =  df_options_show.lower().split(" ")[0] + "_" + method_show
        for imputed_method, imputed_df in IMPUTED_DATA.items():
            
            if imputed_method == method_show:

                random.seed(60)
                rand_series_id = random.sample(set(imputed_df.series_id.unique()), 10)
                imputed_df = imputed_df[imputed_df.series_id.isin(rand_series_id)].copy()  
                group_series = imputed_df.groupby(["series_id"])
                group_list = [(index, group) for index, group in group_series if len(group) > 0]

                for name, group in group_list:
                    impute_object.time_series_plot(group, name, "temperature", type, impute=True)

    if st.sidebar.checkbox('Time Series Forecasting'):

        # training was pre-run using `python testing_modules.py`      

        train_object = Train()
        method = st.sidebar.radio("Select Test Period to View :",options=['Hourly','Daily','Weekly'], index= 0)

        st.markdown("**Evaluation results for all models**")
        with open(os.path.join(Config.FILES["OUTPUT_DIR"], 'overall_metrics_df_{}.csv'.format(method)), 'rb') as f:
            metrics_df = pd.read_csv(f)
        st.write(metrics_df)


        st.markdown("**Evaluation plots for baseline model**")
        with open(os.path.join(Config.FILES["OUTPUT_DIR"], 'baseline_df_{}.csv'.format(method)), 'rb') as f:
            baseline_df = pd.read_csv(f)
        train_object.plot_actual_pred(baseline_df, randomized=False)


        st.markdown("**Evaluation plots for train model**")
        with open(os.path.join(Config.FILES["OUTPUT_DIR"], 'model_output_{}.pkl'.format(method)), 'rb') as f:
            model_output = pickle.load(f)
        st.write("Linear Regression Model:")
        train_object.plot_actual_pred(model_output['lr'], num_plt=5)
        st.write("Random Forest Model (Selected):")
        train_object.plot_actual_pred(model_output['rf'], num_plt=5)
        st.write("Light Gradient Boosting Model:")
        train_object.plot_actual_pred(model_output['lgbm'], num_plt=5)


        st.markdown("**Evaluation plots (recursive) for selected model**")
        with open(os.path.join(Config.FILES["OUTPUT_DIR"], 'recursive_val_df_{}.csv'.format(method)), 'rb') as f:
            recursive_val_df = pd.read_csv(f)
        train_object.plot_actual_pred(recursive_val_df, randomized=False)

        
# Function calls
if __name__ == '__main__':

    st.title('''Building energy consumption - Analysis and Prediction''')
    st.markdown('''

    Please run the following lines to proceed: 
    - 'cd src'
    - 'python feateng_modules.py'
    - 'python training_modules.py'
    
    ''')
    main()



