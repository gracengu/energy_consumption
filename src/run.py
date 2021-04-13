import streamlit as st
import numpy as np
import pandas as pd
import warnings
import time
import os
from glob import glob
import joblib

from ec.config import Config
from ec.analysis import Analysis
from ec.analysis.impute import Imputation
from ec.analysis.feature_engineering import FeatureEngineering
from ec.analysis.clustering import BuildingClustering

# Configurations
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.2f}".format
pd.options.display.max_rows = 200

ANALYSIS_DIRECTORY = "ec/data/analysis/"
TRAIN_DIRECTORY = "ec/data/train/"

IMPUTED_DATA = {}
for file_name in glob(os.path.join(ANALYSIS_DIRECTORY, 'imputed_data_*')):
    imputed_name = os.path.splitext(os.path.basename(file_name))[0]
    imputed_name = imputed_name.split("_")[3]
    print(imputed_name)
    IMPUTED_DATA[imputed_name] = pd.read_csv(file_name)

# Function 
@st.cache(allow_output_mutation=True)
def read_input():
    """
    Imports prediction, metadata, test and training data.

        Parameters:
        None

        Returns:
        Dictionary with prediction, metadata, test and training data as dictionary elements.
        Dataframe concatenate from prediction, training and test data. 
    """

    df_metadata_raw = pd.read_csv(Config.FILES["METADATA_DIR"])
    df_train_raw = pd.read_csv(Config.FILES["TRAIN_DIR"])
    df_test_raw = pd.read_csv(Config.FILES["TEST_DIR"])
    df_prediction_raw = pd.read_csv(Config.FILES["PREDICTION_DIR"])

    data_dict = dict({
        'metadata': df_metadata_raw, 
        'prediction':df_prediction_raw,
        'train': df_train_raw,
        'test': df_test_raw})

    feature_object = FeatureEngineering(data_dict)
    updated_dict = feature_object.add_features()

    final_df = pd.concat([updated_dict['train'], updated_dict['test'], updated_dict['prediction']], axis=0)
    final_df.to_csv(os.path.join(ANALYSIS_DIRECTORY, "merged_data.csv"))

    time.sleep(5)

    return updated_dict, final_df

# Segregate train, test and prediction
def data_segregation(data_dict, data):

    segregated_dict = {}

    for key in ['train', 'test', 'prediction']: 

        timestamp = data_dict.get(key).get('timestamp')
        buildings = data_dict.get(key).get('series_id')
        selected_data = data.loc[data["timestamp"].isin(timestamp) & data["series_id"].isin(buildings)]
        segregated_dict[key] = selected_data

    return segregated_dict

# Function calls
if __name__ == '__main__':

    st.title('''Building energy consumption - Analysis and Prediction''')

    data_dict, data = read_input()
    data_object = Analysis(data)
    data = data_object.datatype_conversion()
    # cleaned_dict = data_segregation(data_dict, data)
    cleaned_dict = data_dict
    train_test = pd.concat([cleaned_dict['train'], cleaned_dict['test']], axis=0)

    if data is not None:

        train_data = cleaned_dict['train']
        test_data = cleaned_dict['test']
        prediction_data = cleaned_dict['prediction']

        metadata_object = Analysis(data_dict['metadata']) 
        train_object = Analysis(train_data) 
        test_object = Analysis(test_data)
        prediction_object = Analysis(prediction_data)
        impute_object = Imputation(impute_method="interpolate", train=train_data, test=test_data)

        if st.sidebar.checkbox('Data Summary'):

            show = st.sidebar.radio("Select Data to View :",options=['Train Data','Test Data','Prediction Data'], index= 0)

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

        if st.sidebar.checkbox('Data Missingness'):

            show = st.sidebar.radio("Select Data to View :",options=['Train Data','Test Data','Prediction Data'], index= 0)
            if show == 'Train Data':
                train_object.missingness_stats()

                # for buildings, data in train_data.groupby('series_id'):
                #     train_object.missingno_matrix(data, 14, '6M')
                # train_object.heatmap_plot(train_object.cols_missing_pect(train_data, "series_id"), 'missing data pattern', rotate=None)

            elif show == 'Test Data':
                test_object.missingness_stats()

            elif show == 'Prediction Data':
                prediction_object.missingness_stats()

        if st.sidebar.checkbox('EDA: Metadata'):
            metadata_object.eda_metadata()

        if st.sidebar.checkbox('EDA: Temperature Zones'):
            train_object.eda_temperate()

        if st.sidebar.checkbox('Time Series Plots'):

            st.write("Time series plots for training data")
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
                            group_list = train_data.loc[train_data[var]==mode,:].groupby(["series_id"])
                            group_list = [(index, group) for index, group in group_list if len(group) > 0]
                            for name, group in group_list:
                                impute_object.time_series_plot(group, name, "consumption")
                                #TODO: To include options to view other variables

        if st.sidebar.checkbox('Clustering'):

            st.subheader("In this section, PCA and Clustering with be demonstrated.")
            df_options = {'Train Data': train_data,'Test Data': test_data}
            show = st.sidebar.radio("Select Data to Perform PCA and Clustering :", \
                options=list(df_options.keys()), index= 0)

            for type, data in df_options.items(): 
                if show == type: 
                    type = type.lower().split(" ")[0]
                    print(type)
                    cluster_object = BuildingClustering(data=data, train_data=train_data, df_type=type)

            if show == "Train Data":
                modes = ['Option to retrain', 'Analyse and Retrain', 'Retrain both PCA and Kmeans', \
                         'Retrain PCA only', 'Retrain Kmeans only', 'Generate Cluster only']
            else: 
                modes = ['Option to retrain', 'Generate Cluster only']

            show = st.sidebar.selectbox("Select your choice :",options=modes, index= 0)

            if show == 'Analyse and Retrain':
                pca_check_logic, pca_model_logic, kmeans_check_logic, kmeans_model_logic = (True,)*4
            elif show == "Retrain both PCA and Kmeans":
                pca_check_logic, pca_model_logic, kmeans_check_logic, kmeans_model_logic = (False, True,)*2
            elif show == "Retrain PCA only":
                pca_check_logic, pca_model_logic, kmeans_check_logic, kmeans_model_logic = False, True, False, False
            elif show == "Retrain Kmeans only":
                pca_check_logic, pca_model_logic, kmeans_check_logic, kmeans_model_logic = False, False, False, True
            elif show == "Generate Cluster only":
                pca_check_logic, pca_model_logic, kmeans_check_logic, kmeans_model_logic = (False,)*4
            elif show == "Option to retrain":
                st.subheader("Please choose whether to analyse PCA and/or Kmeans trend or proceed to retrain only.")

            if show is not 'Option to retrain':      
                cluster_object.main_clustering(pca_check=pca_check_logic, pca_model=pca_model_logic, \
                                            kmeans_check=kmeans_check_logic, kmeans_model=kmeans_model_logic)

        if st.sidebar.checkbox('Missing data imputation'):

            st.subheader("In this section, Missing Data Imputation with be demonstrated.")
            df_options = {'Train Data': train_data,'Test Data': test_data}
            df_options_show = st.sidebar.radio("Select Data to Perform Missing Data Imputation :", \
                                                options=list(df_options.keys()), index= 0)
            cluster_data = pd.read_csv(os.path.join(TRAIN_DIRECTORY, \
                            "clustering_data_{}.csv".format(df_options_show.lower().split(" ")[0])))

            modes = ['Option to impute', 'Impute and Analyse Plots', 'Analyse Plots Only']
            modes_show = st.sidebar.selectbox("Select your choice :",options=modes, index= 0)

            # impute by cluster
            if cluster_data is not None:
                if modes_show == 'Impute and Analyse Plots':

                    imputed_options = Config.IMPUTE_CONFIG["IMPUTATION_OPTIONS"]
                    method_show = st.sidebar.selectbox("Select imputed method:",options=imputed_options, index= 0)
                    default_features = Config.IMPUTE_CONFIG["DEFAULT_FEATURES_{}".format(method_show)] + ["temperature"]
                    features_list = st.sidebar.multiselect('Features selection:', options = list(cluster_data.columns), default = default_features)
                    button_impute = st.sidebar.button("SUBMIT")

                    if button_impute > 0:

                        # Imputation
                        group_segment = cluster_data.groupby(["Segment"])
                        group_list = [(index, group) for index, group in group_segment if len(group) > 0]
                        final_df = pd.DataFrame()
                        for name, group in group_list: 
                            st.subheader("Performing imputation for cluster {} ...".format(name))
                            imputed_df = impute_object.temperature_imputation(group, "temperature", method_show, features_list)
                            final_df = final_df.append(imputed_df, ignore_index=True)
                        final_df.to_csv(os.path.join(ANALYSIS_DIRECTORY, \
                            "imputed_data_{}_{}.csv".format(df_options_show.lower().split(" ")[0], \
                                                            method_show)))
                        st.subheader("Imputation completed.")

                        # Plot
                        st.subheader("Plotting...")
                        group_series = final_df.groupby(["series_id"])
                        group_list = [(index, group) for index, group in group_series if len(group) > 0]
                        for name, group in group_list:
                            impute_object.time_series_plot(group, name, "temperature", impute=True)

                elif modes_show == 'Analyse Plots Only':

                    imputed_options = Config.IMPUTE_CONFIG["IMPUTATION_OPTIONS"]
                    method_show = st.sidebar.selectbox("Select imputed method:",options=imputed_options, index= 0)
                    for imputed_method, imputed_df in IMPUTED_DATA.items():
                        if imputed_method == method_show:
                            # Plot
                            group_series = imputed_df.groupby(["series_id"])
                            group_list = [(index, group) for index, group in group_series if len(group) > 0]
                            for name, group in group_list:
                                impute_object.time_series_plot(group, name, "temperature", impute=True)

                elif modes_show == 'Option to impute':
                    st.write("Please select mode for imputation.")

            #TODO: To include options to choose variable to impute, \
            # now fixed at temperature since this is the only variable having missing data

            # group_list = train_data.groupby(["series_id"])
            # group_list = [(index, group) for index, group in group_list if len(group) > 0]
            # imputed_df = pd.DataFrame()
            # for name, group in group_list[:20]:

            #     vars = Config.FEATURE_DEFINITION["float_cols"]
            #     group_df = group.loc[:, group.columns.isin(vars)]
            #     subgroup_df = group.loc[:, ~group.columns.isin(vars)]
            #     interp_df = train_impute_object.interpolate(group_df, method="slinear", limit_direction=None)
            #     final_df = pd.concat([subgroup_df.reset_index(drop=True), interp_df.reset_index(drop=True)], axis=1)
            #     imputed_df = imputed_df.append(final_df, ignore_index=True)

            # st.dataframe(imputed_df)