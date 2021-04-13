from src.config import Config
import pandas as pd 
import numpy as np
from src.analysis.feature_engineering import FeatureEngineering
from src.analysis.processing import DataProcessing
from src.analysis.eda import Analysis

DATA_DIR = 'C:/Projects/2021/02_building_energy_consumption_forecast/data'
metadata_filepath = DATA_DIR + "/Building Metadata.csv"
prediction_filepath = DATA_DIR + "/Predictions.csv"
test_filepath = DATA_DIR + "/Test Data.csv"
train_filepath = DATA_DIR + "/Training Data.csv"

df_metadata_raw = pd.read_csv(metadata_filepath)
df_prediction_raw = pd.read_csv(prediction_filepath)
df_test_raw = pd.read_csv(test_filepath)
df_train_raw = pd.read_csv(train_filepath)

data_dict = dict({
    'metadata': df_metadata_raw, 
    'prediction':df_prediction_raw,
    'train': df_train_raw,
    'test': df_test_raw})

feature_object = FeatureEngineering(data_dict)
updated_dict = feature_object.add_features()
final_df = pd.concat(updated_dict, axis=0)
data_object = DataProcessing(final_df)
data = data_object.datatype_conversion()


train_timestamp = list(data_dict['train'].get('timestamp'))
test_timestamp = list(data_dict['test'].get('timestamp'))
prediction_timestamp = list(data_dict['prediction'].get('timestamp'))

train_data = data.loc[data["timestamp"].isin(train_timestamp)]
test_data = data.loc[data["timestamp"].isin(test_timestamp)]
prediction_data = data.loc[data["timestamp"].isin(prediction_timestamp)]


data_object = Analysis(train_data) 
data_object.summary_statistics()

data_object = Analysis(test_data) 
data_object.summary_statistics()

data_object = Analysis(prediction_data) 
data_object.summary_statistics()


# keys = ['prediction', 'train', 'test']
# for key in keys: 
#     new = data_dict.get(key)
#     print(key)

# result_dict = [value for key, value in data_dict.items() if key not in 'metadata']
# # print(result_dict)
# print(type(result_dict))
