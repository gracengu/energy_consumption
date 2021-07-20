import os
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelBinarizer
import pickle

from ec.base import Logger
from ec.config import Config

class FeatureEngineering(Config):

    """A class that generates the feature engineered for the input dataset. 

    Attributes
    ----------
    data_dict : dict
        a dictionary of all data where new features are to be added.
    metadata : dataframe
        the metadata of buildings
    keys : list
        a list of all dictionary keys in data_dict

    Methods
    -------
    add_is_day_off
        Flags the observation as is day off (1) or not day off (0). 

    part_of_day
        Bins the hours of the day to different parts of the day. 

    add_features
        Add features to the training dataframe. 

    """    

    st = __import__('streamlit')

    def __init__(self, data, metadata):

        """
        Parameters
        ----------
        data_dict : dict
            a dictionary consisting of all input data namely train, test, prediction and metadata
        
        """
        self._logger    = Logger().logger
        self.data       = data
        self.metadata   = metadata


    @staticmethod
    def part_of_day(x):

        """Bins the hours of the day to different parts of the day. 

        Returns:
            str: The part of day i.e. early morning, morning, noon, eve, night or late night. 
        """        
        if (x > 4) and (x <= 8):
            return 'Early Morning'
        elif (x > 8) and (x <= 12 ):
            return 'Morning'
        elif (x > 12) and (x <= 16):
            return'Noon'
        elif (x > 16) and (x <= 20) :
            return 'Eve'
        elif (x > 20) and (x <= 24):
            return'Night'
        elif (x <= 4):
            return'Late Night'


    def day_off_flag(self, row, cols, iter):

        return  np.where((row[cols] == True) & (row['dayofweek'] == iter), True, row['is_day_off'])


    def is_day_off(self):

        temp = list(self.data.filter(regex = "is_day_off").columns)
        self.data['is_day_off'] = False
        for cols in temp: 
            self.data['is_day_off'] = self.data.apply(self.day_off_flag, axis = 1, args = [cols, temp.index(cols)])

        self._logger.info("Complete adding is day off feature.")


    def split_train_test(self, train_proportion=0.8):
        
        """split train test for the training dataset

        Returns: 
            dataframe: dataset with added new column 'Set' indicating train or test. 
        
        """   

        self.data.sort_values(['series_id', 'timestamp'], inplace=True)
        out_df = pd.DataFrame()
        for series_id in self.data.series_id.unique():
            subset_df = self.data[self.data.series_id==series_id]
            subset_df = subset_df.reset_index(drop=True)
            index_test = np.ceil(subset_df.shape[0]*train_proportion)
            subset_df[["Set"]] = None
            subset_df.loc[:index_test, "Set"] = "Train"
            subset_df.loc[index_test:, "Set"] = "Test_1"
            out_df = out_df.append(subset_df, ignore_index=True)
        self.data = out_df

        self._logger.info("Complete splitting train data to train and test dataset.")


    def datetime_feature(self):

        """add datetime features

        Returns:
            dataframe: dataframe with added datetime features
        """        
        self.data['day_name'] = self.data.timestamp.dt.day_name()
        self.data['hour'] = self.data.timestamp.dt.hour
        self.data['day'] = self.data.timestamp.dt.day
        self.data['week'] = self.data.timestamp.dt.isocalendar().week
        self.data['month'] = self.data.timestamp.dt.month
        self.data['year'] = self.data.timestamp.dt.year
        self.data['dayofweek'] = self.data.timestamp.dt.dayofweek
        self.data['dayofyear'] = self.data.timestamp.dt.dayofyear
        self.data['session'] = self.data.hour.apply(self.part_of_day)
        self.data['is_night'] = self.data.apply(lambda x: x.session in ['Night', 'Late Night'], axis=1)

        self._logger.info("Complete adding datetime features.")


    def manual_temp_imputation(self):

        """perform manual imputation for temperature 

        Returns:
            dataframe: dataframe with imputed temperature column
        """         

        self.data = self.data.sort_values(['series_id', 'timestamp'])

        print(self.data.loc[self.data.series_id == 100948,"temperature"])

        # one timestamp before
        self.data['temperature'] = self.data['temperature'].fillna(self.data.groupby(['series_id'])['temperature'].\
            ffill())
        # one timestamp after
        self.data['temperature'] = self.data['temperature'].fillna(self.data.groupby(['series_id'])['temperature'].\
            bfill())
        # same time one day ago
        self.data['temperature'] = self.data['temperature'].fillna(self.data.groupby(['series_id', 'hour'])\
            ['temperature'].ffill())
        # same time one year ago
        self.data['temperature'] = self.data['temperature'].fillna(self.data.groupby(['series_id', 'dayofyear'])\
            ['temperature'].ffill())
        # same week (mean)
        self.data['temperature'] = self.data.groupby(['series_id', 'week']).temperature.\
            transform(lambda x:x.fillna(x.mean()))
        # same month (mean)
        self.data['temperature'] = self.data.groupby(['series_id', 'month']).temperature.\
            transform(lambda x:x.fillna(x.mean()))
        # same timestamp, surface, base_temperature (mean)
        self.data['temperature'] = self.data.groupby(['timestamp', 'surface', 'base_temperature']).temperature. \
            transform(lambda x:x.fillna(x.mean()))
        # one timestamp before
        self.data['temperature'] = self.data['temperature'].fillna(self.data.groupby(['series_id'])['temperature']. \
            ffill())
        # same timestamp mean
        self.data['temperature'] = self.data.groupby(['timestamp']).temperature.transform(lambda x:x.fillna(x.mean()))

        print(self.data.loc[self.data.series_id == 100948,"temperature"])

        self._logger.info("Complete imputation for temperature.")


    def cyclical_trans(self):

        """transform features into cyclical features

        Returns: 
            dataframe: dataframe with added cyclical features
        """        

        self.data['dayofweek_sin'] = np.sin(2*np.pi*self.data['dayofweek'] /7)
        self.data['dayofweek_cos'] = np.cos(2*np.pi*self.data['dayofweek'] /7)

        self.data['dayofyear_sin'] = np.sin(2*np.pi*self.data['dayofyear']-1 /365)
        self.data['dayofyear_cos'] = np.cos(2*np.pi*self.data['dayofyear']-1 /365)

        self.data['month_sin'] = np.sin(2*np.pi*self.data['month']-1 /12)
        self.data['month_cos'] = np.cos(2*np.pi*self.data['month']-1 /12)

        self.data['hour_sin'] = np.sin(2*np.pi*self.data['hour']-1 /24)
        self.data['hour_cos'] = np.cos(2*np.pi*self.data['hour']-1 /24)

        self._logger.info("Complete cyclical transformation for time features.")

    def onehot_categorical(self, phase):

        """one-hot-encoding for categorical features

        Returns: 
            dataframe: dataframe with one-hot-encoded features
        """        

        if phase == 'train':
            surface_enc = LabelBinarizer()
            surface_enc.fit(self.data['surface'])
            transformed = surface_enc.transform(self.data['surface'])
            ohe_df = pd.DataFrame(transformed, columns=surface_enc.classes_)
            self.data = pd.concat([self.data.reset_index(drop=True), ohe_df], axis=1)
            self._save_pkl(surface_enc, "surface_enc.pkl")

            bt_enc = LabelBinarizer()
            bt_enc.fit(self.data['base_temperature'])
            transformed = bt_enc.transform(self.data['base_temperature'])
            transformed = np.hstack((1-transformed, transformed))
            ohe_df = pd.DataFrame(transformed, columns=bt_enc.classes_)
            self.data = pd.concat([self.data.reset_index(drop=True), ohe_df], axis=1)
            self._save_pkl(bt_enc, "base_temp_enc.pkl")

        else: 
            file_name = './ec/models/surface_enc.pkl'
            surface_enc = pickle.load(open(file_name, 'rb'))
            transformed = surface_enc.transform(self.data['surface'])
            ohe_df = pd.DataFrame(transformed, columns=surface_enc.classes_)
            self.data = pd.concat([self.data.reset_index(drop=True), ohe_df], axis=1)

            file_name = './ec/models/base_temp_enc.pkl'
            bt_enc = pickle.load(open(file_name, 'rb'))
            transformed = bt_enc.transform(self.data['base_temperature'])
            transformed = np.hstack((1-transformed, transformed))
            ohe_df = pd.DataFrame(transformed, columns=bt_enc.classes_)
            self.data = pd.concat([self.data.reset_index(drop=True), ohe_df], axis=1)

        self._logger.info("Complete one hot encoding for features.")
            

    def targetenc_categorical(self, phase):

        """target encoding for categorical features

        Returns:
            dataframe: the dataframe with target encoded features
        """        

        categorical_features = ['base_temperature', 'surface', 'session']
        target = 'consumption'
        
        if phase == 'train':
            self.data['is_day_off'] = self.data['is_day_off'].astype(int)
            encoder = TargetEncoder(cols=categorical_features)
            transformed = encoder.fit_transform(self.data[categorical_features], self.data[target])
            transformed.columns = ['encoded_' + str(col) for col in transformed.columns]
            self.data = pd.concat([self.data.reset_index(drop=True), transformed], axis = 1)
            self._save_pkl(encoder, "target_enc.pkl")

        else:
            file_name = './ec/models/target_enc.pkl'
            encoder = pickle.load(open(file_name, 'rb'))
            transformed = encoder.fit_transform(self.data[categorical_features], self.data[target])
            transformed.columns = ['encoded_' + str(col) for col in transformed.columns]
            self.data = pd.concat([self.data.reset_index(drop=True), transformed], axis = 1)

        self._logger.info("Complete target encoding for features.")


    def _save_pkl(self, pkl, save_name):

        """save encodings as pickle file
        """     

        dir_path = "./ec/models/" + save_name
        with open(dir_path, "wb") as f:
            pickle.dump(pkl, f)
        self._logger.info("Save Complete for {}".format(dir_path))


    def feature_engineering(self, metadata, set="train"):

        """main feature engineering steps

        Returns:
            dataframe: the dataframe with added features
        """        

        self.datetime_feature()
        self.is_day_off()
        self.manual_temp_imputation()
        self.cyclical_trans()
        self.onehot_categorical(set)
        self.targetenc_categorical(set)
        if set == "train":
            self.split_train_test()


    def lag_features(self, method):

        self._logger.info("Generating lag features...")

        if method == 'Hourly':

            # last week same hour
            self.data['last_week_consumption'] = self.data['consumption']
            self.data['last_week_consumption'] = self.data.groupby(['series_id'])['last_week_consumption']\
                .transform(lambda x:x.shift(7*24))
            # previous day
            self.data['last_day_consumption'] = self.data['consumption']
            self.data['last_day_consumption'] = self.data.groupby(['series_id'])['last_day_consumption']\
                .transform(lambda x:x.shift(24))
            # last hour
            self.data['last_hour_consumption'] = self.data['consumption']
            self.data['last_hour_consumption'] = self.data.groupby(['series_id'])['last_hour_consumption']\
                .transform(lambda x:x.shift(1))

        elif method == 'Daily':
            # last week same day
            self.data['last_week_consumption'] = self.data['consumption']
            self.data['last_week_consumption'] = self.data.groupby(['series_id'])['last_week_consumption']\
                .transform(lambda x:x.shift(7))
            # previous day
            self.data['last_day_consumption'] = self.data['consumption']
            self.data['last_day_consumption'] = self.data.groupby(['series_id'])['last_day_consumption']\
                .transform(lambda x:x.shift(1))

        elif method == 'Weekly':
            # last week 
            self.data['last_week_consumption'] = self.data['consumption']
            self.data['last_week_consumption'] = self.data.groupby(['series_id'])['last_week_consumption']\
                .transform(lambda x:x.shift(1))
            # last 2 week 
            self.data['last_2week_consumption'] = self.data['consumption']
            self.data['last_2week_consumption'] = self.data.groupby(['series_id'])['last_2week_consumption']\
                .transform(lambda x:x.shift(2))
                
        else:
            print ('Invalid method')

        return self.data


    @st.cache(allow_output_mutation=True)
    def add_features(self, key):

        """Add features to the training dataframe. 

        Parameters: None

        Returns:
            dict: A dictionary containing all existing and new features. 
        """ 

        self._logger.info("Running Feature Engineering for {} dataset".format(key))

        if key == 'metadata':

            day_off_count_df = pd.merge(self.metadata['series_id'].to_frame(), \
                self.metadata.filter(like='day_off')[self.metadata.filter(like='day_off') == True], \
                    left_index=True, right_index=True).groupby('series_id').count()
            day_off_count_df["day_off_count"] = day_off_count_df.sum(axis = 1)
            self.metadata = pd.merge(self.metadata, \
                day_off_count_df.loc[:, day_off_count_df.columns.isin(["series_id", "day_off_count"])], \
                    on = "series_id") 
            self.data = self.metadata
              
        else:

            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values(by = ["series_id", "timestamp"])
            self.data = pd.merge(self.data, self.metadata.filter(regex='day_off|series_id|surface|base_temperature'), \
                on="series_id")
            self.feature_engineering(self.metadata, key)
            self.data = self.data[self.data.columns.drop(list(self.data.filter(regex='day_is_day_off')))]
            if key != 'train':
                if key == 'test':
                    self.data['Set'] = "Test_2"
                else: 
                    self.data['Set'] = key
            self.data.reset_index(inplace=True, drop=True)    
   
        return self.data

