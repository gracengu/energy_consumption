import numpy as np
import pandas as pd
from category_encoders import TargetEncoder

from ec.config import Config

class FeatureEngineering(Config):

    st = __import__('streamlit')

    def __init__(self, data_dict):
        self.data_dict  = data_dict
        self.metadata   = data_dict['metadata']
        self.keys       = ['train', 'test', 'prediction', 'metadata']

    @staticmethod
    def add_is_day_off(row, cols, iter):

        return  np.where((row[cols] == True) & (row['weekday'] == iter), True, row['is_day_off'])

    @staticmethod
    def part_of_day(x):

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
    
    @st.cache(allow_output_mutation=True)
    def add_features(self):

        updated_dict = {}

        for key in self.keys: 

            if key == 'metadata':

                day_off_count_df = pd.merge(self.metadata['series_id'].to_frame(), self.metadata.filter(like='day_off')[self.metadata.filter(like='day_off') == True], left_index=True, right_index=True).groupby('series_id').count()
                day_off_count_df["day_off_count"] = day_off_count_df.sum(axis = 1)
                data = pd.merge(self.metadata, day_off_count_df.loc[:, day_off_count_df.columns.isin(["series_id", "day_off_count"])], on = "series_id")   
                updated_dict[key] = data

            else:

                data = self.data_dict.get(key)
            
                # change datatype
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # sort data 
                data = data.sort_values(by = ["series_id", "timestamp"])
                    
                # add hour 
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                
                # add day
                data['day'] = pd.to_datetime(data['timestamp']).dt.day
                
                # add weekday
                data['weekday'] = pd.to_datetime(data['timestamp']).dt.dayofweek
                
                # add week
                data['week'] = pd.to_datetime(data['timestamp']).dt.isocalendar().week
                
                # add month
                data['month'] = pd.to_datetime(data['timestamp']).dt.month
                
                # add year
                data['year'] = pd.to_datetime(data['timestamp']).dt.year

                # merge with metadata
                data = pd.merge(data, self.metadata.filter(regex='day_off|series_id|surface|base_temperature'), on="series_id")
                
                # add day_off_count
                doc = pd.merge(self.metadata['series_id'].to_frame(), self.metadata.filter(like='day_off')[self.metadata.filter(like='day_off') == True], left_index=True, right_index=True).groupby('series_id').count()
                doc["day_off_count"] = doc.sum(axis = 1)
                data = pd.merge(data, doc.loc[:, doc.columns.isin(["series_id", "day_off_count"])], on = "series_id")
                
                # add is_day_off
                temp = list(data.filter(regex = "is_day_off").columns)
                data['is_day_off'] = False
                for cols in temp: 
                    data['is_day_off'] = data.apply(self.add_is_day_off, axis = 1, args = [cols, temp.index(cols)])
                    
                # add session
                data['session'] = data['hour'].apply(self.part_of_day)
                
                # add is_night    
                data['is_night'] = data.apply(lambda x: x.session in ['Night', 'Late Night'], axis=1)
                
                # target encode features 
                data['is_day_off'] = data['is_day_off'].astype(int)
                features_encode = ['is_day_off', 'day_off_count', 'base_temperature', 'surface', 'hour', 'month', 'session', 'is_night']
                encoder = TargetEncoder(cols=features_encode)
                enc_features = encoder.fit_transform(data[features_encode],data['consumption'])
                enc_features.columns = ['encoded_' + str(col) for col in enc_features.columns]
                data = pd.concat([data, enc_features], axis = 1)
                    
                # remove metadata
                data = data[data.columns.drop(list(data.filter(regex='day_is_day_off')))]

                # add column to identify if train test or prediction
                data['data'] = key
                    
                updated_dict[key] = data
    
        return updated_dict