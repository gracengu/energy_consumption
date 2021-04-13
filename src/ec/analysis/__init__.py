import streamlit as st
import pandas as pd
import missingno as msno

import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler

from ec.config import Config
from ec.analysis.impute import Imputation

# Check missingness in data
class Analysis(Config):

    st = __import__('streamlit')

    def __init__(self, data):
        self.data = data

    @st.cache
    def datatype_conversion(self):

        """
        Converts datatype of columns from imported data. 

            Parameters: 
            data (dataframe): imported data

            Returns: 
            Dataframe to be used for feature selection 
        """

        category_cols       = self.FEATURE_DEFINITION["category_cols"]
        integer_cols        = self.FEATURE_DEFINITION["integer_cols"]
        float_cols          = self.FEATURE_DEFINITION["float_cols"]
        datetime_cols       = self.FEATURE_DEFINITION["datetime_cols"]
        data                = self.data
        
        data[category_cols] = data[category_cols].astype('category',copy=False) 
        data[integer_cols] = data[integer_cols].astype('int64',copy=False)
        data[float_cols] = data[float_cols].astype('float64',copy=False)
        data[datetime_cols] = data[datetime_cols].astype('datetime64[ns]',copy=False)

        return data
        
    def summary_stats(self):
            
        '''
        Pretty print the statistics of the data imported. If number of columns exceed 10, table will be transposed. 

            Parameters: 
            df_raw(dataframe): Raw dataframe with converted datatypes.  

            Returns: 
            Summary of statistics(dataframe): DateTime data: A dataframe consisting of XXX ;
                                              Numerical data: A dataframe consisting of count, mean, sd, quartiles, min, and max; 
                                              Categorical data: A dataframe consisting of count, unique, top and frequency. 
        '''
        # Return outputs according to format
        numerical = self.data.select_dtypes(include=["int","float"])
        datetime = self.data.select_dtypes(include=["category"])
        categorical = self.data.select_dtypes(include=["datetime"])

        if not datetime.empty :
            df_stats_datetime = datetime.describe(datetime_is_numeric=True)
            if (len(datetime) > 10):
                df_stats_datetime = pd.DataFrame(df_stats_datetime.transpose())
            st.write("Datetime data \n ")
            st.write(df_stats_datetime.astype('object'))

        if not numerical.empty:
            df_stats_num = numerical.describe()
            if (len(numerical) > 10):
                df_stats_num = pd.DataFrame(df_stats_num.transpose())
            st.write("Numeric data \n")
            st.dataframe(df_stats_num.style.format("{:.2f}"))

        if not categorical.empty :
            df_stats_cat = categorical.describe()
            if (len(categorical) > 10):
                df_stats_cat = pd.DataFrame(df_stats_cat.transpose())
            st.write("Categorical data ")
            st.write(df_stats_cat.astype('object'))

    def missingness_stats(self):

        st.write("Summary of Missing Values Per Column:")
        missing_col = self.data.isnull().sum()
        percent_missing_col = round(missing_col * 100 / len(self.data),2)
        df_missing_col = pd.DataFrame({'No. of missing values per column': missing_col, 'Missingness (%)': percent_missing_col})
        df_missing_col.sort_values('No. of missing values per column', ascending=False, inplace=True)
        st.dataframe(df_missing_col.astype("object"))

        st.write("Missingness Plot:")
        fig, ax = plt.subplots()
        ax.barh(self.data.columns,missing_col.sort_values())
        ax.set_xlabel('No. of missing values')
        ax.set_ylabel("Column's name")
        st.pyplot(fig)

    def missingno_matrix(self, df, fontsize, time_freq):

        """Missing value matrix on dataframe

        Visualize the pattern on missing values between columns

        Parameters
        ----------
        df : str
        var : int
            If False, the x-axis and y-axis labels will not be displayed.

        Returns
        -------
        fig : object
            Missing values percentage matrix for each variables
        """

        df.index = pd.to_datetime(df["timestamp"], errors='coerce')
        df = df.resample('D').mean()
        fig, ax = plt.subplots(figsize=(17,8))
        ax = msno.matrix(df, labels=True, fontsize=fontsize, freq=time_freq, ax=ax, sparkline=True, inline=True);
        st.pyplot(fig)

    def cols_missing_pect(self, df, first_index):
        """Acquiring number of missing values across each variables

        Prepare a dataframe on amount of missing values in percentage of each variables in each string

        Parameters
        ----------
        df : object
            Input dataframe
        first_index : datetime
            First date where the data point for the variable is acquired

        Returns
        -------
        missing_df : object
            Dataframe on percentage of missing values for each variables
        """
        cols = Config.FEATURE_DEFINITION['integer_cols'] + \
               Config.FEATURE_DEFINITION['float_cols'] + \
               Config.FEATURE_DEFINITION['category_cols']
        missing_df = pd.DataFrame(columns=cols)

        for building, data in df.groupby('series_id'):
            fig, ax = plt.subplots(figsize=(7,5))
            data = data[cols + ['timestamp']]
            min_date = data[first_index].first_valid_index()
            if min_date:
                data = data[data.index >= min_date]
                data = data.reset_index(drop=True).resample('M', on='timestamp').first().drop(columns=["timestamp"])
                string_missing_df = (data.isnull().sum() * 100 / len(data))
                string_missing_df['series_id'] = building
                missing_df = missing_df.append(string_missing_df, ignore_index=True)
        missing_df = missing_df.set_index('series_id')
        
        return missing_df
        
    def heatmap_plot(self, df, plot_title, rotate=None):
        """Heatmap plot on missing value percentage

        Generate a heatmap that show the percentage of missing values of all variables based on the ID,  
        in this project, it will be "STRINGS"

        Parameters
        ----------
        df : object
            Input dataframe
        plot_title : str
            Title of the heatmap plot
        rotate : int
            Degree of x-axis label to be rotate, if the labels are too long, better to rotate

        Returns
        -------
        fig : object
            Heatmap chart
        """
        fig, ax = plt.subplots(figsize=(40,25)) 

        sns.heatmap(df, cmap='coolwarm', linewidth=0.1, annot=True, ax=ax)
        _ = plt.xlabel('COLUMNS', fontsize=13, weight='bold')
        _ = plt.ylabel('BUILDING ID', fontsize=13, weight='bold')
        _ = plt.title(plot_title, fontsize=17, weight='bold')
        _ = ax.tick_params(top=True, labeltop=True)
        _ = plt.xticks(rotation=rotate)
        _ = plt.show()
    
        st.pyplot(fig)

    def eda_metadata(self):

        st.write("No. of buildings based on surface:")
        fig, ax = plt.subplots()
        ax.bar(self.data['surface'].unique(),self.data['surface'].value_counts())
        ax.set_xlabel('Building Sizes')
        ax.set_ylabel("Building Counts")
        st.pyplot(fig)

        st.write(r"No. of buildings based on base temperature ($^\circ$C):")
        fig, ax = plt.subplots()
        ax.bar(self.data['base_temperature'].unique(),self.data['base_temperature'].value_counts())
        ax.set_xlabel('Base Temperature Modes')
        ax.set_ylabel("Building Counts")
        st.pyplot(fig)

        st.write("No. of buildings based on day off count:")
        fig, ax = plt.subplots()
        ax.bar(self.data['day_off_count'].unique(),self.data['day_off_count'].value_counts())
        ax.set_xlabel('Day Off Counts')
        ax.set_ylabel("Building Counts")
        st.pyplot(fig)

        st.write("No. of buildings based on building size and base temperature:")
        fig = sns.catplot(y="surface", hue="base_temperature", kind="count", data=self.data)
        st.pyplot(fig)

        st.write("No. of buildings based building size and day off counts:")
        fig = sns.catplot(y="surface", hue="day_off_count", kind="count", data=self.data)
        st.pyplot(fig)

    def eda_temperate(self):

        # st.write("Box plot for temperature based on base temperature:")
        # fig = sns.boxplot(x="base_temperature", y="temperature", data=self.data)
        # ax.fig.savefig("doc/boxplot_temp.png")
        # st.pyplot(fig)

        fig = px.box(self.data, x='base_temperature', y='temperature', hover_name='series_id')
        fig.update_layout(template='seaborn',title='Distribution of temperature based on base temperature')
        st.plotly_chart(fig)


        fig = px.scatter(self.data, x='temperature', color='base_temperature', y='consumption', hover_name='series_id')
        fig.update_layout(template='seaborn',title='Distribution of consumption across base_temperature and temperature')
        st.plotly_chart(fig)
        
        grouping = self.data.groupby(['surface','base_temperature','series_id'],as_index=False)['consumption'].mean().dropna()
        fig = px.box(grouping, x='surface', color='base_temperature', y='consumption', hover_name='series_id', category_orders = self.CATEGORICAL_ORDERS)
        fig.update_layout(template='seaborn',title='Distribution of consumption across base_temperature and surface')
        st.plotly_chart(fig)
        

        grouping = self.data.groupby(['surface','is_day_off','series_id'],as_index=False)['consumption'].mean().dropna()
        fig = px.box(grouping, x='surface', color='is_day_off', y='consumption', hover_name='series_id', category_orders = self.CATEGORICAL_ORDERS)
        fig.update_layout(template='seaborn',title='Distribution of consumption across is_day_off and surface')
        st.plotly_chart(fig)
        
        
        grouping = self.data.groupby(['surface','day_off_count','series_id'],as_index=False)['consumption'].mean().dropna()
        fig = px.box(grouping, x='surface', color='day_off_count', y='consumption', hover_name='series_id', category_orders = self.CATEGORICAL_ORDERS)
        fig.update_layout(template='seaborn',title='Distribution of consumption across day_off_count and surface')
        st.plotly_chart(fig)

    def min_max_scaling(self):

        scaler = MinMaxScaler()
        numerical_independent = self.data.loc[:, self.data.columns.isin(Config.FEATURE_DEFINITION["float_cols"])]
        numerical_normalised = pd.DataFrame(scaler.fit_transform(numerical_independent), 
                                            columns = numerical_independent.columns)

        return numerical_normalised

    def one_hot_encoding(self):

        """
        Performs one hot encoding for categorical features for both dependent and indepedent variables. 
        This is defined as static method as it will be used by embedded and wrapper method as well. 

        Args: 
        data (dataframe): output from read_input() 
        categorical (): categorical independent variables as defined by users
        dependent (dataframe): categorical dependent variable (as defined by users) column from data

        Returns: 
        df_dependent_enc (dataframe): Encoded categorical dependent variable
        df_catindependent_enc (dataframe): Encoded categorical independent variable

        Possible improvement: 
        Use 'pass' in exception would be preferred but not sure how best to return output. 

        """
    
        try: 

            # Encode dependent variable
            le = LabelEncoder()
            le.fit(self.data["consumption"])
            df_dependent_enc = pd.DataFrame(le.transform(self.data["consumption"]))

            # Encode independent variable
            categorical_features = Config.FEATURE_DEFINITION["category_cols"]
            categorical_df = self.data.loc[:, self.data.columns.isin(categorical_features)]
            oe = OrdinalEncoder()
            oe.fit(categorical_df)
            df_catindependent_enc = pd.DataFrame(oe.transform(categorical_df))
            df_catindependent_enc.columns = categorical_df.columns

        except KeyError as e: 

            st.write("Cannot perform one-hot encoding for numerical variables. Please check if variables are properly defined.")
            st.write(self.data.columns != "consumption")
            df_dependent_enc = []
            df_catindependent_enc = []

        else:
            
            return  df_dependent_enc, df_catindependent_enc

    def missing_data_disabled(self, data):
            
        data = data.replace('?', np.NaN)
        missingval = data.isnull().values.any() 
        percent_miss = self.DEFAULT_SUMMARY_CONFIG["missing_percentage"]

        if missingval == False:
            st.write("\nThere is no missing value in the data.\n")
        else:

            st.markdown('''\nFeature selection requires clean data with no missing values!
            We will remove missing data but to avoid loss of information,
            you may refer to this link for ways in missingness treatment:
            https://www.kaggle.com/parulpandey/a-guide-to-handling-missing-values-in-python.''')
        
            st.write("Summary of Missing Values Per Column:")
            missing_col = data.isnull().sum()
            percent_missing_col = round(missing_col * 100 / len(data),2)
            df_missing_col = pd.DataFrame({'No. of missing values per column': missing_col, 'Missingness (%)': percent_missing_col})
            df_missing_col.sort_values('No. of missing values per column', ascending=False, inplace=True)
            st.write(df_missing_col.astype("object"))
            sns.set_palette(sns.color_palette(self.PETRONAS_COLOR_PALETTE.values()))
            plt.figure(figsize=self.PLOT_CONFIG["sns_size"])
            ax = missing_col.sort_values().plot(kind='barh')
            plt.xlabel('No. of missing values', fontsize = self.PLOT_CONFIG["figure_labelsize"])
            plt.ylabel("Column's name", fontsize = self.PLOT_CONFIG["figure_labelsize"])
            fig = plt.show()
            st.pyplot(fig)
                    
            if (df_missing_col['Missingness (%)']>percent_miss).any() == True:
                st.write("Columns with more than {}% missing values will be remove as follows:\n".format(percent_miss))
                a = df_missing_col[df_missing_col['Missingness (%)'] > percent_miss]
                a = list(a.index)            
                st.write("\nShape of original data: {}".format(data.shape))
                data_test = data.copy()
                data_test.drop(data[a], inplace = True, axis = 1)
            
                if data_test.isnull().values.any() == True:
                    st.write("\nNow, missing rows will be deleted.\n")
                    data_test.dropna(axis=0, how='any', inplace=True)
                    data_test.reset_index(drop=True,inplace=True)
                    st.write("Shape of final clean data: {}".format(data_test.shape))      
                else:
                    st.write("Shape of final clean data: {}".format(data_test.shape)) 
            else:
                st.write("There is no columns with more than {}% missing values. Now, missing rows will be deleted.\n".format(percent_miss))
                data_test = data.copy()
                data_test.dropna(axis=0, how='any', inplace=True)
                data_test.reset_index(drop=True,inplace=True)
                st.write("Shape of original data: {}".format(data.shape))
                st.write("Shape of clean data: {}".format(data_test.shape))

            data = data_test

        categorical = pd.DataFrame()
        numerical = pd.DataFrame()
        datetime = pd.DataFrame()

        for col in list(data.columns):
            if col in list(self.categorical.columns):
                categorical = pd.concat([categorical, data[col]],axis=1)  
            elif col in list(self.numerical.columns):
                numerical = pd.concat([numerical, data[col]],axis=1)
            elif col in list(self.datetime.columns): 
                datetime = pd.concat([datetime, data[col]],axis=1)

        for col in list(data.columns):
            if col in self.dependent.name: 
                df_dependent = pd.Series(data[col])

        df_independent = data.drop(columns = df_dependent.name)