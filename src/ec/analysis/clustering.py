
import streamlit as st
import numpy as np
import pandas as pd
import os 
import joblib
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns
from kmodes import kmodes

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from ec.config import Config

MODEL_DIRECTORY = "ec/models/"
OUTPUT_DIRECTORY = "ec/data/train/"
MODELS = {}
for file_name in glob(os.path.join(MODEL_DIRECTORY, '*pkl')):
    model_name = os.path.splitext(os.path.basename(file_name))[0]
    MODELS[model_name] = joblib.load(file_name)

class BuildingClustering(Config):

    def __init__(self, data, train_data, df_type):
        self.data = data
        self.train = train_data
        self.df_type = df_type

    def data_segregation(self, train_df, df, retrain=True):

        default_features = ['encoded_surface', 'encoded_hour', 'encoded_base_temperature', 'encoded_is_day_off']
        if retrain: 
            features_list = st.sidebar.multiselect('Features selection:', options = list(df.columns), default = default_features)
        else: 
            features_list = default_features

        if features_list is not None: 
            feature_traindf = train_df.loc[:, train_df.columns.isin(features_list)].values
            features_traindf = StandardScaler().fit_transform(feature_traindf)

            feature_df = df.loc[:, df.columns.isin(features_list)].values
            feature_df = StandardScaler().fit_transform(feature_df)

        return feature_traindf, feature_df

    def pca_plot(self, feature_df): 

        y = self.data[['consumption']].values
        pca = PCA()
        pca.fit(feature_df)
        plt.figure(figsize = (10,8))
        plt.plot(range(1,len(feature_df[0])+1), pca.explained_variance_ratio_.cumsum(), marker = "o", linestyle = "--")
        plt.title("Explained variance by components")
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative Explained Variance")
        st.pyplot(plt)
        final_pc = [i+1 for i, j in enumerate(pca.explained_variance_ratio_.cumsum()) \
                        if j > Config.CLUSTER_CONFIG["EXPLAINED_VARIANCETHRESHOLD"]][0]
        return final_pc
        
    def pca_build(self, feature_df, n_pc, output_file_name): 
        
        pca_model = PCA(n_components = n_pc)
        pca_model.fit(feature_df)
        os.remove(output_file_name)
        joblib.dump(pca_model, output_file_name)

    def pca_load(self, features_df):
        
        MODELS['pca_model'].fit(features_df)
        scores_pca = pd.DataFrame(MODELS['pca_model'].transform(features_df))
        scores_pca.columns = ["PC"+str(i+1) for i, j in enumerate(scores_pca.columns)]
        
        return scores_pca
        
    def kmeans_elbow(self, scores_pca, testing_cluster, method): 

        wcss = []
        for i in range(1, testing_cluster):
            kmeans_pca = KMeans(n_clusters=i, init = method, random_state = 42)
            kmeans_pca.fit(scores_pca)
            wcss.append(kmeans_pca.inertia_)
            
        plt.figure(figsize = (10,8))
        plt.plot(range(1,testing_cluster), wcss, marker = "o", linestyle = "--")
        plt.xlabel("No. of clusters")
        plt.ylabel("WCSS")
        plt.title("K-means with PCA Clustering")
        st.pyplot(plt)
        
    def kmeans_build(self, scores_pca, final_cluster, method, output_file_name, df_type): 
                
        print("Here")
        kmeans_model = KMeans(n_clusters=final_cluster, init=method, random_state=42)
        kmeans_model.fit(scores_pca)
        try: 
            os.remove(output_file_name)
        except FileNotFoundError as e:
            pass
        joblib.dump(kmeans_model, output_file_name)
        
    def kmeans_load(self, predict_pca): 
        
        predicted = MODELS['kmeans_model'].predict(predict_pca)
        scores_pca = pd.DataFrame(predict_pca)
        # scores_pca = scores_pca.loc[:, ~scores_pca.columns.isin([features_selected])]
        scores_pca['Segment'] = predicted
        
        # Plot 2 dimensional clustering
        x_axis = scores_pca['PC2']
        y_axis = scores_pca['PC1']
        plt.figure(figsize = (10,8))
        sns.scatterplot(x_axis, y_axis, hue = scores_pca['Segment']) #, palette = sns.color_palette("tab10")
        plt.title('Clusters by PCA')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        st.pyplot(plt)
        
        clustered_data = pd.concat([self.data.reset_index(drop=True), scores_pca.reset_index(drop=True)], axis = 1)
        
        return clustered_data

    def main_clustering(self, pca_check = False, kmeans_check = False, pca_model = False, kmeans_model = False):
        
        ''' Can be further revised to accomodate changes in no. of PCs and no. of k
        '''

        # Perform data segregation for train and predict 
        x_train, x_df = self.data_segregation(self.train, self.data, retrain=kmeans_model)
        print(x_df)

        # Principle component analysis
        if pca_check: 

            st.subheader("PCA plot:")
            self.pca_plot(x_train)
            st.subheader("Decide on the number of components based on explained variance > 80%:")

        if pca_model:
            user_pc = st.sidebar.number_input("Please select max no. of PCs:", 1, max_value=len(x_train[0])+1)
            button_pc = st.sidebar.button("SUBMIT", key="1")
            file_name = MODEL_DIRECTORY + "pca_model.pkl"
            if button_pc > 0: 
                button_pc = 0
                self.pca_build(x_train, user_pc, file_name)
            else:
                st.write("Using default: {}".format(Config.CLUSTER_CONFIG["DEFAULT_PC"]))
                self.pca_build(x_train, Config.CLUSTER_CONFIG["DEFAULT_PC"], file_name)

            df_pca = self.pca_load(x_df)
            train_pca = self.pca_load(x_train)
            st.write(df_pca)
        else: 
            st.write("Since retrain model is not required, we will use the previously saved model:")
            df_pca = self.pca_load(x_df)
            train_pca = self.pca_load(x_train)
            st.write(df_pca)

        # Kmeans clustering
        if kmeans_check:
            st.subheader("Decide on the number of clusters based on elbow plot:")
            user_maxclus = st.sidebar.number_input("Please select max no. of clusters for plotting:", 1, \
                                                        max_value=Config.CLUSTER_CONFIG["MAX_CLUSTERS_ELBOW"])
            button_kmeansplot = st.sidebar.button("SUBMIT", key="2")
            if button_kmeansplot > 0: 
                button_kmeansplot = 0
                self.kmeans_elbow(train_pca, user_maxclus, "k-means++")
            else:
                st.write("Using default: {}".format(Config.CLUSTER_CONFIG["MAX_CLUSTERS_ELBOW"]))
                self.kmeans_elbow(train_pca, Config.CLUSTER_CONFIG["MAX_CLUSTERS_ELBOW"], "k-means++")
            
        # Build kmeans model
        if kmeans_model:
            user_clus = st.sidebar.number_input("Please select no. of clusters based on elbow:", 1, \
                                                    Config.CLUSTER_CONFIG["MAX_CLUSTERS_ELBOW"])
            button_kmeans = st.sidebar.button("SUBMIT", key="3")
            file_name = MODEL_DIRECTORY + "kmeans_model.pkl"
            if button_kmeans > 0: 
                button_kmeans = 0
                print("Here")
                self.kmeans_build(train_pca, user_clus, "k-means++",file_name)
                # MODELS["kmeans_model"] = joblib.load(file_name)
                final_df = self.kmeans_load(df_pca)
                final_df.to_csv(os.path.join(OUTPUT_DIRECTORY, "clustering_data_{}.csv".format(self.df_type)))

        else: 
            st.write("Since retrain model is not required, we will use the previously saved model:")
            final_df = self.kmeans_load(df_pca)
            final_df.to_csv(os.path.join(OUTPUT_DIRECTORY, "clustering_data_{}.csv".format(self.df_type)))
            
        
        

        