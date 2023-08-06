# Code for Streamlit UI

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df=pd.read_csv('new.csv')
x=df.iloc[:,[5,7,10]].values
# Pre-trained KMeans model
kmeans_model = KMeans(n_clusters=4, init="k-means++", random_state=0)
kmeans_model.fit(x)
# Cluster names
cluster_names = ['Sensible Customers', 'High Value Customer', 'Target Customer', 'Valuable Customer']

# Streamlit app title and description
st.title('Customer Segmentation')
st.write('Enter customer data below and click "Predict" to get the segmentation result.')

# Streamlit input fields for customer data
gender = st.selectbox ("Gender",['Choosen Option','Male','Female'])
age = st.text_input('Age')
spent_in_week = st.text_input('Spent in a Week')
annual_income = st.text_input('Annual Income')


# Streamlit button to trigger prediction
if st.button('Predict'):
    # Prepare the customer data as a NumPy array
    input_data = np.array([[age, spent_in_week, annual_income]])

    # Predict the cluster using the KMeans model
    cluster = kmeans_model.predict(input_data)[0]
    cluster_name = cluster_names[cluster]
    cluster = int(cluster)
     # Display the segmentation result
    st.success(f'Segmentation Result: Cluster  {cluster} <===> {cluster_name}')

    #streamlit run file_name.py/ streamlit run main.py
