import streamlit as st
#import pandas as pd
import numpy as np
import pickle

# Load the pre-trained KMeans model from the .pkl file
#with open('kmeans_model.pkl', 'rb') as f:
kmeans_model = pickle.load('kmeans_model.pkl')

# Cluster names
cluster_names = ['Target Customers', 'High Value Customer', 'Valuable Customer', 'Sensible Customer']

# Streamlit app title and description
st.title('Customer Segmentation')
st.write('This Project is done by Suman Rakshit')
st.write('Enter customer data below and click "Predict" to get the segmentation result. Criteria for the data should be Age(18-70),Annual Income(10000-300000)dollars and Spent in week(100-1000)dollars.')

# Streamlit input fields for customer data
gender=st.selectbox('Gender',['option','Male','Female'])
age = st.text_input('Age')
annual_income = st.text_input('Annual Income')
spent_in_week = st.text_input('Spent in a Week')

# Streamlit button to trigger prediction
if st.button('Predict'):
    # Prepare the customer data as a NumPy array
    input_data = np.array([[age, annual_income, spent_in_week]])

    # Predict the cluster using the loaded KMeans model
    cluster = kmeans_model.predict(input_data)[0]
    cluster_name = cluster_names[cluster]

    # Display the segmentation result
    st.success(f'Segmentation Result: Cluster {cluster} - {cluster_name}')
