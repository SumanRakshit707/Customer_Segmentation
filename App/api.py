#Code for FastAPI
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.cluster import KMeans


app = FastAPI(title='Customer Segmentation', version='1.0', description='KMeans model is used for segmentation')
df=pd.read_csv('new.csv')
class CustomerData(BaseModel):
    age: int
    annual_income: int
    spent_in_week: int
    
x=df.iloc[:,[5,7,10]].values

#For creating 4 clusters 
num_clusters = 4
kmeans_model = KMeans(n_clusters=num_clusters, init="k-means++", random_state=0)
kmeans_model.fit(x)

@app.post('/predict/')
async def predict_customer_segmentation(df: CustomerData):
    # Convert input data to a NumPy array and predict the cluster
    input_data = np.array([[df.age, df.annual_income, df.spent_in_week]])
    cluster = kmeans_model.predict(input_data)[0]

    # Define the cluster names
    cluster_names = ['Sensible Customers', 'Valuable Customer', 'Target Customer', 'High Value Customer']

    # Get the cluster name corresponding to the predicted cluster
    cluster_name = cluster_names[cluster]
    
    cluster = int(cluster)

    return {
        "cluster_number": cluster,
        "cluster_name": cluster_name
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    #uvicorn api:app --reload
