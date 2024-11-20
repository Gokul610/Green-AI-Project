import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
st.title("EV Charging Station Optimization")

df = pd.read_csv(r'station_data_dataverse.csv')

# checking columns
required_columns = ['locationId', 'reportedZip', 'kwhTotal', 'sessionId']
if not all(col in df.columns for col in required_columns):
    st.error(f"Missing required columns: {set(required_columns) - set(df.columns)}")
    st.stop()

# Data processing
location_summary = df.groupby(['locationId', 'reportedZip']).agg(
    total_kwh=('kwhTotal', 'sum'),
    session_count=('sessionId', 'count')
).reset_index()

scaler = StandardScaler()
location_summary[['total_kwh', 'session_count']] = scaler.fit_transform(
    location_summary[['total_kwh', 'session_count']]
)

# Elbow Method 
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(location_summary[['total_kwh', 'session_count']])
    inertia.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), inertia)
ax1.set_title("Elbow Method For Optimal k")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Inertia")

# KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
location_summary['cluster'] = kmeans.fit_predict(
    location_summary[['total_kwh', 'session_count']]
)

sil_score = silhouette_score(
    location_summary[['total_kwh', 'session_count']],
    location_summary['cluster']
)
st.subheader("Silhouette Score")
st.write(f"Silhouette Score: {sil_score:.4f}")

# visualization
pca = PCA(n_components=2)
location_summary[['pca1', 'pca2']] = pca.fit_transform(
    location_summary[['total_kwh', 'session_count']]
)

st.subheader("Geographic Clustering with PCA")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x='pca1', y='pca2', hue='cluster',
    data=location_summary, palette='tab10', ax=ax2
)
ax2.set_title("Geographic Clustering with PCA")
ax2.set_xlabel("total_kwh")
ax2.set_ylabel("session_count")
st.pyplot(fig2)
