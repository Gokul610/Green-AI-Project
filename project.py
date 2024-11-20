# import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
# load dataset
df = pd.read_csv(r'C:\Users\ELCOT\Desktop\gkproject\station_data_dataverse.csv')
print("Dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
# checking columns
required_columns = ['locationId', 'reportedZip', 'kwhTotal', 'sessionId']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing required columns: {set(required_columns) - set(df.columns)}")
#normalize data
location_summary = df.groupby(['locationId', 'reportedZip']).agg(total_kwh=('kwhTotal', 'sum'),session_count=('sessionId', 'count')).reset_index()
#standardize features
scaler = StandardScaler()
location_summary[['total_kwh', 'session_count']] = scaler.fit_transform(location_summary[['total_kwh', 'session_count']])
#optimal cluster finding using elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(location_summary[['total_kwh', 'session_count']])
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
#clustering
n_clusters = 5  # Adjust based on your analysis
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
location_summary['cluster'] = kmeans.fit_predict(location_summary[['total_kwh', 'session_count']])
#Evaluation metric
sil_score = silhouette_score(location_summary[['total_kwh', 'session_count']],location_summary['cluster'])
print(f"Silhouette Score: {sil_score:.4f}")
#visualization using PCA(principal component analysis)
pca = PCA(n_components=2)
location_summary[['pca1', 'pca2']] = pca.fit_transform(location_summary[['total_kwh', 'session_count']])
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster',data=location_summary, palette='tab10')
plt.title("Geographic Clustering with PCA")
plt.xlabel("total_kwh")
plt.ylabel("session_count")
plt.legend(title="Cluster")
plt.show()

#Save the cluster
location_summary.to_csv('location_clusters.csv', index=False)
print("\nCluster assignments saved to 'location_clusters.csv'.")

# geographic cluster analysis
if 'reportedZip' in df.columns:
    print("\nGeographic Cluster Analysis:")
    zip_summary = location_summary.groupby('cluster').agg(avg_zip=('reportedZip', 'mean'),count=('reportedZip', 'count')).reset_index()
    print(zip_summary)

location_summary.to_csv('location_clusters.csv', index=False)
print("\nCluster assignments saved as 'location_clusters.csv'.")
#save the model
with open('geographic_clustering_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print("\nKMeans clustering model saved as 'geographic_clustering_model.pkl'.")
#load the model
loaded_model=joblib.load('geographic_clustering_model.pkl')
joblib.dump(loaded_model, 'geographic_clustering_model.pkl')
print('model sucessfully loaded')