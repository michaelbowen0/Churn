import pandas as pd
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv("customer_data.csv")

# Select relevant features for clustering
X = data[['age', 'income', 'gender', 'churn']]

# Perform KMeans clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Assign cluster labels to each customer
data['cluster'] = kmeans.labels_

# Analyze cluster characteristics
print(data.groupby('cluster').mean())
