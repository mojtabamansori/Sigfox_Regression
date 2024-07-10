import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


dataset = np.array(pd.read_csv('../Dataset/Orginal.csv'))
Input = dataset[:, :137]
Output = dataset[:, 138:]

scaler = StandardScaler()
scaled_Input = scaler.fit_transform(Input)

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(scaled_Input)
kmeans_cluster_labels = kmeans.labels_
print(np.unique(kmeans_cluster_labels))
plt.figure(figsize=(15, 6))
plt.scatter(Output[:, 0], Output[:, 1], c=kmeans_cluster_labels[:], cmap='tab10', alpha=0.7)

plt.xlabel('First Column of Output (x)')
plt.ylabel('Second Column of Output (y)')
plt.title('K-means Clustering (2 Clusters)')

plt.tight_layout()
plt.show()
