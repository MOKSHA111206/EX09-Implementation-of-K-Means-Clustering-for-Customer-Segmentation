# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Data Preparation: Load customer data (features such as age, income, spending score) and preprocess it (handle missing values, scale features).

2.Model Initialization: Import KMeans from sklearn.cluster and initialize the model with a specified number of clusters (k).

3.Model Fitting: Fit the model to the customer data using kmeans.fit(data), which groups customers into clusters based on similarities.

4.Results Visualization: Use a scatter plot to visualize the clusters, labeling each point according to its assigned cluster.

## Program:
```

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: AMMINENI MOKSHASREE
RegisterNumber:  2305001001

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
dt=pd.read_csv("/content/Mall_Customers_EX8.csv")
dt
x=dt[['Annual Income (k$)','Spending Score (1-100)']]
plt.figure(figsize=(4,4))
plt.scatter(dt['Annual Income (k$)'],dt['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(x)
centroids=Kmeans.cluster_centers_
labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/9cb1ebf2-0145-490d-8929-da3d75bb6491)
![image](https://github.com/user-attachments/assets/9b4be9e5-b352-4ec7-b444-092d2c7dafbb)
![image](https://github.com/user-attachments/assets/0f895a09-9219-4ead-9471-bf8f9eb754c8)
![image](https://github.com/user-attachments/assets/bc4376a2-b8ef-483a-bc88-dc0b5428b3df)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
