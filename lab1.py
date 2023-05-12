import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from sklearn.cluster import KMeans
from sklearn import metrics

print('\nMade by Ilia\n')
# Load data from input file
X=[]
input_file = open('lab01.csv')
for row in input_file:
    X.append(list(map(float,row.split(';'))))
    
# Convert to numpy array
X = np.array(X)


# Estimate the bandwidth of X
bandwidth_X = estimate_bandwidth(X, quantile=0.15, n_samples=len(X))

# Cluster data with MeanShift
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Extract the centers of clusters
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# Estimate the number of clusters
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

scores = []

values = np.arange(2, 15)
for num_clusters in values:
    # Train the KMeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    score = metrics.silhouette_score(X, kmeans.labels_, 
                metric='euclidean', sample_size=len(X))

    print("\nNumber of clusters =", num_clusters)
    print("Silhouette score =", score)
                    
    scores.append(score)

plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Silhouette score vs number of clusters')
# Plot the points and cluster centers
plt.figure()
markers = 'o*xvs3p'
for i, marker in zip(range(num_clusters), markers):
    # Plot points that belong to the current cluster
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='black')

    # Plot the cluster center
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o', 
            markerfacecolor='black', markeredgecolor='black', 
            markersize=15)

step_size = 0.01

# Define the range of values for the grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), 
        np.arange(y_min, y_max, step_size))

# Predict the cluster labels for each point in the grid
Z = meanshift_model.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
Z = Z.reshape(x_vals.shape)

# Plot the boundaries of the clusters
plt.figure()
plt.clf()
plt.imshow(Z, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), 
    y_vals.min(), y_vals.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')

# Overlay the input points
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', 
        edgecolors='black', s=80)

# Plot the centers of the clusters
cluster_centers = meanshift_model.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
        marker='o', s=210, linewidths=4, color='black', 
        zorder=12, facecolors='black')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title('Boundaries of clusters')
plt.show()
