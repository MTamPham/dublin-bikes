'''
    Author: Tam M Pham
    Created date: 22/11/2018
    Modified date: 05/02/2019
    Description:
        Clustering Charlemont Street station from 01-10-2016
'''

import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
from common import Common
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# initialize the scaler
scaler = StandardScaler()

start = time.time()
print("Please be patient, it might take a while...")

# get the relative path of preparation data file
rel_path = os.path.relpath(Common.CLEAN_DATA_FILE_FULL_PATH)

# read CSV files using Pandas
df = pd.read_csv(rel_path, delimiter = ",", parse_dates=["Date"])

# get Charlemont Street station by filtering by the number being 5 and date being from 2016-10-01
df = df[(df["Number"] == 5) & (df["Date"] >= "2016-10-01")].reset_index()
df = df[["Check In", "Check Out", "Available Bike Stands"]]

# look at the variance of the data, if it looks similar, ignore; otherwise, the data must be scaled
print("Variance of dataframe\n", df.var())

# the variance is different, the data must be scaled (scaled is a from of standardization)
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
print("Scaled dataframe \n", scaled_df)

# DETERMINE NUMBER OF CLUSTERS FOR K-MEANS
# wss is an acronym of within sum of squares, it can be calculated by equation: sum()
wss = []
for i in range(1, 11):
    clusters = KMeans(i).fit(scaled_df)
    wss.append(clusters.inertia_)   # inertia to idenify the sum of squares of samples to the nearest cluster center
#print("wss={}", wss)

# saving plotting figure to file
clusters_df = pd.DataFrame({"num_clusters": range(1, 11), "wss": wss})
plt.plot(clusters_df.num_clusters, clusters_df.wss, marker = "o" )
plt.title('Optimal number of clusters using Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Total Within Sum of Squares')
plt.savefig(Common.PLOTS_DIR + "/cluster_analysis.png")
#plt.show()
plt.gcf().clear()

# plotting raw data
plt.scatter(df[["Check In"]], df[["Check Out"]], c='black', s=7)
plt.xlabel('Check In')
plt.ylabel('Check Out')
plt.savefig(Common.PLOTS_DIR + "/raw_charlemont.png")
#plt.show()
plt.gcf().clear()

print("It seems that 4 is the ideal number of clusters")
cluster_no = 4
# initialize K-means and fit the input data
kmeans = KMeans(n_clusters=4).fit(df)
# predict the clusters
labels = kmeans.predict(df)
# get the cluster centers
centroids = kmeans.cluster_centers_

# plotting clustered data
for i in range(cluster_no):
    clustered_data = np.array([df.iloc[j] for j in range(len(df)) if labels[j] == i])
    plt.scatter(clustered_data[:, 0], clustered_data[:, 1], cmap='rainbow', s=7)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=7, c='#050505')
plt.xlabel('Check In')
plt.ylabel('Check Out')
plt.savefig(Common.PLOTS_DIR + "/four_clusters_charlemont.png")
#plt.show()
plt.gcf().clear()

plt.close()
end = time.time()
print("Done exploration after {} seconds".format((end - start)))