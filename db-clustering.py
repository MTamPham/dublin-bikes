'''
    Author: Tam M Pham
    Created date: 22/11/2018
    Modified date: 05/02/2019
    Description:
        Prepare data for analyzing
'''

import os
import numpy as np
import pandas as pd
import time
from common import Common
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import sys

start = time.time()
print("Please be patient, it might take a while...")

def fill_na(df):
    # iterate through rows
    for i, row in df.iterrows():
        # iterate through columns of the current row
        for j, column in row.iteritems():
            # if the current row is the first row and it has N/A value, fill in with the next non-N/A value
            if (i == 0 and np.isnan(df.loc[i,j])):
                k = i+1
                # iterate to find the next non-N/A value
                while (np.isnan(df.loc[k,j])):
                    k = k+1
                df.at[i,j] = df.at[k,j]
            elif (np.isnan(df.at[i,j])):    # if the current row is other rows and it has N/A value, fill in with the previous value
              df.at[i,j] = df.at[i-1,j]
            else:   # keep its value
                df.at[i,j] = df.at[i,j]
    return df

def spread_data_by_time(df):
    "Spread the dataframe by time i.e. get values of time and make it become columns"
    # group data by Number, Date and Time
    df = df.set_index(["Number", "Date", "Time"])["Available Bike Stands"]
    "Set multi-index"
    #print(df)
    # then unstack the data frame, reset index
    df = df.unstack().reset_index()
    "Unstack the dataframe"
    #print(df)
    # remove first column of data frame since it is multi-indexing
    df = df.rename_axis(None, axis=1)
    # drop columns Number and Date
    df = df.drop(["Number", "Date"], axis=1)
    # apply method to fill N/A values
    df = fill_na(df)
    "Fill N/A values"
    #print(df)
    return df

# define a function to calculate the total within-cluster sum of square
def calc_wss(df):
    row_len = len(df.index)
    wss = []    # within-cluster sum of squares
    wss.append((row_len-1) * df.var().sum())   # this equation comes from http://statmethods.net/advstats/cluster.html

    for i in range(2, 15):
        clusters = KMeans(i).fit(df)
        wss.append(clusters.inertia_)
    return wss

def fit_data_using_kmeans(df, cluster_no, time_lvls):
    # initialize K-means and fit the input data
    kfit = KMeans(n_clusters=cluster_no).fit(df)
    # predict the clusters
    labels = kfit.predict(df)
    # get the cluster centers
    centroids = kfit.cluster_centers_
    # format cluster centers to dateframe for viewing
    centroids_df = pd.DataFrame(centroids, columns=time_lvls)
    centroids_df["Cluster"] = centroids_df.index + 1
    # reshape the data for plotting
    centroids_df = pd.melt(centroids_df, id_vars=["Cluster"], var_name="Time", value_name="Available Bike Stands")
    # convert Time from string to datetime to set x-ticks locator, it wouldn't work if Time is an object (string)
    # the best way to convert string to time is pd.to_datetime(centroids_df["Time"], format="%H:%M:%S").dt.time, 
    # but tick locator requires full datetime to work (TODO: research how to solve this problem later)
    centroids_df["Time"] = pd.to_datetime(centroids_df["Time"], format="%H:%M:%S")
    return centroids_df

# get the relative path of preparation data file
rel_path = os.path.relpath(Common.CLEAN_DATA_FILE_FULL_PATH)

# read CSV files using Pandas
df = pd.read_csv(rel_path, delimiter = ",", parse_dates=["Date"])

# remove unwanted data
df = df[(df["Date"] >= "2016-10-14") & (df["Date"] <= "2017-10-14")].reset_index()

#######################################################################
############# DATA HANDLER FOR CLUSTERING ALL STATIONS ################
#######################################################################
# clone data frame from db_all_data.csv to another dataframe to pre-processing
prep_df = df[["Number", "Date", "Time", "Available Bike Stands"]].copy()
# spread the data
prep_df = spread_data_by_time(prep_df)

# the time levels is used for reshaping the data
time_lvls = df[["Time"]].drop_duplicates(keep='first').sort_values(by=["Time"]).reset_index(drop=True)["Time"]

# create clustering plot folder if it isn't existing
Common.createFolder(Common.CLUSTERING_PLOTS_DIR)

######################################################################
######## FIND THE IDEAL NUMBER OF CLUSTER USING ELBOW METHOD #########
######################################################################
# row counts = 18919, wss = [394283112.1580415, 271741519.98348534, 208182110.88657007, 170175067.4365119, 153912663.7234805, 141350949.25061303, 132525368.99288647, 124489926.74327439, 118145017.14839986, 112996845.35149214]
wss = calc_wss(prep_df)

clusters_df = pd.DataFrame({"num_clusters": range(1, 15), "wss": wss})
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(clusters_df.num_clusters, clusters_df.wss, marker = "o" )
ax.set(title = "Optimal number of clusters using Elbow method", xlabel="Number of Clusters", ylabel="Total Within Sum of Squares")
fig.savefig(Common.CLUSTERING_PLOTS_DIR + "/wss_all_stations.png")
fig.clear()

####################################################################
############### FIT THE DATA USING K-MEANS ALGORITHM ###############
####################################################################
centroids_df = fit_data_using_kmeans(prep_df, Common.CLUSTERING_NUMBER, time_lvls)

###################################################################
############ PLOT THE DATA FOR CLUSTERING ALL STATIONS ############
###################################################################
print("Plotting K-Means of all stations during the week")
# plot available bike stands based on cluster number
fig, ax = plt.subplots(figsize=(8, 7))
for label, cluster_df in centroids_df.groupby("Cluster"):
    ax.plot(cluster_df["Time"], cluster_df["Available Bike Stands"], label=label)
# locate sticks every 1 hour
ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
# show locate label with hour and minute format
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# show rotate tick lables automatically with 30 degree, it would show 90 degree if we don't call it
fig.autofmt_xdate()
# set title, xlable and ylabel of the figure
ax.set(title="Clustering for all stations", xlabel="Hour of day", ylabel="Available Bike Stands")
# set location of legend
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 0.65), loc=2, borderaxespad=0.)
# show grid
ax.grid(linestyle="-")
# margin x at 0 and y at 0.1
ax.margins(x=0.0, y=0.1)
# set margins
plt.subplots_adjust(left=0.09, right=0.85, top=0.95, bottom=0.1)
# save the plot to file
fig.savefig(Common.CLUSTERING_PLOTS_DIR + "/clustering_all_stations.png")
fig.clear()

###################################################################
############### COMPARISON BETWEEN WEEKDAYS/ WEEKEND ##############
###################################################################
weekdays = pd.Series(np.array(["Mon", "Tue", "Wed", "Thu", "Fri"]))
#################### WEEKDAYS ###################
weekday_df = df[df["Weekday"].isin(weekdays)][["Number", "Date", "Time", "Available Bike Stands"]]
# spread the data
weekday_df = spread_data_by_time(weekday_df)
# row counts = 10771, wss = [230530144.25717294, 145539618.7550903, 113843491.11600238, 95199552.71731418, 87781998.91295272, 81464846.73866414, 76266754.45641534, 71781659.30167598, 68189449.59577079, 65251957.952212185, 63045303.854908444, 61009697.83267804, 59363827.75259201, 57647244.66527195]
wss = calc_wss(weekday_df)

clusters_df = pd.DataFrame({"num_clusters": range(1, 15), "wss": wss})
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(clusters_df.num_clusters, clusters_df.wss, marker = "o" )
ax.set(title = "Weekdays (Mon - Fri) optimal number of clusters", xlabel="Number of Clusters", ylabel="Total Within Sum of Squares")
fig.savefig(Common.CLUSTERING_PLOTS_DIR + "/wss_weekdays.png")
fig.clear()

centroids_df = fit_data_using_kmeans(weekday_df, Common.CLUSTERING_NUMBER, time_lvls)

print("Plotting K-Means of all stations during weekdays")
# plot available bike stands based on cluster number
fig, ax = plt.subplots(figsize=(8, 7))
for label, cluster_df in centroids_df.groupby("Cluster"):
    ax.plot(cluster_df["Time"], cluster_df["Available Bike Stands"], label=label)
# locate sticks every 1 hour
ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
# show locate label with hour and minute format
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# show rotate tick lables automatically with 30 degree, it would show 90 degree if we don't call it
fig.autofmt_xdate()
# set title, xlable and ylabel of the figure
ax.set(title="Weekdays Clustering (Mon - Fri)", xlabel="Hour of day", ylabel="Available Bike Stands")
# set location of legend
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 0.65), loc=2, borderaxespad=0.)
# show grid
ax.grid(linestyle="-")
# margin x at 0 and y at 0.1
ax.margins(x=0.0, y=0.1)
# set margins
plt.subplots_adjust(left=0.09, right=0.85, top=0.95, bottom=0.1)
# save the plot to file
fig.savefig(Common.CLUSTERING_PLOTS_DIR + "/clustering_weekdays.png")
fig.clear()

#################### WEEKENDS ###################
weekend_df = df[~df["Weekday"].isin(weekdays)][["Number", "Date", "Time", "Available Bike Stands"]]
# spread the data
weekend_df = spread_data_by_time(weekend_df)
# row counts = 8148, wss = [230530144.25717294, 145539618.7550903, 113843491.11600238, 95199552.71731418, 87781998.91295272, 81464846.73866414, 76266754.45641534, 71781659.30167598, 68189449.59577079, 65251957.952212185, 63045303.854908444, 61009697.83267804, 59363827.75259201, 57647244.66527195]
wss = calc_wss(weekend_df)

clusters_df = pd.DataFrame({"num_clusters": range(1, 15), "wss": wss})
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(clusters_df.num_clusters, clusters_df.wss, marker = "o" )
ax.set(title = "Weekends (Sat - Sun) optimal number of clusters", xlabel="Number of Clusters", ylabel="Total Within Sum of Squares")
fig.savefig(Common.CLUSTERING_PLOTS_DIR + "/wss_weekends.png")
fig.clear()

centroids_df = fit_data_using_kmeans(weekend_df, Common.CLUSTERING_NUMBER, time_lvls)

print("Plotting K-Means of all stations during weekends")
# plot available bike stands based on cluster number
fig, ax = plt.subplots(figsize=(8, 7))
for label, cluster_df in centroids_df.groupby("Cluster"):
    ax.plot(cluster_df["Time"], cluster_df["Available Bike Stands"], label=label)
# locate sticks every 1 hour
ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
# show locate label with hour and minute format
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# show rotate tick lables automatically with 30 degree, it would show 90 degree if we don't call it
fig.autofmt_xdate()
# set title, xlable and ylabel of the figure
ax.set(title="Weekends Clustering (Sat - Sun)", xlabel="Hour of day", ylabel="Available Bike Stands")
# set location of legend
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 0.65), loc=2, borderaxespad=0.)
# show grid
ax.grid(linestyle="-")
# margin x at 0 and y at 0.1
ax.margins(x=0.0, y=0.1)
# set margins
plt.subplots_adjust(left=0.09, right=0.85, top=0.95, bottom=0.1)
# save the plot to file
fig.savefig(Common.CLUSTERING_PLOTS_DIR + "/clustering_weekends.png")
fig.clear()

##############################################################################
######### FIND THE SUITABLE CLUSTER NUMBER BELONGS TO EACH STATION ###########
##############################################################################
centers = fit_data_using_kmeans(prep_df, Common.CLUSTERING_NUMBER, time_lvls)
centers = centers.groupby(["Cluster"])["Available Bike Stands"].sum().reset_index() # 1649.796156, 4065.806215, 3160.653014, 2417.590592
centers = centers.rename(columns={"Available Bike Stands": "Sum Of Stands"})

agg_df = df[["Number", "Name", "Time", "Available Bike Stands"]].copy()
agg_df = agg_df.groupby(["Number", "Name", "Time"])["Available Bike Stands"].mean().reset_index()
agg_df = agg_df.rename(columns={"Available Bike Stands": "Stands"})
agg_df["Cluster"] = "None"

# Loop through each station and find the cluster which is closer to it
for i in range(1, 103):
    # Get the sum of stands for each station
    sum_stands = agg_df[agg_df["Number"] == i]["Stands"].sum()

    min_diff = centers.copy()
    min_diff["Station Sum"] = sum_stands
    min_diff["Diff"] = np.abs(min_diff["Sum Of Stands"] - min_diff["Station Sum"])
    min_diff = min_diff.sort_values(by=["Diff"]).reset_index(drop=True).head(1)

    agg_df.loc[agg_df["Number"] == i, "Cluster"] = min_diff["Cluster"].values[0]

Common.goToSubDirectory(Common.CLEAN_DATA_DIR)
Common.saveCSV(agg_df, "./db_clustered_stations.csv")

end = time.time()
print("Done exploration after {} seconds".format((end - start)))