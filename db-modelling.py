'''
    Author: Tam M Pham
    Created date: 13/02/2019
    Modified date: 13/02/2019
    Description:
        Using Gradient Boosting algorithm for bike prediction
'''

import numpy as np
import pandas as pd
import time
from common import Common
import math
from sklearn import ensemble
from sklearn.model_selection import train_test_split

start = time.time()
print("Please be patient, it might take a while...")

def refine_time(string):
    time = pd.to_datetime(string, format="%H:%M:%S")
    hour = time.strftime('%H')
    minute = "00" if time.minute < 30 else "30"
    return "%s:%s" % (hour, minute)

def define_season(string):
    date = pd.to_datetime(string, format="%Y-%m-%d")
    month = date.month
    
    if month == 11 or month == 12 or month == 1:
        return "Winter"
    elif month == 2 or month == 3 or month == 4:
        return "Spring"
    elif month == 5 or month == 6 or month == 7:
        return "Summer"
    else:
        return "Autumn"
    

# get clusters dataframe
clusters = Common.getDataFrameFromFile(Common.CLUSTERED_DATA_FILE_FULL_PATH, True)

# get all data dataframe
all_df = Common.getDataFrameFromFile(Common.CLEAN_DATA_FILE_FULL_PATH)
all_df = all_df[(all_df["Date"] >= "2016-10-14") & (all_df["Date"] <= "2017-10-14")].reset_index(drop=True)

# left merge these two dataframes together based on Number, Date and Time
merged_df = pd.merge(all_df
                    , clusters[["Number", "Time", "Cluster"]]
                    , on=["Number", "Time"]
                    , how="left")

# Calculate activity in each cluster
cluster_act = merged_df.copy()
cluster_act["Activity"] = cluster_act["Check In"] + cluster_act["Check Out"]
cluster_act = cluster_act.groupby(["Number", "Cluster"])["Activity"].sum().reset_index(name="Total Activity")

# Find the most active station per cluster
top_stations = cluster_act.copy()
top_stations = top_stations[top_stations.groupby(["Cluster"])["Total Activity"].transform(max) == top_stations["Total Activity"]].reset_index(drop=True)
print(top_stations)
# Find the least active station per cluster
bot_stations = cluster_act.copy()
bot_stations = bot_stations[bot_stations.groupby(["Cluster"])["Total Activity"].transform(min) == bot_stations["Total Activity"]].reset_index(drop=True)
print(bot_stations)

# Turn the station number of the most active station and the least active station into a list
selected = top_stations["Number"].tolist() + bot_stations["Number"].tolist()

# Randomly select other 3 stations in each cluster for the Gradient boosting modelling
for i in range(1, Common.CLUSTERING_NUMBER + 1):    # iterate throught from cluster 1 to cluster 4
    # select random 3 number which must being neither in the most active station and the least active station
    subset = merged_df[(merged_df["Cluster"] == i) & (~merged_df["Number"].isin(selected))].sample(n = 3)
    rand_list = subset["Number"].tolist()
    selected = selected + rand_list
print(selected)

############################################################################
######################## PREPARE DATA FOR MODELLING ########################
############################################################################
# get details of stations based on the selection above
time_df = merged_df[merged_df["Number"].isin(selected)].copy()

# group time into 48 factors
time_df["Time"] = time_df["Time"].apply(lambda x: refine_time(x))
time_df["Season"] = time_df["Date"].apply(lambda x: define_season(x))
time_df["av_bikes"] = time_df["Bike Stands"] - time_df["Available Bike Stands"]
time_df = time_df.groupby(["Number", "Name", "Address", "Date", "Time", "Bike Stands", "Weekday", "Season"]).agg({"av_bikes": "mean", "Cluster": "first"}).reset_index()
time_df["av_bikes"] = time_df["av_bikes"].round(0)
time_df["prev_bike_num"] = time_df.groupby(["Number", "Name", "Address", "Date"])["av_bikes"].shift(1)
time_df["prev_bike_num"] = time_df.apply(
    lambda row: row["av_bikes"] if np.isnan(row["prev_bike_num"]) else row["prev_bike_num"],
    axis=1
)
time_df["prev_bike_num2"] = time_df.groupby(["Number", "Name", "Address", "Date"])["av_bikes"].shift(3)
time_df["prev_bike_num2"] = time_df.apply(
    lambda row: row["av_bikes"] if np.isnan(row["prev_bike_num2"]) else row["prev_bike_num2"],
    axis=1
)
# convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
time_df["av_bikes"] = time_df.av_bikes.astype(np.int64)
time_df["prev_bike_num"] = time_df.prev_bike_num.astype(np.int64)

Common.saveCSV(time_df, "./time_df.csv")
#print(time_df)
#print(time_df[["Number", "Time", "Season"]])

# read CSV file containing geographical info
geo = Common.getDataFrameFromFile("./geo-data/db-geo.csv", True)
gb_df = pd.merge(time_df
                    , geo[["Number", "Latitude", "Longitude"]]
                    , on=["Number"]
                    , how="left")
Common.saveCSV(gb_df, "./gb_df.csv")

# read CSV file containing weather info
weather = Common.getDataFrameFromFile("./weather-data/M2_weather.csv", True)
weather = weather.drop_duplicates(subset=["station_id", "datetime", "AtmosphericPressure", "WindSpeed", "AirTemperature"], keep='first')
weather["datetime"] = pd.to_datetime(weather["datetime"], format="%m/%d/%Y %H:%M")
weather["Date"] = weather["datetime"].dt.strftime('%Y-%m-%d')
weather["Date"] = pd.to_datetime(weather["Date"], format="%Y-%m-%d")
weather["Time"] = weather["datetime"].dt.strftime('%H:%M:00')
weather["AirTemperature"].fillna((weather["AirTemperature"].mean()), inplace = True)
#print(weather[["Date", "Time", "AtmosphericPressure", "WindSpeed", "AirTemperature"]])
gb_df = pd.merge(gb_df
                , weather[["Date", "Time", "AtmosphericPressure", "WindSpeed", "AirTemperature"]]
                , on=["Date", "Time"]
                , how="left")


# read CSV file containing holiday info
# TODO

# Create training and test samples
n = math.floor(len(gb_df)/ 10)  # 16794
lb = 7 * n + 1  # 117559
ub = 10 * n     # 167940

train = gb_df[gb_df.index.isin(range(0, lb))].copy().reset_index(drop=True)
test = gb_df[gb_df.index.isin(range(lb, ub))].copy().reset_index(drop=True)

x = train["Time"]
y = train["av_bikes"]
xx = test["Time"]

# Fit regression model
params = {'n_estimators': 10000, 'max_depth': 6,
          'learning_rate': 0.01, 'loss': 'quantile',
          'alpha' : 0.95}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(x, y)

y_upper = clf.predict(xx)

clf.set_params(alpha=1.0 - alpha) 
y_lower = clf.predict(xx)

clf.set_params(loss='ls')
clf.fit(X, y)
y_pred = clf.predict(xx)

# Plot the function, the prediction and the 90% confidence interval based on
# the MSE
ax, fig = plt.figure()
ax.plot(x, y, 'b.', markersize=10, label=u'Observations')
ax.plot(xx, y_pred, 'r-', label=u'Prediction')
ax.plot(xx, y_upper, 'k-')
ax.plot(xx, y_lower, 'k-')
ax.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.5, fc='b', ec='None', label='90% prediction interval')
ax.set(title = "Gradient Boosting", xlabel="Time", ylabel="Average Bikes")
ax.legend(title="", bbox_to_anchor=(1.02, 0.65), loc=2, borderaxespad=0.)
plt.show()


end = time.time()
print("Done exploration after {} seconds".format((end - start)))