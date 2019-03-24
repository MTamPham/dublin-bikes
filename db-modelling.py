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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from sklearn import preprocessing   # label encoder
from sklearn import ensemble        # library of Gradient Boosting
from sklearn.model_selection import train_test_split    # split data to training set and tesing set
from sklearn.metrics import mean_squared_error      # calculate MSE
from sklearn.externals import joblib    # for saving and loading model
import sys

start = time.time()
print("Please be patient, it might take a while...")
    
# get clusters dataframe
clusters = Common.getDataFrameFromFile(Common.CLUSTERED_DATA_FILE_FULL_PATH, True)

# get all data dataframe
all_df = Common.getDataFrameFromFile(Common.CLEAN_DATA_FILE_FULL_PATH, True)
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
print("Stations selected randomly is ", selected)

############################################################################
######################## PREPARE DATA FOR MODELLING ########################
############################################################################
# get details of stations based on the selection above
time_df = merged_df[merged_df["Number"].isin(selected)].copy()

# group time into 48 factors
time_df["Time"] = time_df["Time"].apply(lambda x: Common.refineTime(x))
time_df["Season"] = time_df["Date"].apply(lambda x: Common.defineSeason(x))
time_df[Common.PREDICTING_FACTOR] = time_df["Available Stands"]
time_df = time_df.groupby(["Number", "Name", "Address", "Date", "Time", "Bike Stands", "Weekday", "Season"]).agg({Common.PREDICTING_FACTOR: "mean", "Cluster": "first"}).reset_index()
time_df[Common.PREVIOUS_PREDICTING_FACTOR] = time_df.groupby(["Number", "Name", "Address", "Date"])[Common.PREDICTING_FACTOR].shift(1)
time_df[Common.PREVIOUS_PREDICTING_FACTOR] = time_df.apply(
    lambda row: row[Common.PREDICTING_FACTOR] if np.isnan(row[Common.PREVIOUS_PREDICTING_FACTOR]) else row[Common.PREVIOUS_PREDICTING_FACTOR],
    axis=1
)
# convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
time_df[Common.PREDICTING_FACTOR] = time_df[Common.PREDICTING_FACTOR].astype(np.int64)
time_df[Common.PREVIOUS_PREDICTING_FACTOR] = time_df[Common.PREVIOUS_PREDICTING_FACTOR].astype(np.int64)

# read CSV file containing geographical info
geo = Common.getDataFrameFromFile("./geo-data/db-geo.csv", True)
gb_df = pd.merge(time_df
                    , geo[["Number", "Latitude", "Longitude"]]
                    , on=["Number"]
                    , how="left")

# read CSV file containing weather info
weather = Common.getDataFrameFromFile("./weather-data/M2_weather.csv", True)
weather = weather.drop_duplicates(subset=["station_id", "datetime", "AtmosphericPressure", "WindSpeed", "AirTemperature"], keep='first')
weather["datetime"] = pd.to_datetime(weather["datetime"], format="%m/%d/%Y %H:%M")
weather["Date"] = weather["datetime"].dt.strftime(Common.DATE_FORMAT)
weather["Time"] = weather["datetime"].dt.strftime(Common.TIME_FORMAT)

# build important factors and formula to predict the bike number
gb_df = pd.merge(gb_df
                , weather[["Date", "Time", "AtmosphericPressure", "WindSpeed", "AirTemperature"]]
                , on=["Date", "Time"]
                , how="left")
gb_df["AtmosphericPressure"].fillna((gb_df["AtmosphericPressure"].mean()), inplace = True)
gb_df["WindSpeed"].fillna((gb_df["WindSpeed"].mean()), inplace = True)
gb_df["AirTemperature"].fillna((gb_df["AirTemperature"].mean()), inplace = True)
gb_df["Weekday Code"] = pd.to_datetime(gb_df["Date"], format=Common.DATE_FORMAT).dt.weekday
# label encoding for weekdays, time and season
le_season = preprocessing.LabelEncoder()
gb_df["Season Code"] = le_season.fit_transform(gb_df["Season"])
le_time = preprocessing.LabelEncoder()
gb_df["Time Code"] = le_time.fit_transform(gb_df["Time"])
#Common.saveCSV(gb_df, "./gb_df.csv")
#print(f"Data has {len(gb_df)} rows")

# read CSV file containing holiday info
# TODO

######################################################################
######### TRAINING MODEL USING GRADIENT BOOSTING ALGORITHM ###########
######################################################################
# Create training and testing samples with 67% for training set, 33% for testing set using library
seed = 7
test_size = 0.33
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)

x = gb_df[Common.CONSIDERING_FACTORS].copy()
y = gb_df[Common.PREDICTING_FACTOR].copy()
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=test_size, random_state=seed)

# feed training data to Gradient Boosting model
model.fit(x_train, y_train)

# Plot feature importance
feature_importance = model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig, ax = plt.subplots()
ax.barh(pos, feature_importance[sorted_idx], align='center')
ax.set(title = 'Variable Importance',xlabel = 'Relative Importance')
plt.yticks(pos, x.columns[sorted_idx])
# set margins
plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)
fig.savefig(Common.PREDICTING_PLOTS_DIR + "/feature_importance.png")
fig.clear()

# after viewing feature importance, weather information doesn't impact the result
# so take it out
x = gb_df[Common.IMPORTANT_FACTORS].copy()
y = gb_df[Common.PREDICTING_FACTOR].copy()
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=test_size, random_state=seed)

# feed training data to Gradient Boosting model
model.fit(x_train, y_train)

# save model
joblib.dump(model, Common.GRADIENT_BOOSTING_MODEL_FULL_PATH)

######################################################################
################ TESTING OUR GRADIENT BOOSTING MODEL #################
######################################################################
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)

df_test = pd.DataFrame(x_test, columns=Common.IMPORTANT_FACTORS)
df_test["Time"] = le_time.inverse_transform(df_test["Time Code"].astype(np.int64))
df_test = df_test.drop(["Time Code"], axis = 1)
df_test[Common.PREDICTING_FACTOR] = y_test
df_test["pred"] = y_pred.round(0).astype(np.int64)

df_test = pd.merge(df_test
                    , gb_df[["Number", "Address", "Bike Stands", "Latitude", "Longitude", "Time"]]
                    , how="left"
                    , on=["Latitude", "Longitude", "Time"])
df_test = df_test.groupby(["Number", "Address", "Time"]).agg({Common.PREDICTING_FACTOR: "mean", "pred": "mean", "Bike Stands": "max"}).reset_index()
#print(df_test.dtypes)

# get station numbers in testing set
station_numbers = df_test["Number"].unique()
print("Station numbers in testing set: " ,station_numbers)
# calculate number of stations in testing set
n_stations = len(station_numbers)
n_station_row = round(n_stations / Common.MAX_AXES_ROW)
n_station_row = n_station_row + 1 if n_station_row * Common.MAX_AXES_ROW < n_stations else n_station_row
print(f"We need to generate a figure with {n_station_row} rows for {n_stations}")

index = 0
fig, axes = plt.subplots(figsize = (12, 10), nrows = n_station_row, ncols = Common.MAX_AXES_ROW, sharex = True, sharey= True, constrained_layout=False)
for row in axes:
    for ax in row:
        #print(f"Rendering in {index}")
        if index >= n_stations:
            # locate sticks every 1 hour
            ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
            # show locate label with hour and minute format
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # set smaller size for tick labels
            ax.xaxis.set_tick_params(labelsize=7)
            # increase index of next station by 1 before continuing
            index += 1
            continue
        condition = df_test["Number"] == station_numbers[index]
        ax_x = pd.to_datetime(df_test[condition]["Time"], format="%H:%M:%S")
        ax_y1 = df_test[condition][Common.PREDICTING_FACTOR]
        ax_y2 = df_test[condition]["pred"]
        ax_y3 = df_test[condition]["Bike Stands"]
        ax.plot(ax_x, ax_y1, "b-", label='Actual')
        ax.plot(ax_x, ax_y2, "r-", label='Predicted')
        ax.plot(ax_x, ax_y3, "-.", color = 'black', label='Bike Stands')
        ax.fill_between(ax_x.dt.to_pydatetime(), ax_y2 - rmse, ax_y2 + rmse, facecolor='#3a3a3a', alpha=0.5)
        y_min = 0
        y_max = all_df["Bike Stands"].max()
        ax.set_ylim([y_min, y_max])
        # locate sticks every 1 hour
        ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
        # show locate label with hour and minute format
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        # set smaller size for tick labels
        ax.xaxis.set_tick_params(labelsize=7)
        # set title for each axe
        ax_title = df_test[condition]["Address"].unique()[0]
        ax.set_title(ax_title)
        # margin x at 0 and y at 0.1
        ax.margins(x=0.0, y=0.1)
        ax.grid(linestyle="-")
        # increase index of next station by 1
        index += 1
        handles, labels = ax.get_legend_handles_labels()

# show rotate tick lables automatically with 90 degree
fig.autofmt_xdate(rotation = "90")
# set title of the figure
fig.suptitle("Gradient Boosting prediction and actual number")
fig.subplots_adjust(hspace=0.6)
# Set common labels
fig.text(0.5, 0.12, "Time", ha='center', va='center', fontsize="medium")
fig.text(0.06, 0.5, "Mean Available Stands", ha='center', va='center', rotation='vertical', fontsize="medium")
# plot the legend
fig.legend(handles, labels, title="Color", loc='center', bbox_to_anchor=(0.5, 0.06, 0., 0.), ncol=4)
fig.savefig(Common.PREDICTING_PLOTS_DIR + "/prediction.png")
fig.clear()

end = time.time()
print("Done exploration after {} seconds".format((end - start)))
sys.exit()

###################################################################
##################### TODO ###########################
###################################################################
hr_unseen_gb_df = unseen_gb_df[(unseen_gb_df["Weekday"] == "Mon") \
                    & (unseen_gb_df["Time"].dt.strftime(Common.TIME_FORMAT) == "08:00:00")].copy().reset_index(drop=True)

hr_unseen_gb_df["error"] = rmse
hr_unseen_gb_df["max"] = round(hr_unseen_gb_df["pred"] + hr_unseen_gb_df["error"]).astype(np.int64)
hr_unseen_gb_df["min"] = round(hr_unseen_gb_df["pred"] - hr_unseen_gb_df["error"]).astype(np.int64)
hr_unseen_gb_df.loc[hr_unseen_gb_df["min"] < 0, "min"] = 0
hr_unseen_gb_df["diff"] = hr_unseen_gb_df["Avg Bikes"] - hr_unseen_gb_df["pred"]
hr_unseen_gb_df["max_diff"] = np.negative(hr_unseen_gb_df["Avg Bikes"] - hr_unseen_gb_df["max"])
hr_unseen_gb_df["min_diff"] = hr_unseen_gb_df["Avg Bikes"] - hr_unseen_gb_df["min"]
hr_unseen_gb_df[hr_unseen_gb_df["min_diff"] < 0, "min_diff"] = np.negative(hr_unseen_gb_df["min_diff"])
'''
hr_unseen_gb_df["Evaluate Pred"] = hr_unseen_gb_df.apply(lambda x: "Sufficient" if x["Avg Bikes"] >= x["lower_bound"] and \
                            x["Avg Bikes"] <= x["upper_bound"] \
                            else "Oversupply" if x["Avg Bikes"] > x["upper_bound"] \
                            else "Insufficient")
hr_unseen_gb_df["Station Range"] = "1-25" if hr_unseen_gb_df["Number"] < 26 \
                            else "26-50" if hr_unseen_gb_df["Number"] < 51 \
                            else "51-75" if hr_unseen_gb_df["Number"] < 76 \
                            else "76-102"
                            '''

Common.saveCSV(hr_unseen_gb_df, "./hr_unseen_gb_df.csv")
#sys.exit()

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(hr_unseen_gb_df["Number"], hr_unseen_gb_df["Avg Bikes"], yerr=[hr_unseen_gb_df["min_diff"], hr_unseen_gb_df["max_diff"]], fmt='.k')
#ax.plot(hr_unseen_gb_df["Avg Bikes"])
fig.savefig("./inventory.png")
plt.gcf().clear()

