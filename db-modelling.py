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

# Configure settings to display full data
#pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
#pd.set_option('display.max_colwidth', -1)  # or 199

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
time_df["Avg Bikes"] = time_df["Bike Stands"] - time_df["Available Bike Stands"]
time_df = time_df.groupby(["Number", "Name", "Address", "Date", "Time", "Bike Stands", "Weekday", "Season"]).agg({"Avg Bikes": "mean", "Cluster": "first"}).reset_index()
time_df["Avg Bikes"] = time_df["Avg Bikes"].round(0)
time_df["Prev Bikes"] = time_df.groupby(["Number", "Name", "Address", "Date"])["Avg Bikes"].shift(1)
time_df["Prev Bikes"] = time_df.apply(
    lambda row: row["Avg Bikes"] if np.isnan(row["Prev Bikes"]) else row["Prev Bikes"],
    axis=1
)
# convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
time_df["Avg Bikes"] = time_df["Avg Bikes"].astype(np.int64)
time_df["Prev Bikes"] = time_df["Prev Bikes"].astype(np.int64)

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
#gb_df["datetime"] = gb_df["Date"] + ' ' + gb_df["Time"]
#gb_df["timestamp"] = pd.to_datetime(gb_df["datetime"], format="%Y-%m-%d %H:%M:%S").values.astype(np.int64) / 1000000
#gb_df = gb_df.drop(["datetime"], axis = 1)
gb_df["Weekday Code"] = pd.to_datetime(gb_df["Date"], format=Common.DATE_FORMAT).dt.weekday
# label encoding for weekdays, time and season
le_season = preprocessing.LabelEncoder()
gb_df["Season Code"] = le_season.fit_transform(gb_df["Season"])
le_time = preprocessing.LabelEncoder()
gb_df["Time Code"] = le_time.fit_transform(gb_df["Time"])
# binary encode
#ohe = preprocessing.OneHotEncoder()

# read CSV file containing holiday info
# TODO

######################################################################
######### TRAINING MODEL USING GRADIENT BOOSTING ALGORITHM ###########
######################################################################
gb_important_df = gb_df[Common.IMPORTANT_FACTORS].copy()
#Common.saveCSV(gb_important_df, Common.CLEAN_DATA_DIR + "/gb_train_test_set.csv")
#x = np.atleast_2d(gb_df.loc[:, gb_important_df.columns != 'Avg Bikes'].values).T
x = gb_important_df.loc[:, gb_important_df.columns != 'Avg Bikes'].values
y = gb_important_df["Avg Bikes"].values
xx = time.time()

# Create training and testing samples with 75% for training set, 25% for testing set using library
seed = 7
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)
# feed training data to Gradient Boosting model
model.fit(x_train, y_train)
# save model
joblib.dump(model, Common.CLEAN_DATA_DIR + "/gb_model.csv")

# Plot feature importance
feature_importance = model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, gb_important_df.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
#plt.savefig(Common.PLOTS_DIR + "/feature_importance.png")
plt.gcf().clear()

######################################################################
################ TESTING OUR GRADIENT BOOSTING MODEL #################
######################################################################
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)

df_test = pd.DataFrame(x_test, columns=["Weekday Code", "Time Code", "Prev Bikes", "Cluster", "Latitude", "Longitude", "Season Code", "WindSpeed", "AirTemperature"])
df_test["Time"] = le_time.inverse_transform(df_test["Time Code"].astype(np.int64))
df_test["Time"] = pd.to_datetime(df_test["Time"], format="%H:%M:%S")
df_test = df_test.drop(["Time Code"], axis = 1)
df_test["Avg Bikes"] = y_test
df_test["pred"] = y_pred.round(0).astype(np.int64)
df_test = pd.merge(df_test, gb_df[["Number", "Address", "Bike Stands"]], how = "left", left_index = True, right_index = True)
df_test = df_test.groupby(["Number", "Address", "Time"]).agg({"Avg Bikes": "mean", "pred": "mean", "Bike Stands": "max"}).reset_index()
#print(df_test.dtypes)

# get station numbers in testing set
station_numbers = df_test["Number"].unique()
print("Station numbers in testing set: " ,station_numbers)
# calculate number of stations in testing set
n_stations = len(station_numbers)

n_station_row = round(n_stations / Common.MAX_AXES_ROW)
n_station_row = n_station_row + 1 if n_stations % Common.MAX_AXES_ROW > 0 else n_station_row
#print(f"We need to generate a figure with {n_station_row} rows")

index = 0
fig, axes = plt.subplots(figsize = (8, 7), nrows = n_station_row, ncols = Common.MAX_AXES_ROW, sharex = True, sharey= True, constrained_layout=False)
for row in axes:
    for ax in row:
        #print(f"Rendering in {index}")
        if index >= n_stations:
            #fig.delaxes(ax)
            #revious_ax_index = (i - 1) * Common.MAX_AXES_ROW + j
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
        ax_x = df_test[condition]["Time"]
        ax_y1 = df_test[condition]["Avg Bikes"]
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
fig.subplots_adjust(hspace=0.2)
# Set common labels
fig.text(0.5, 0.12, "Time", ha='center', va='center', fontsize="medium")
fig.text(0.06, 0.5, "Mean Available Bikes", ha='center', va='center', rotation='vertical', fontsize="medium")
# plot the legend
fig.legend(handles, labels, title="Color", loc='center', bbox_to_anchor=(0.5, 0.06, 0., 0.), ncol=4)
#fig.savefig(Common.PLOTS_DIR + "/prediction.png")
plt.gcf().clear()

######################################################################
############################## UNSEEN DATA ###########################
######################################################################
unseen_all_df = Common.getDataFrameFromFile(Common.CLEAN_DATA_FILE_FULL_PATH, True)
unseen_all_df = unseen_all_df[(unseen_all_df["Date"] >= "2017-11-06") & (unseen_all_df["Date"] <= "2017-11-13")].reset_index(drop=True)
if len(unseen_all_df) <= 0:
    print("There is no data after 2017-11-06")
    end = time.time()
    print("Done exploration after {} seconds".format((end - start)))
    sys.exit()

# left merge these two dataframes together based on Number, Date and Time
unseen_merged_df = pd.merge(unseen_all_df
                    , clusters[["Number", "Time", "Cluster"]]
                    , on=["Number", "Time"]
                    , how="left")

# get details of stations based on the selection above
unseen_time_df = unseen_merged_df.copy()

# group time into 48 factors
unseen_time_df["Time"] = unseen_time_df["Time"].apply(lambda x: Common.refineTime(x))
unseen_time_df["Season"] = unseen_time_df["Date"].apply(lambda x: Common.defineSeason(x))
unseen_time_df["Avg Bikes"] = unseen_time_df["Bike Stands"] - unseen_time_df["Available Bike Stands"]
unseen_time_df = unseen_time_df.groupby(["Number", "Name", "Address", "Date", "Time", "Bike Stands", "Weekday", "Season"]).agg({"Avg Bikes": "mean", "Cluster": "first"}).reset_index()
unseen_time_df["Avg Bikes"] = unseen_time_df["Avg Bikes"].round(0)
unseen_time_df["Prev Bikes"] = unseen_time_df.groupby(["Number", "Name", "Address", "Date"])["Avg Bikes"].shift(1)
unseen_time_df["Prev Bikes"] = unseen_time_df.apply(
    lambda row: row["Avg Bikes"] if np.isnan(row["Prev Bikes"]) else row["Prev Bikes"],
    axis=1
)
# convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
unseen_time_df["Avg Bikes"] = unseen_time_df["Avg Bikes"].astype(np.int64)
unseen_time_df["Prev Bikes"] = unseen_time_df["Prev Bikes"].astype(np.int64)

unseen_gb_df = pd.merge(unseen_time_df
                    , geo[["Number", "Latitude", "Longitude"]]
                    , on=["Number"]
                    , how="left")
# build important factors and formula to predict the bike number
unseen_gb_df = pd.merge(unseen_gb_df
                , weather[["Date", "Time", "AtmosphericPressure", "WindSpeed", "AirTemperature"]]
                , on=["Date", "Time"]
                , how="left")
unseen_gb_df["AtmosphericPressure"].fillna((unseen_gb_df["AtmosphericPressure"].mean()), inplace = True)
unseen_gb_df["WindSpeed"].fillna((unseen_gb_df["WindSpeed"].mean()), inplace = True)
unseen_gb_df["AirTemperature"].fillna((unseen_gb_df["AirTemperature"].mean()), inplace = True)
unseen_gb_df["Weekday Code"] = pd.to_datetime(unseen_gb_df["Date"], format=Common.DATE_FORMAT).dt.weekday
# label encoding for weekdays, time and season
le_season = preprocessing.LabelEncoder()
unseen_gb_df["Season Code"] = le_season.fit_transform(unseen_gb_df["Season"])
le_time = preprocessing.LabelEncoder()
unseen_gb_df["Time Code"] = le_time.fit_transform(unseen_gb_df["Time"])
# convert time string to time to plot data
unseen_gb_df["Time"] = pd.to_datetime(unseen_gb_df["Time"], format="%H:%M:%S")

# get station numbers in unseen set
station_numbers = unseen_gb_df["Number"].unique()
print("Station numbers in unseen set: ", station_numbers)
# calculate number of stations in testing set
n_stations = len(station_numbers)

# load Gradient Boosting model
model = joblib.load(Common.CLEAN_DATA_DIR + "/gb_model.csv")

##################################################################
######################## PREDICT UNSEEN DATA #####################
##################################################################
# only select important factors
unseen_gb_important_df = unseen_gb_df[Common.IMPORTANT_FACTORS].copy()
#Common.saveCSV(unseen_gb_important_df, Common.CLEAN_DATA_DIR + "/gb_unseen_set.csv")

# make a copy of unseen dataset to another dataframe, except avg bikes column
validate_df = unseen_gb_important_df.loc[:, unseen_gb_important_df.columns != 'Avg Bikes'].copy()
# add predicted result into unseen dataset
unseen_gb_df["pred"] = model.predict(validate_df).round(0).astype(np.int64)

mse = mean_squared_error(unseen_gb_important_df["Avg Bikes"], unseen_gb_df["pred"])
rmse = math.sqrt(mse)
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)

unseen_gb_df = unseen_gb_df.groupby(["Number", "Address", "Time", "Weekday"]).agg({"Avg Bikes": "mean", "pred": "mean", "Bike Stands": "max"}).reset_index()
unseen_gb_df["Avg Bikes"] = unseen_gb_df["Avg Bikes"].round(0).astype(np.int64)
unseen_gb_df["pred"] = unseen_gb_df["pred"].round(0).astype(np.int64)
#Common.saveCSV(unseen_gb_df, "./unseen_gb_df.csv")
# plot by weekdays
n_wdays = len(Common.SHORT_WEEKDAY_ORDER)
n_wday_row = round(n_wdays / Common.MAX_AXES_ROW)
n_wday_row = n_wday_row + 1 if (n_wdays % Common.MAX_AXES_ROW) > 0 else n_wday_row
for i in range(1, Common.MAX_STATION_NUMBER + 1):
    index = 0
    fig, axes = plt.subplots(figsize = (8, 7), nrows = n_wday_row, ncols = Common.MAX_AXES_ROW, sharex = True, sharey= True, constrained_layout=False)
    # real index of current station in array
    j = -1
    try:
        j = station_numbers.tolist().index(i)
    except:
        print(f"Not found {i}")
    if j == -1:
        continue
    fig_title = unseen_gb_df[unseen_gb_df.Number == station_numbers[j]].Address.unique()[0]
    # no data of current station number, move to the next station number
    if (len(unseen_gb_df[unseen_gb_df.Number == station_numbers[j]]) <= 0):
        print(f"No data for plotting station {fig_title}")
        continue
    for row in axes:
        for ax in row:
            if index >= n_wdays:
                # locate sticks every 1 hour
                ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
                # show locate label with hour and minute format
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                # set smaller size for tick labels
                ax.xaxis.set_tick_params(labelsize=7)
                # increase index of next station by 1 before continuing
                index += 1
                continue
            condition = (unseen_gb_df["Number"] == station_numbers[j]) & (unseen_gb_df["Weekday"] == Common.SHORT_WEEKDAY_ORDER[index])
            ax_x = unseen_gb_df[condition]["Time"]
            ax_y1 = unseen_gb_df[condition]["Avg Bikes"]
            ax_y2 = unseen_gb_df[condition]["pred"]
            ax_y3 = unseen_gb_df[condition]["Bike Stands"]
            ax.plot(ax_x, ax_y1, "b-", label=u'Actual')
            ax.plot(ax_x, ax_y2, "r-", label=u'Predicted')
            ax.plot(ax_x, ax_y3, "-.", color = 'black', label=u'Bike Stands')
            #print(ax_x.dtypes)
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
            ax_title = unseen_gb_df[condition]["Weekday"].unique()[0]
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
    fig.suptitle(f"Predictions for unseen data in {fig_title} from week of 06/11/17")
    fig.subplots_adjust(hspace=0.2)
    # Set common labels
    fig.text(0.5, 0.12, "Time", ha='center', va='center', fontsize="medium")
    fig.text(0.06, 0.5, "Mean Available Bikes", ha='center', va='center', rotation='vertical', fontsize="medium")
    # plot the legend
    fig.legend(handles, labels, title="Color", loc='center', bbox_to_anchor=(0.5, 0.06, 0., 0.), ncol=4)
    #fig.savefig(f"{Common.PREDICTING_PLOTS_DIR}/unseen_prediction_{i}.png")
    plt.close()

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

end = time.time()
print("Done exploration after {} seconds".format((end - start)))