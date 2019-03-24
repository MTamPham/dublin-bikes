'''
    Author: Tam M Pham
    Created date: 18/03/2019
    Modified date: 18/03/2019
    Description:
        Scattering predict available bike stand values 
        and actual available bike stand values
'''

import os
import numpy as np
import pandas as pd
import time
from common import Common
import sys
import math
import threading
import fnmatch
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.externals import joblib    # for saving and loading model
from sklearn import preprocessing   # label encoder
from sklearn.metrics import mean_squared_error      # calculate MSE

start = time.time()

def setBikeStands(number):
    if (number == 79):
        total_stands = 27
    elif (number == 5):
        total_stands = 40
    elif (number == 100):
        total_stands = 25
    elif (number == 66):
        total_stands = 40
    else:
        total_stands = 23
    return total_stands

def calcBikes(tota_stands_amount, stands_amount, bikes_amount):
    if (np.isnan(bikes_amount)):
        return bikes_amount
    else:
        return tota_stands_amount - stands_amount

# get the current working directory
working_dir = os.getcwd()

# load Gradient Boosting model
model = joblib.load(Common.GRADIENT_BOOSTING_MODEL_FULL_PATH)

# get clusters dataframe
clusters = Common.getDataFrameFromFile(Common.CLUSTERED_DATA_FILE_FULL_PATH, True)

# read CSV file containing weather info
weather = Common.getDataFrameFromFile("./weather-data/M2_weather.csv", True)
weather = weather.drop_duplicates(subset=["station_id", "datetime", "AtmosphericPressure", "WindSpeed", "AirTemperature"], keep='first')
weather["datetime"] = pd.to_datetime(weather["datetime"], format="%m/%d/%Y %H:%M")
weather["Date"] = weather["datetime"].dt.strftime(Common.DATE_FORMAT)
weather["Time"] = weather["datetime"].dt.strftime(Common.TIME_FORMAT)

# read CSV file containing geographical info
geo = Common.getDataFrameFromFile("./geo-data/db-geo.csv", True)

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
unseen_time_df[Common.PREDICTING_FACTOR] = unseen_time_df["Available Stands"]
unseen_time_df = unseen_time_df.groupby(["Number", "Name", "Address", "Date", "Time", "Bike Stands", "Weekday", "Season"]).agg({Common.PREDICTING_FACTOR: "mean", "Cluster": "first"}).reset_index()
unseen_time_df[Common.PREDICTING_FACTOR] = unseen_time_df[Common.PREDICTING_FACTOR].round(0)
unseen_time_df[Common.PREVIOUS_PREDICTING_FACTOR] = unseen_time_df.groupby(["Number", "Name", "Address", "Date"])[Common.PREDICTING_FACTOR].shift(1)
unseen_time_df[Common.PREVIOUS_PREDICTING_FACTOR] = unseen_time_df.apply(
    lambda row: row[Common.PREDICTING_FACTOR] if np.isnan(row[Common.PREVIOUS_PREDICTING_FACTOR]) else row[Common.PREVIOUS_PREDICTING_FACTOR],
    axis=1
)
# convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
unseen_time_df[Common.PREDICTING_FACTOR] = unseen_time_df[Common.PREDICTING_FACTOR].astype(np.int64)
unseen_time_df[Common.PREVIOUS_PREDICTING_FACTOR] = unseen_time_df[Common.PREVIOUS_PREDICTING_FACTOR].astype(np.int64)

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
model = joblib.load(Common.GRADIENT_BOOSTING_MODEL_FULL_PATH)

##################################################################
######################## PREDICT UNSEEN DATA #####################
##################################################################
# add predicted result into unseen dataset
unseen_gb_df["pred"] = model.predict(unseen_gb_df[Common.IMPORTANT_FACTORS]).round(0).astype(np.int64)

mse = mean_squared_error(unseen_gb_df[Common.PREDICTING_FACTOR], unseen_gb_df["pred"])
rmse = math.sqrt(mse)
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)

unseen_gb_df = unseen_gb_df.groupby(["Number", "Address", "Time", "Weekday"]).agg({Common.PREDICTING_FACTOR: "mean", "pred": "mean", "Bike Stands": "max"}).reset_index()
unseen_gb_df[Common.PREDICTING_FACTOR] = unseen_gb_df[Common.PREDICTING_FACTOR].round(0).astype(np.int64)
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
            # actual bikes of its station
            ax_y1 = unseen_gb_df[condition][Common.PREDICTING_FACTOR]
            # predict bikes of its station
            ax_y2 = unseen_gb_df[condition]["pred"]
            # total bike stands of its station
            ax_y3 = unseen_gb_df[condition]["Bike Stands"]
            ax.plot(ax_x, ax_y1, "b-", label=u'Actual')
            ax.plot(ax_x, ax_y2, "r-", label=u'Predicted')
            ax.plot(ax_x, ax_y3, "-.", color = 'black', label=u'Bike Stands')
            #print(ax_x.dtypes)
            ax.fill_between(ax_x.dt.to_pydatetime(), ax_y2 - rmse, ax_y2 + rmse, facecolor='#3a3a3a', alpha=0.5)
            y_min = 0
            y_max = unseen_gb_df["Bike Stands"].max()
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
    fig.text(0.06, 0.5, "Mean Available Stands", ha='center', va='center', rotation='vertical', fontsize="medium")
    # plot the legend
    fig.legend(handles, labels, title="Color", loc='center', bbox_to_anchor=(0.5, 0.06, 0., 0.), ncol=4)
    fig.savefig(f"{Common.PREDICTING_PLOTS_DIR}/unseen/unseen_prediction_{i}.png")
    plt.close()

##############################################################
################## EVALUATE 15 MINUTES PERIOD ################
##############################################################
# change to raw-data directory to fetch JSON files
data_dir = os.path.join(working_dir, "report")
os.chdir(data_dir)
print(f"Change current directory to {data_dir}")
# read all CSV files underneath
data_arr = []
for f in os.listdir():
    try:
        data = pd.read_csv(f, delimiter=",", encoding='latin1')
        data_arr.append(data)
        #print(f"Reading {f}")
    except:
        print(f"{f} goes wrong")
df = pd.concat(data_arr)

# change back to root directory
os.chdir(working_dir)
print(f"Change current directory to {working_dir}")

# re-index dataframe
df = df.reset_index(drop=True)

# remove Unnamed column with Nan values, don't know why it creates an unnamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# drop Time column
df = df.drop("Time", axis=1)

# rename columns
df = df.rename(columns={"Actual Available Bike Stands": "Actual Stands", "Pred Available Bike Stands": "Pred Stands"})

# calculate the number of bikes
df["Bike Stands"] = df.apply(lambda row: setBikeStands(row["Number"]), axis=1)
df["Pred Bikes"] = df.apply(lambda row: calcBikes(row["Bike Stands"], row["Pred Stands"], row["Pred Bikes"]), axis=1)
df["Actual Bikes"] = df.apply(lambda row: calcBikes(row["Bike Stands"], row["Actual Stands"], row["Actual Bikes"]), axis=1)

# drop NA
df = df.dropna()

# convert timestamp to readable date using pandas
df["Actual Time"] = pd.to_datetime(df["Actual Time"], unit='s')

# handle datetime
df["Date"] = df["Actual Time"].dt.strftime(Common.DATE_FORMAT)
df["Time"] = df["Actual Time"].apply(lambda x: "%s:%s:00" % (x.strftime('%H'), Common.refineMinute(x.minute)))

# leave out data from 12:30 am to 05:30 am
df = df[(df["Time"] >= "05:30:00")].reset_index()

# convert time string to time to plot data
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

# convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
df["Number"] = df["Number"].astype(np.int64)
df["Actual Stands"] = df["Actual Stands"].astype(np.int64)
df["Actual Bikes"] = df["Actual Bikes"].astype(np.int64)
df["Pred Stands"] = df["Pred Stands"].astype(np.int64)
df["Pred Bikes"] = df["Pred Bikes"].astype(np.int64)

# evaluate how good the model is
df["Predict Sufficiency"] = df.apply(lambda row: 1 if row["Pred Stands"] <= row["Actual Stands"] else 0, axis=1)
correct_cases = len(df[df["Predict Sufficiency"] == 1])
total_cases = len(df)
correct_ratio = (correct_cases / total_cases) * 100
print(f"Correct {correct_cases}/{total_cases} = {correct_ratio}%")
#print(df[["Number", "Time", "Pred Stands", "Actual Stands", "Predict Sufficiency"]])

# handle data for plotting
df = df[["Number", "Time", "Actual Stands", "Pred Stands", "Bike Stands"]]
df = df.groupby(["Number", "Time"]).agg({"Actual Stands": "mean", "Pred Stands": "mean", "Bike Stands": "max"}).reset_index()
df["Actual Stands"] = df["Actual Stands"].round(0).astype(np.int64)
df["Pred Stands"] = df["Pred Stands"].round(0).astype(np.int64)
for i in range(0, len(Common.EVALUATION_STATIONS)):
    fig, ax = plt.subplots(figsize = (8, 7))
    
    condition = (df["Number"] == Common.EVALUATION_STATIONS[i])
    ax_x = df[condition]["Time"]
    # actual bikes
    ax_y1 = df[condition]["Actual Stands"]
    # predict bikes
    ax_y2 = df[condition]["Pred Stands"]
    # total bike stands of its station
    ax_y3 = df[condition]["Bike Stands"]
    ax.plot(ax_x, ax_y1, "b-", label=u'Actual')
    ax.plot(ax_x, ax_y2, "r-", label=u'Predicted')
    ax.plot(ax_x, ax_y3, "-.", color = 'black', label=u'Bike Stands')            
    y_min = 0
    y_max = df["Bike Stands"].max()
    ax.set_ylim([y_min, y_max])
    # locate sticks every 1 hour
    ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
    # show locate label with hour and minute format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # set smaller size for tick labels
    ax.xaxis.set_tick_params(labelsize=7)
    # set title, xlable and ylabel of the figure
    ax.set(title=f"Evaluate station {Common.EVALUATION_STATIONS[i]} in the period of 15 minutes", xlabel="Hour of day", ylabel="Available Stands")
    # set location of legend
    ax.legend(title="Color", loc='center', bbox_to_anchor=(0.5, -0.2, 0., 0.), ncol=4, borderaxespad=0.)
    # margin x at 0 and y at 0.1
    ax.margins(x=0.0, y=0.1)
    ax.grid(linestyle="-")
    # show rotate tick lables automatically with 90 degree
    fig.autofmt_xdate(rotation = "90")
    fig.savefig(f"{Common.EVALUATION_PLOTS_DIR}/station_{Common.EVALUATION_STATIONS[i]}_15mins.png")
    plt.close()

#print(df)

##############################################################
################## EVALUATE 30 MINUTES PERIOD ################
##############################################################
# change to raw-data directory to fetch JSON files
data_dir = os.path.join(working_dir, "report")
os.chdir(data_dir)
print(f"Change current directory to {data_dir}")
# read all CSV files underneath
data_arr = []
for f in os.listdir():
    try:
        data = pd.read_csv(f, delimiter=",", encoding='latin1')
        data_arr.append(data)
        #print(f"Reading {f}")
    except:
        print(f"{f} goes wrong")
df = pd.concat(data_arr)

# change back to root directory
os.chdir(working_dir)
print(f"Change current directory to {working_dir}")

# re-index dataframe
df = df.reset_index(drop=True)

# remove Unnamed column with Nan values, don't know why it creates an unnamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# drop Time column
df = df.drop("Time", axis=1)

# rename columns
df = df.rename(columns={"Actual Available Bike Stands": "Actual Stands", "Pred Available Bike Stands": "Pred Stands"})

# calculate the number of bikes
df["Bike Stands"] = df.apply(lambda row: setBikeStands(row["Number"]), axis=1)
df["Pred Bikes"] = df.apply(lambda row: calcBikes(row["Bike Stands"], row["Pred Stands"], row["Pred Bikes"]), axis=1)
df["Actual Bikes"] = df.apply(lambda row: calcBikes(row["Bike Stands"], row["Actual Stands"], row["Actual Bikes"]), axis=1)

# drop NA
df = df.dropna()

# convert timestamp to readable date using pandas
df["Actual Time"] = pd.to_datetime(df["Actual Time"], unit='s')

# handle datetime
df["Date"] = df["Actual Time"].dt.strftime(Common.DATE_FORMAT)
df["Time"] = df["Actual Time"].apply(lambda x: "%s:%s:00" % (x.strftime('%H'), Common.refineMinute(x.minute)))

# leave out data from 12:30 am to 05:30 am
df = df[(df["Time"] >= "05:30:00")].reset_index()

# convert time string to time to plot data
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

# convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
df["Number"] = df["Number"].astype(np.int64)
df["Actual Stands"] = df["Actual Stands"].astype(np.int64)
df["Actual Bikes"] = df["Actual Bikes"].astype(np.int64)
df["Pred Stands"] = df["Pred Stands"].astype(np.int64)
df["Pred Bikes"] = df["Pred Bikes"].astype(np.int64)

# evaluate how good the model is
df["Predict Sufficiency"] = df.apply(lambda row: 1 if row["Pred Stands"] <= row["Actual Stands"] else 0, axis=1)
correct_cases = len(df[df["Predict Sufficiency"] == 1])
total_cases = len(df)
correct_ratio = (correct_cases / total_cases) * 100
print(f"Correct {correct_cases}/{total_cases} = {correct_ratio}%")

# handle data for plotting
df = df[["Number", "Time", "Actual Stands", "Pred Stands", "Bike Stands"]]
df = df.groupby(["Number", "Time"]).agg({"Actual Stands": "mean", "Pred Stands": "mean", "Bike Stands": "max"}).reset_index()
df["Actual Stands"] = df["Actual Stands"].round(0).astype(np.int64)
df["Pred Stands"] = df["Pred Stands"].round(0).astype(np.int64)
for i in range(0, len(Common.EVALUATION_STATIONS)):
    fig, ax = plt.subplots(figsize = (8, 7))
    
    condition = (df["Number"] == Common.EVALUATION_STATIONS[i])
    ax_x = df[condition]["Time"]
    # actual bikes
    ax_y1 = df[condition]["Actual Stands"]
    # predict bikes
    ax_y2 = df[condition]["Pred Stands"]
    # total bike stands of its station
    ax_y3 = df[condition]["Bike Stands"]
    ax.plot(ax_x, ax_y1, "b-", label=u'Actual')
    ax.plot(ax_x, ax_y2, "r-", label=u'Predicted')
    ax.plot(ax_x, ax_y3, "-.", color = 'black', label=u'Bike Stands')            
    y_min = 0
    y_max = df["Bike Stands"].max()
    ax.set_ylim([y_min, y_max])
    # locate sticks every 1 hour
    ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
    # show locate label with hour and minute format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # set smaller size for tick labels
    ax.xaxis.set_tick_params(labelsize=7)
    # set title, xlable and ylabel of the figure
    ax.set(title=f"Evaluate station {Common.EVALUATION_STATIONS[i]} in period of 30 minutes", xlabel="Hour of day", ylabel="Available Stands")
    # set location of legend
    ax.legend(title="Color", loc='center', bbox_to_anchor=(0.5, -0.2, 0., 0.), ncol=4, borderaxespad=0.)
    # margin x at 0 and y at 0.1
    ax.margins(x=0.0, y=0.1)
    ax.grid(linestyle="-")
    # show rotate tick lables automatically with 90 degree
    fig.autofmt_xdate(rotation = "90")
    fig.savefig(f"{Common.EVALUATION_PLOTS_DIR}/station_{Common.EVALUATION_STATIONS[i]}_30mins.png")
    plt.close()

#print(df)

end = time.time()
print("Done preparation after {} seconds".format((end - start)))
