'''
    Author: Tam M Pham
    Created date: 22/11/2018
    Modified date: 25/11/2018
    Description:
        Prepare data for analyzing
'''

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import constant

def count_check_in(last_available_stands, current_available_stands):
    diff = current_available_stands - last_available_stands 
    if (diff > 0):
        return diff
    return 0

def count_check_out(last_available_stands, current_available_stands):
    diff = current_available_stands - last_available_stands 
    if (diff < 0):
        return abs(diff)
    return 0

def refine_min(min):
    if min < 10: return "00"
    elif min < 20: return "10"
    elif min < 30: return "20"
    elif min < 40: return "30"
    elif min < 50: return "40"
    elif min < 60: return "50"
    else: 
        return np.nan

def shorten_weekday(weekday_name):
    return weekday_name[0:3]

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error while creating folder " + directory)

# test: read a single file
# get the relative path of file
#rel_path = os.path.relpath("./raw-data/export1.csv")
# read CSV files using Pandas
#df = pd.read_csv(rel_path, delimiter = ",")

# read multiple files
# remember the current directory
# in our case, this file is a directly child of root directory
# so, we can make the most of its nature
#root_dir = os.path.abspath(os.path.dirname("__file__"))
#print(root_dir)
root_dir = "/Users/tampm/MaynoothUniversity/CS440-SCIAFinalThesis/dublin-bikes"
# make sure that the root directory is the project directory
os.chdir(root_dir)
# get absolute path of raw-data folder
data_dir = os.path.abspath("raw-data")
# change to raw-data directory to fetch CSV files
os.chdir(data_dir)
print("Change to data directory -> {0}".format(os.getcwd()))
# read all CSV files underneath
df = pd.concat([pd.read_csv(f, delimiter=",", encoding='latin1') for f in os.listdir()], ignore_index = True, sort=False)
# change back to root directory
os.chdir(root_dir)
print("Change back to root directory -> {0}".format(os.getcwd()))

# remove data before 1st October 2016
#df = df[(df["last_update"] >= 1475276400000)]

# TODO: find out how to remove duplicated data efficiently
df = df.drop_duplicates()
#cols = ["address","last_update","available_bike_stands"]
#df = df.loc[(df[cols].shift() != df[cols]).any(axis=1)]

# remove incomplete data
df = df.dropna()

# convert timestamp to readable date using pandas
# reference https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.html
#df["last_update"] = df.apply(lambda row: datetime.utcfromtimestamp(row["last_update"]/1000), axis=1)
df["last_update"] = pd.to_datetime(df["last_update"],unit='ms', utc=True)
# add year, month, day, hour, min, sec, date, time column to dataframe
df["year"] = df["last_update"].dt.year
df["month"] = df["last_update"].dt.month
df["day"] = df["last_update"].dt.day
df["hour"] = df["last_update"].dt.hour
df["min"] = df["last_update"].dt.minute
df["min"] = df.apply(lambda row: refine_min(row["min"]), axis=1)
df["sec"] = "00"
df["date"] = df["last_update"].dt.date
df["time"] = df["last_update"].dt.time
# convert hour from int64 to string to join with minute and second
# because join() only accepts string
df["hour"] = df["hour"].astype("str").str.zfill(2)
# join hour, min, sec together
df["time"] = df[["hour", "min", "sec"]].apply(lambda row: ':'.join(row), axis=1)
# add weekday column with short name
df["weekday"] = df["last_update"].dt.day_name()
df["weekday"] = df.apply(lambda row: shorten_weekday(row["weekday"]), axis=1)

# group by number and sort datetime in order
df.groupby(["number"])
df.sort_values(by=["year","month","day","hour","min","sec"])

# copy the available stands number of the previous row as the last available stands number
df["last_available_stands"] = df["available_bike_stands"].shift(1)

# calculate the check in and check out activities and then insert them into the dataframe
df["check_in"] = df.apply(lambda row: count_check_in(row["last_available_stands"], row["available_bike_stands"]), axis=1)
df["check_out"] = df.apply(lambda row: count_check_out(row["last_available_stands"], row["available_bike_stands"]), axis=1)

# apply different aggregations per column by grouping by number, name, address, date, time, weekday
df = df.groupby(["number", "name", "address", "date", "time", "weekday"]).agg({"bike_stands": "min", "available_bike_stands": "last", "check_in": "sum", "check_out": "sum"}).reset_index()

# make the saved data preparation directory
createFolder("." + constant.CLEAN_DATA_DIR)
# delete db_all_data.csv file if it exists there
cleanDataFilePath = "." + constant.CLEAN_DATA_DIR + "/" + constant.CLEAN_DATA_FILE
if os.path.exists(cleanDataFilePath):
    os.remove(cleanDataFilePath)
# save the data preparation for using later
df.to_csv("./saved-data/db_all_data.csv", sep=",", encoding='utf-8', index=False)

# print result out
#print(df)
print(df[["address","date","time","bike_stands","available_bike_stands","check_in","check_out"]])