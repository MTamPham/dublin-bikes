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
import time
import constant

def count_check_in(diff):
    if (diff > 0):
        return diff
    return 0

def count_check_out(diff):
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

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error while creating folder " + directory)

start = time.time()

# read a single file
#rel_path = os.path.relpath("./raw-data/export1.csv")
#df = pd.read_csv(rel_path, delimiter = ",")

# read multiple files
root_dir = "/Users/tampm/MaynoothUniversity/CS440-SCIAFinalThesis/dublin-bikes"
os.chdir(root_dir)
# change to raw-data directory to fetch CSV files
data_dir = os.path.abspath("raw-data")
os.chdir(data_dir)
print("Change to data directory -> {0}".format(os.getcwd()))
# read all CSV files underneath
df = pd.concat([pd.read_csv(f, delimiter=",", encoding='latin1') for f in os.listdir()], ignore_index = True, sort=False)

# change back to root directory
os.chdir(root_dir)
print("Change back to root directory -> {0}".format(os.getcwd()))

# remove Unnamed column with Nan values, don't know why it creates an unnamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# remove incomplete data
df = df.dropna()

# remove duplicates
df = df.drop_duplicates(subset=["number", "last_update", "available_bike_stands"], keep='first')

# re-index dataframe
df = df.reset_index(drop=True)

# convert timestamp to readable date using pandas
# reference https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.html
#df["last_update"] = df.apply(lambda row: datetime.utcfromtimestamp(row["last_update"]/1000), axis=1)
df["last_update"] = pd.to_datetime(df["last_update"], unit='ms', utc=True)

# handle datetime and weekday in dataframe
df["date"] = df["last_update"].dt.strftime('%Y-%m-%d')
df["time"] = df["last_update"].apply(lambda x: "%s:%s:00" % (x.strftime('%H'), refine_min(x.minute)))
df["weekday"] = df["last_update"].dt.strftime("%a")

# group by number and sort datetime in order
df.sort_values(by=["number", "date", "time"])

# copy the available stands number of the previous row as last available stands number
df["last_available_stands"] = df.groupby(["number", "name", "address", "date"])["available_bike_stands"].shift(1)
# if last available stands is NaN, use the available bike stands number of that time
df["last_available_stands"] = df.apply(
    lambda row: row["available_bike_stands"] if np.isnan(row["last_available_stands"]) else row["last_available_stands"],
    axis=1
)

# convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
df["number"] = df.number.astype(np.int64)
df["bike_stands"] = df.bike_stands.astype(np.int64)
df["available_bike_stands"] = df.available_bike_stands.astype(np.int64)
df["last_available_stands"] = df.last_available_stands.astype(np.int64)

# calculate diff
df["diff"] = df.available_bike_stands - df.last_available_stands

# calculate the check in and check out activities and then insert them into the dataframe
df["check_in"] = df["diff"].apply(lambda x: count_check_in(x))
df["check_out"] = df["diff"].apply(lambda x: count_check_out(x))

# apply different aggregations per column by grouping by number, name, address, date, time, weekday
df = df.groupby(["number", "name", "address", "date", "time", "weekday"]).agg({"bike_stands": "min", 
        "diff": "sum", "available_bike_stands": "last", "check_in": "sum", "check_out": "sum"}).reset_index()

# rename columns
df = df.rename(columns={"number": "Number", "name": "Name", "address": "Address", "date": "Date", "time": "Time", "weekday": "Weekday", "bike_stands": "Bike Stands", "diff": "Diff", "available_bike_stands": "Available Bike Stands", "check_in": "Check In", "check_out": "Check Out"})

print("Saving data to CSV file")
# make the saved data preparation directory
createFolder("." + constant.CLEAN_DATA_DIR)
# delete db_all_data.csv file if it exists there
cleanDataFilePath = "." + constant.CLEAN_DATA_DIR + "/" + constant.CLEAN_DATA_FILE
if os.path.exists(cleanDataFilePath):
    os.remove(cleanDataFilePath)
# save the data preparation for using later
df.to_csv(cleanDataFilePath, sep=",", encoding='utf-8', index=False)

# print result out
#print(df)
#print(df[["address","date","time","bike_stands","available_bike_stands","check_in","check_out"]])

end = time.time()
print("Done preparation after {} seconds".format((end - start)))
