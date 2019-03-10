'''
    Author: Tam M Pham
    Created date: 05/02/2019
    Modified date: 05/02/2019
    Description:
        Read raw data from multile CSV files, then pre-processing the data 
        and finally store the data to file to re-use the data for exploring
        and clustering
'''

import os
import numpy as np
import pandas as pd
import time
from common import Common
import sys
import threading
import fnmatch
import re

def count_check_in(diff):
    if (diff > 0):
        return diff
    return 0

def count_check_out(diff):
    if (diff < 0):
        return abs(diff)
    return 0

start = time.time()

#########################################################
##################### READ RAW DATA #####################
#########################################################
# get the current working directory
working_dir = os.getcwd()
# change to raw-data directory to fetch JSON files
data_dir = os.path.join(working_dir, "new-data")
os.chdir(data_dir)
print(f"Change current directory to {data_dir}")

# get number of JSON files underneath a directory
files = fnmatch.filter(os.listdir(data_dir), '*.json')
n_files = len(files)
print(f"Total JSON files is {n_files}")

JSON_FILE_NAME_PATTERN = "([A-Z]{1}[a-z]{2})-([0-2][0-9]|3[0-1])-(0[0-9]|1[0-2])-\d{4}"
file_dictionaries = {}  # 424 items
for f in files:
    matcher = re.search(JSON_FILE_NAME_PATTERN, f)
    if matcher:
        match_str = matcher.group()
        if match_str in file_dictionaries:
            file_array = file_dictionaries[match_str]
            file_array.append(f)
            file_dictionaries[match_str] = file_array
            continue
        file_dictionaries[match_str] = [f]
    else:
        print(f"Not match {f}")
#print(len(list(file_dictionaries.keys())))

for date in file_dictionaries.keys():
    ignore = False
    df = pd.DataFrame()
    date_file = f"{working_dir}/temp/{date}.csv"
    if Common.exists(date_file) == True:
        ignore = True
        continue
    for f in file_dictionaries[date]:
        temp_df = pd.read_json(f"{f}")
        if len(temp_df) <= 0:
            continue
        # drop unwanted columns
        temp_df = temp_df.drop(["available_bikes", "banking", "bonus", "contract_name", "position", "status"], axis=1)
        df = pd.concat([df, temp_df], ignore_index = True, sort=False)
        # drop NA
        df = df.dropna()
        # remove duplicates
        df = df.drop_duplicates(subset=["number", "last_update", "available_bike_stands"], keep='first')
        # re-index dataframe
        df = df.reset_index(drop=True)
    Common.saveCSV(df, date_file)

# change to temp directory which contains combined CSV files of JSON files
json_data_dir = os.path.join(working_dir, "temp")
os.chdir(json_data_dir)
print(f"Change current directory to {json_data_dir}")
# read all JSON files underneath
json_df = pd.concat([pd.read_csv(f, delimiter=",", encoding='latin1') for f in os.listdir()], ignore_index = True, sort=False)
print(f"Total rows from JSON: {len(json_df)}")

# change to raw-data directory to fetch CSV files
data_dir = os.path.join(working_dir, "raw-data")
os.chdir(data_dir)
print(f"Change current directory to {data_dir}")
# read all CSV files underneath
csv_df = pd.concat([pd.read_csv(f, delimiter=",", encoding='latin1') for f in os.listdir()], ignore_index = True, sort=False)
print(f"Total rows from CSV: {len(csv_df)}")    # 6255036

# merge two data together
df = pd.concat([json_df, csv_df], ignore_index = True, sort = False)
# station 1 used to be Chatham Street but is Claredon Row so far
df.loc[df["number"] == 1, "name"] = "CLARENDON ROW"
df.loc[df["number"] == 1, "address"] = "Claredon Row"
print(f"Total rows after merging: {len(df)}")

# change back to root directory
os.chdir(working_dir)
print(f"Change current directory to {working_dir}")

########################################################
##################### DATA HANDLER #####################
########################################################
# remove Unnamed column with Nan values, don't know why it creates an unnamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# drop NA
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
df["time"] = df["last_update"].apply(lambda x: "%s:%s:00" % (x.strftime('%H'), Common.refineMinute(x.minute)))
df["weekday"] = df["last_update"].dt.strftime("%a")

# group by number and sort datetime in order
df.sort_values(by=["number", "date", "time"], inplace=True)

# copy the available stands number of the previous row as last available stands number
df["last_available_stands"] = df.groupby(["number", "name", "address"])["available_bike_stands"].shift(1)
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

###############################################################
############### SAVE PREPROCESSING DATA TO FILE ###############
###############################################################
path = os.path.join(working_dir, Common.CLEAN_DATA_DIR + "/db_new_data.csv")
print(f"Saving data to CSV file to {path}")
Common.saveCSV(df, path)
#print(df)

end = time.time()
print("Done preparation after {} seconds".format((end - start)))
