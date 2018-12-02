'''
    Author: Tam M Pham
    Created date: 22/11/2018
    Modified date: 02/12/2018
    Description:
        Plotting distribution of activity throughout the week
        Finding the most 10th busy and least 10th busy stations
'''

import os
import numpy as np
import pandas as pd
import calendar
import time
import constant

start = time.time()

print("Please be patient, it might take a while...")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error while creating folder " + directory)

# get the relative path of preparation data file
rel_path = "." + constant.CLEAN_DATA_DIR + "/" + constant.CLEAN_DATA_FILE
rel_path = os.path.relpath(rel_path)

# read CSV files using Pandas
df = pd.read_csv(rel_path, delimiter = ",", parse_dates=["date"])

# calculate total activity of each timestamp of a station
df["total_activity"] = df.apply(lambda row: row["check_in"] + row["check_out"], axis = 1)
df = df[["number", "address", "date", "weekday", "total_activity"]]
df = df.groupby(["address", "weekday"])["total_activity"].sum()

# after calculating the sum of total activiy, the current index is address; we need to reset the index to start from 0
df = df.reset_index()

# make weekday into columns, 
df_weekday = df.pivot(index = "address", columns = "weekday", values = "total_activity")
# fill Nan value by 0
df_weekday = df_weekday.fillna(0)
# order weekday, it should be ordered as "Mon","Tue","Wed","Thu","Fri","Sat", "Sun"
days = ["Mon","Tue","Wed","Thu","Fri","Sat", "Sun"]
df_weekday = df_weekday.reindex(columns=days)
#print(df_weekday)

# box plot distribution of activity throughout the week
boxplot = df_weekday.boxplot(column=days)
figure = boxplot.get_figure()
createFolder("." + constant.PLOTS_DIR)
plot_path = "." + constant.PLOTS_DIR + "/usage_through_week.png"
figure.savefig(plot_path)

# find the most busy station (which has the highest total activity)
df_busy_stations = df.copy()
df_busy_stations = df_busy_stations.groupby("address").agg({"total_activity": "sum"})
print("The most 10th busy stations are: ")
print(df_busy_stations.head(10))
print("The least 10th busy stations are: ")
print(df_busy_stations.tail(10))

end = time.time()
print("Done exploration after {} seconds".format((end - start)))