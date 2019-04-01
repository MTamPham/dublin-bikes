'''
    Author: Tam M Pham
    Created date: 22/11/2018
    Modified date: 03/01/2019
    Description:
        Plotting distribution of activity throughout the week
        Finding the most 10th busy and least 10th busy stations
'''

import os
import numpy as np
import pandas as pd
import calendar
import time
from common import Common
import matplotlib.pyplot as plt

start = time.time()

Common.create_folder(Common.PLOTS_DIR)

# get the relative path of preparation data file
rel_path = os.path.relpath(Common.CLEAN_DATA_FILE_FULL_PATH)

# read CSV files using Pandas
df = pd.read_csv(rel_path, delimiter = ",", parse_dates=["Date"])

# see how many occurrence of data for date, the date which has minor values (<10) means the data is somehow missing
#print(df.groupby([df["Date"].dt.date])["Date"].count())

# after viewing, notice that July 2016 has minor values
#print(df[df["Date"].dt.month == 7].reset_index(drop=True))

top_check_ins = pd.DataFrame(df.groupby(df["Address"])["Check In"].sum().sort_values(ascending=False).head(10))
top_check_ins = pd.merge(top_check_ins, df, on="Address")
#print("Top 10 check in stations:")
#print(top_check_ins)

top_check_outs = pd.DataFrame(df.groupby(df["Address"])["Check Out"].sum().sort_values(ascending=False).head(10))
top_check_outs = pd.merge(top_check_outs, df, on="Address")
#print("Top 10 check out stations:")
#print(top_check_outs)

total_activity = df.copy()
total_activity["Total Activity"] = total_activity["Check In"] + total_activity["Check Out"]
total_activity = total_activity.groupby(total_activity["Address"])["Total Activity"].sum()

top_activity = total_activity.copy().sort_values(ascending=False).head(10)
print("Top 10 busiest stations:")
print(top_activity)

bot_activity = total_activity.copy().sort_values().head(10)
print("Top 10 quiest stations:")
print(bot_activity)

##############################################################
################# FIND AVERAGE USAGE PER DAY #################
##############################################################
avg_ci_usage_day = df.copy()
avg_ci_usage_day = avg_ci_usage_day.groupby(["Number", "Name", "Weekday"])["Check In"].mean()
avg_ci_usage_day = avg_ci_usage_day.unstack()
avg_ci_usage_day.boxplot(column=Common.SHORT_WEEKDAY_ORDER)
plt.title("")   
plt.suptitle("")    # get rid of the default title of box plotting
plt.ylabel("avg_cin")
plt.savefig(Common.PLOTS_DIR + "/avg_ci_usage_day.png")
plt.gcf().clear()

avg_co_usage_day = df.copy()
avg_co_usage_day = avg_co_usage_day.groupby(["Number", "Name", "Weekday"])["Check Out"].mean()
avg_co_usage_day = avg_co_usage_day.unstack()
avg_co_usage_day.boxplot(column=Common.SHORT_WEEKDAY_ORDER)
plt.title("")   
plt.suptitle("")    # get rid of the default title of box plotting
plt.ylabel("avg_cout")
plt.savefig(Common.PLOTS_DIR + "/avg_co_usage_day.png")
plt.gcf().clear()

##############################################################
##################### FIND USAGE PER DAY #####################
##############################################################
usage_day = df.copy()
usage_day["Total Activity"] = usage_day["Check In"] + usage_day["Check Out"]
usage_day = usage_day.groupby(["Number", "Name", "Weekday"])["Total Activity"].sum()
usage_day = usage_day.unstack()
usage_day.boxplot(column=Common.SHORT_WEEKDAY_ORDER)
plt.title("Distribution of activity throughout the week")   
plt.suptitle("")    # get rid of the default title of box plotting
plt.ylabel("Activity")
plt.savefig(Common.PLOTS_DIR + "/usage_day.png")
plt.gcf().clear()

end = time.time()
print("Done exploration after {} seconds".format((end - start)))