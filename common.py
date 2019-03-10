'''
    Author: Tam M Pham
    Created date: 05/02/2019
    Modified date: 05/03/2019
    Description:
        Define common constants and methods to call in other files.
'''

import os
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import re
from sklearn.externals import joblib    # for saving and loading model
from sklearn import preprocessing   # label encoder
import requests
import numpy as np

class Common:
    CLEAN_DATA_DIR = "./saved-data"
    #CLEAN_DATA_FILE_FULL_PATH = "./saved-data/db_all_data.csv"
    CLEAN_DATA_FILE_FULL_PATH = "./saved-data/db_new_data.csv"
    CLUSTERED_DATA_FILE_FULL_PATH = "./saved-data/db_clustered_stations.csv"
    PLOTS_DIR = "./plots"
    CLUSTERING_PLOTS_DIR = "./plots/clustering"
    PREDICTING_PLOTS_DIR = "./plots/predicting"
    SHORT_WEEKDAY_ORDER = ['Mon','Tue','Wed','Thu','Fri','Sat', 'Sun']
    CLUSTERING_NUMBER = 4
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:00"
    REPORT_FILE_NAME_FORMAT = "%a-%d-%m-%Y-%H-%M-%S"
    MAX_STATION_NUMBER = 102    # Dublin bikes only has 102 stations while implementing this project
    MAX_AXES_ROW = 3    # plot 3 axes each row
    IMPORTANT_FACTORS = ["Avg Bikes", "Weekday Code", "Time Code", "Prev Bikes", "Cluster", "Latitude", "Longitude", "Season Code", "WindSpeed", "AirTemperature"]

    @staticmethod
    def refineMinute(min):
        if min < 10: return "00"
        elif min < 20: return "10"
        elif min < 30: return "20"
        elif min < 40: return "30"
        elif min < 50: return "40"
        elif min < 60: return "50"
        else: 
            return np.nan

    @staticmethod
    def refineTime(string):
        time = pd.to_datetime(string, format="%H:%M:%S")
        hour = time.strftime('%H')
        minute = "00" if time.minute < 30 else "30"
        return "%s:%s:00" % (hour, minute)

    @staticmethod
    def defineSeason(string):
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

    @staticmethod
    def getDataFrameFromFile(path, notParseDate=False):
        # get the relative path of preparation data file
        rel_path = os.path.relpath(path)
        # read CSV files using Pandas
        if (notParseDate == False):
            df = pd.read_csv(rel_path, delimiter = ",", parse_dates=["Date"])
        else:
            df = pd.read_csv(rel_path, delimiter = ",")
        return df

    @staticmethod
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error while creating folder " + directory)

    @staticmethod
    def exists(name):
        #print(f"Your file/folder name is {name}")
        # if name has an extension, it should be a file (except cofiguration file in this case)
        matcher = re.search("\w+\.\w+$", name)
        if matcher:
            #print("=> This is a file")
            return os.path.isfile(name)
        else:
        # if name doesn't have an extension, it is a folder
            #print("=> This is a directory")
            return os.path.isdir(name)

    @staticmethod
    def getWorkingDirectory():
        return os.getcwd()

    @staticmethod
    def goToSubDirectory(subDirectory):
        new_dir = os.path.join(Common.getWorkingDirectory(), subDirectory)
        os.chdir(new_dir)

    @staticmethod
    def saveCSV(df, filePath):
        df.to_csv(filePath, sep=",", encoding='utf-8', index=False)

    @staticmethod
    def deleteFile(path):
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def convertStringToDateTime(str, format):
        return dt.datetime.strptime(str, format)

    @staticmethod
    def predict(station_number, next_minutes=0):
        station_number = int(station_number)
        #print(f"Station: {station_number} {type(station_number)}, minutes: {next_minutes} {type(next_minutes)}")
        next_hour = round(float(int(next_minutes) / 60), 2)

        # load Gradient Boosting model
        model = joblib.load(Common.CLEAN_DATA_DIR + "/gb_model.csv")

        now = dt.now() + timedelta(hours=next_hour)
        hour = now.strftime("%H")
        minute = Common.refineMinute(int(now.strftime("%M")))
        time = Common.refineTime(f"{hour}:{minute}:00")
        weekday = now.strftime("%a")
        season = Common.defineSeason(now.strftime(Common.DATE_FORMAT))
        #print(type(station_number))

        # get clusters dataframe
        clusters = Common.getDataFrameFromFile(Common.CLUSTERED_DATA_FILE_FULL_PATH, True)

        # get all data dataframe
        all_df = Common.getDataFrameFromFile(Common.CLEAN_DATA_FILE_FULL_PATH, True)

        # get info of passing station number
        filter_df = all_df[(all_df["Number"] == station_number)].copy().reset_index(drop=True)

        # left merge these two dataframes together based on Number, Date and Time
        filter_df = pd.merge(filter_df
                            , clusters[["Number", "Time", "Cluster"]]
                            , on=["Number", "Time"]
                            , how="left")

        # group time into 48 factors
        filter_df["Time"] = filter_df["Time"].apply(lambda x: Common.refineTime(x))
        filter_df["Season"] = filter_df["Date"].apply(lambda x: Common.defineSeason(x))
        filter_df["Avg Bikes"] = filter_df["Bike Stands"] - filter_df["Available Bike Stands"]
        filter_df = filter_df.groupby(["Number", "Name", "Address", "Date", "Time", "Bike Stands", "Weekday", "Season"]).agg({"Avg Bikes": "mean", "Cluster": "first"}).reset_index()
        filter_df["Avg Bikes"] = filter_df["Avg Bikes"].round(0)
        filter_df["Prev Bikes"] = filter_df.groupby(["Number", "Name", "Address", "Date"])["Avg Bikes"].shift(1)
        filter_df["Prev Bikes"] = filter_df.apply(
            lambda row: row["Avg Bikes"] if np.isnan(row["Prev Bikes"]) else row["Prev Bikes"],
            axis=1
        )

        # convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
        filter_df["Avg Bikes"] = filter_df["Avg Bikes"].astype(np.int64)
        filter_df["Prev Bikes"] = filter_df["Prev Bikes"].astype(np.int64)

        # read CSV file containing geographical info
        geo = Common.getDataFrameFromFile("./geo-data/db-geo.csv", True)
        filter_df = pd.merge(filter_df
                            , geo[["Number", "Latitude", "Longitude"]]
                            , on=["Number"]
                            , how="left")


        filter_df["Weekday Code"] = pd.to_datetime(filter_df["Date"], format=Common.DATE_FORMAT).dt.weekday
        # label encoding for weekdays, time and season
        le_season = preprocessing.LabelEncoder()
        filter_df["Season Code"] = le_season.fit_transform(filter_df["Season"])
        le_time = preprocessing.LabelEncoder()
        filter_df["Time Code"] = le_time.fit_transform(filter_df["Time"])

        filter_df = filter_df[(filter_df["Time"] == time) & (filter_df["Weekday"] == weekday) 
                            & (filter_df["Season"] == season)].reset_index(drop=True)
        filter_df = filter_df.groupby(["Number", "Name", "Address", "Weekday Code", "Time Code", "Season Code", "Cluster", "Latitude", "Longitude"]) \
                            .agg({"Bike Stands": "max", "Avg Bikes": "mean", "Prev Bikes": "mean"}).reset_index()
        filter_df["Avg Bikes"] = filter_df["Avg Bikes"].round(0).astype(np.int64)
        filter_df["Prev Bikes"] = filter_df["Prev Bikes"].round(0).astype(np.int64)

        # get latitude, longitude of finding station
        lat = 0
        lng = 0
        response = requests.get("https://api.jcdecaux.com/vls/v1/stations?contract=dublin&apiKey=172c8bcf7be48ceb4ac1aee732500142d2b3651a")
        if response.status_code == 200:
            jcdecaux = response.json()
            for i in range(1, len(jcdecaux)):
                #print(jcdecaux[i]["number"])
                if jcdecaux[i]["number"] == station_number:
                    lat = jcdecaux[i]["position"]["lat"]
                    lng = jcdecaux[i]["position"]["lng"]
                    filter_df["Latitude"] = lat
                    filter_df["Longitude"] = lng
                    filter_df["Prev Bikes"] = jcdecaux[i]["available_bike_stands"]
                    filter_df["Current Bike Stands"] = jcdecaux[i]["bike_stands"]
        else:
            lat = filter_df["Latitude"].values
            lng = filter_df["Longitude"].values

        #print(filter_df)
        #print(filter_df.dtypes)
        #sys.exit()
        # get weather of finding station
        response = requests.get(f"https://api.darksky.net/forecast/d988158d08ac590b03dea633e5abaa60/{lat},{lng}")
        if response.status_code == 200:
            weather = response.json()
            apressure = weather["currently"]["pressure"]
            wspeed = weather["currently"]["windSpeed"]
            atemperature = (weather["currently"]["temperature"] - 32) * (5/9)
            filter_df["AtmosphericPressure"] = apressure
            filter_df["WindSpeed"] = wspeed
            filter_df["AirTemperature"] = atemperature
        else:
            # read CSV file containing weather info
            weather = Common.getDataFrameFromFile("./weather-data/M2_weather.csv", True)
            weather = weather.drop_duplicates(subset=["station_id", "datetime", "AtmosphericPressure", "WindSpeed", "AirTemperature"], keep='first')
            weather["datetime"] = pd.to_datetime(weather["datetime"], format="%m/%d/%Y %H:%M")
            weather["Time"] = weather["datetime"].dt.strftime(Common.TIME_FORMAT)
            weather["AtmosphericPressure"] = weather.groupby(["Time"])["AtmosphericPressure"].mean()
            weather["WindSpeed"] = weather.groupby(["Time"])["WindSpeed"].mean()
            weather["AirTemperature"] = weather.groupby(["Time"])["AirTemperature"].mean()
            filter_df = pd.merge(filter_df
                                , weather[["AtmosphericPressure", "WindSpeed", "AirTemperature"]]
                                , how="left")

        feed_df = filter_df[Common.IMPORTANT_FACTORS].copy()
        feed_df = feed_df.drop("Avg Bikes", axis=1)
        pred = model.predict(feed_df).round(0).astype(np.int64)[0]
        old_bike_stands = filter_df["Bike Stands"].values[0]
        curr_bike_stands = filter_df["Current Bike Stands"].values[0]

        #print(filter_df[["Name", "Bike Stands", "Current Bike Stands"]])
        #print(filter_df.dtypes)

        return pred, old_bike_stands, curr_bike_stands