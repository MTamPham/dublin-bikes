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
import sys

class Common:
    CLEAN_DATA_DIR = "./saved-data"
    #CLEAN_DATA_FILE_FULL_PATH = "./saved-data/db_all_data.csv"
    CLEAN_DATA_FILE_FULL_PATH = "./saved-data/db_new_data.csv"
    CLUSTERED_DATA_FILE_FULL_PATH = "./saved-data/db_clustered_stations.csv"
    GRADIENT_BOOSTING_MODEL_FULL_PATH = "./saved-data/gb_model.csv"
    PLOTS_DIR = "./plots"
    CLUSTERING_PLOTS_DIR = "./plots/clustering"
    PREDICTING_PLOTS_DIR = "./plots/predicting"
    UNSEEN_PREDICTING_PLOTS_DIR = "./plots/predicting/unseen"
    EVALUATION_PLOTS_DIR = "./plots/evaluation"
    SHORT_WEEKDAY_ORDER = ['Mon','Tue','Wed','Thu','Fri','Sat', 'Sun']
    CLUSTERING_NUMBER = 4
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:00"
    REPORT_FILE_NAME_FORMAT = "%a-%d-%m-%Y-%H-%M-%S"
    MAX_STATION_NUMBER = 102    # Dublin bikes only has 102 stations while implementing this project
    MAX_AXES_ROW = 3    # plot 3 axes each row
    CONSIDERING_FACTORS = ["Weekday Code", "Time Code", "Prev Stands", "Cluster", "Latitude", "Longitude", "Season Code", "WindSpeed", "AirTemperature"]
    IMPORTANT_FACTORS = ["Weekday Code", "Time Code", "Prev Stands", "Cluster", "Latitude", "Longitude", "Season Code"]
    PREDICTING_FACTOR = "Avg Stands"
    PREVIOUS_PREDICTING_FACTOR = "Prev Stands"
    EVALUATION_STATIONS = [79, 5, 100, 66, 33]

    @staticmethod
    def refine_minute(min):
        if min < 10: return "00"
        elif min < 20: return "10"
        elif min < 30: return "20"
        elif min < 40: return "30"
        elif min < 50: return "40"
        elif min < 60: return "50"
        else: 
            return np.nan

    @staticmethod
    def refine_time(string):
        time = pd.to_datetime(string, format="%H:%M:%S")
        hour = time.strftime('%H')
        minute = "00" if time.minute < 30 else "30"
        return "%s:%s:00" % (hour, minute)

    @staticmethod
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

    @staticmethod
    def get_dataframe_from_file(path, notParseDate=False):
        # get the relative path of preparation data file
        rel_path = os.path.relpath(path)
        # read CSV files using Pandas
        if (notParseDate == False):
            df = pd.read_csv(rel_path, delimiter = ",", parse_dates=["Date"])
        else:
            df = pd.read_csv(rel_path, delimiter = ",")
        return df

    @staticmethod
    def create_folder(directory):
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
    def get_working_directory():
        return os.getcwd()

    @staticmethod
    def go_to_sub_directory(subDirectory):
        new_dir = os.path.join(Common.get_working_directory(), subDirectory)
        os.chdir(new_dir)

    @staticmethod
    def save_csv(df, filePath):
        df.to_csv(filePath, sep=",", encoding='utf-8', index=False)

    @staticmethod
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def convertStringToDateTime(str, format):
        return dt.datetime.strptime(str, format)

    @staticmethod
    def predict(station_number, next_minutes=0):
        #print(f"Executing predict() with {station_number}({type(station_number)}) and {next_minutes}({type(next_minutes)})")
        station_number = int(station_number)
        next_hour = round(float(int(next_minutes) / 60), 2)

        # load Gradient Boosting model
        model = joblib.load(Common.GRADIENT_BOOSTING_MODEL_FULL_PATH)

        now = dt.now() + timedelta(hours=next_hour)
        hour = now.strftime("%H")
        minute = Common.refine_minute(int(now.strftime("%M")))
        time = Common.refine_time(f"{hour}:{minute}:00")
        weekday = now.strftime("%a")
        season = Common.define_season(now.strftime(Common.DATE_FORMAT))

        # get clusters dataframe
        clusters = Common.get_dataframe_from_file(Common.CLUSTERED_DATA_FILE_FULL_PATH, True)

        # get all data dataframe
        all_df = Common.get_dataframe_from_file(Common.CLEAN_DATA_FILE_FULL_PATH, True)

        # get info of passing station number
        filter_df = all_df[(all_df["Number"] == station_number)].copy().reset_index(drop=True)
        index = len(filter_df) - 1        
        saved_last_update = filter_df.loc[index, "Date"] + " " + filter_df.loc[index, "Time"]

        if (next_minutes == 0):
            bike_stands, available_stands, available_bikes, lat, lng, last_update = Common.retrieve_jcdecaux(station_number)
            return available_stands, filter_df["Bike Stands"].values[0], bike_stands, last_update

        # left merge these two dataframes together based on Number, Date and Time
        filter_df = pd.merge(filter_df
                            , clusters[["Number", "Time", "Cluster"]]
                            , on=["Number", "Time"]
                            , how="left")

        # group time into 48 factors
        filter_df["Time"] = filter_df["Time"].apply(lambda x: Common.refine_time(x))
        filter_df["Season"] = filter_df["Date"].apply(lambda x: Common.define_season(x))
        filter_df[Common.PREDICTING_FACTOR] = filter_df["Available Stands"]
        filter_df = filter_df.groupby(["Number", "Name", "Address", "Date", "Time", "Bike Stands", "Weekday", "Season"]).agg({Common.PREDICTING_FACTOR: "mean", "Cluster": "first"}).reset_index()
        filter_df[Common.PREDICTING_FACTOR] = filter_df[Common.PREDICTING_FACTOR].round(0)
        filter_df[Common.PREVIOUS_PREDICTING_FACTOR] = filter_df.groupby(["Number", "Name", "Address", "Date"])[Common.PREDICTING_FACTOR].shift(1)
        filter_df[Common.PREVIOUS_PREDICTING_FACTOR] = filter_df.apply(
            lambda row: row[Common.PREDICTING_FACTOR] if np.isnan(row[Common.PREVIOUS_PREDICTING_FACTOR]) else row[Common.PREVIOUS_PREDICTING_FACTOR],
            axis=1
        )

        # convert float64 columns to int64 columns, don't know why it converts numeric columns to float64
        filter_df[Common.PREDICTING_FACTOR] = filter_df[Common.PREDICTING_FACTOR].astype(np.int64)
        filter_df[Common.PREVIOUS_PREDICTING_FACTOR] = filter_df[Common.PREVIOUS_PREDICTING_FACTOR].astype(np.int64)

        # read CSV file containing geographical info
        geo = Common.get_dataframe_from_file("./geo-data/db-geo.csv", True)
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
                            .agg({"Bike Stands": "max", Common.PREDICTING_FACTOR: "mean", Common.PREVIOUS_PREDICTING_FACTOR: "mean"}).reset_index()
        filter_df[Common.PREDICTING_FACTOR] = filter_df[Common.PREDICTING_FACTOR].round(0).astype(np.int64)
        filter_df[Common.PREVIOUS_PREDICTING_FACTOR] = filter_df[Common.PREVIOUS_PREDICTING_FACTOR].round(0).astype(np.int64)

        #print(filter_df)
        #print(filter_df.dtypes)
        
        bike_stands, available_stands, available_bikes, lat, lng, last_update = Common.retrieve_jcdecaux(station_number)
        if (bike_stands == 0 and available_stands == 0) :
            filter_df["last_update"] = saved_last_update
            filter_df["Current Bike Stands"] = filter_df["Bike Stands"]
        else:
            filter_df["Latitude"] = lat
            filter_df["Longitude"] = lng
            #filter_df["Prev Bikes"] = available_bikes
            filter_df[Common.PREVIOUS_PREDICTING_FACTOR] = available_stands
            filter_df["Current Bike Stands"] = bike_stands
            filter_df["last_update"] = pd.to_datetime(last_update, unit='ms', utc=True)

        pred = model.predict(filter_df[Common.IMPORTANT_FACTORS]).round(0).astype(np.int64)[0]
        old_bike_stands = filter_df["Bike Stands"].values[0]
        curr_bike_stands = filter_df["Current Bike Stands"].values[0]
        last_update = filter_df["last_update"].values[0]

        #print(filter_df)
        #print(filter_df.dtypes)

        return pred, old_bike_stands, curr_bike_stands, last_update

    @staticmethod
    def retrieve_jcdecaux(station_number):
        print("Calling JCDecaux API")
        response = requests.get(f"https://api.jcdecaux.com/vls/v1/stations/{station_number}?contract=dublin&apiKey=172c8bcf7be48ceb4ac1aee732500142d2b3651a")
        if response.status_code == 200:
            station = response.json()
            return station["bike_stands"], station["available_bike_stands"], station["available_bikes"], \
                    station["position"]["lat"], station["position"]["lng"], station["last_update"]
        else:
            return 0, 0, 0, 0, 0, 0
