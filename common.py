import os
import pandas as pd
import datetime as dt

class Common:
    CLEAN_DATA_DIR = "./saved-data"
    CLEAN_DATA_FILE_FULL_PATH = "./saved-data/db_all_data.csv"
    #CLEAN_DATA_FILE_FULL_PATH = "./saved-data/db_new_data.csv"
    PLOTS_DIR = "./plots"
    CLUSTERING_PLOTS_DIR = "./plots/clustering"
    SHORT_WEEKDAY_ORDER = ['Mon','Tue','Wed','Thu','Fri','Sat', 'Sun']

    @staticmethod
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error while creating folder " + directory)

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
