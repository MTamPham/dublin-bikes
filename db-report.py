'''
    Author: Tam M Pham
    Created date: 05/03/2019
    Modified date: 05/03/2019
    Description:
        Create a client side to collect our predict to compare to the real data in JCDecaux
'''

import sched
import time
import requests
from common import Common
import sys

TIME_DELAY = 15 # minutes
STATIONS = [79, 5, 100, 66, 33] # north, south, west, east, middle
FILE_NAME_FORMAT = "./report/{}.csv"

scheduler = sched.scheduler(time.time, time.sleep)
time_arr = []

def func():
    now = time.time()
    print('EVENT: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))))    
    # collect current bike stands as the result of last 30 mins
    collect()
    # predict available bike stands in next 30 mins
    predict()

def collect():
    if (len(time_arr) <= 1):
        return
    last_time = time_arr[len(time_arr) - 1]
    now = round(time.time())
    for i in range(0, len(STATIONS)):
        station = STATIONS[i]
        response = requests.get(f"https://api.jcdecaux.com/vls/v1/stations/{station}?contract=dublin&apiKey=172c8bcf7be48ceb4ac1aee732500142d2b3651a")
        json_data = response.json()
        available_bike_stands = json_data["available_bike_stands"]
        available_bikes = json_data["available_bikes"]
        new_content = f"{str(now)},{str(available_bike_stands)},{str(available_bikes)}"
        replaceLine(FILE_NAME_FORMAT.format(last_time), i + 1, new_content)

def predict():
    now = round(time.time())
    now_str = time.strftime(Common.REPORT_FILE_NAME_FORMAT, time.localtime(now))
    Common.create_folder("report")
    with open(FILE_NAME_FORMAT.format(now_str), "w") as f:
        text = "Number,Time,Pred Available Bike Stands,Pred Bikes,Actual Time,Actual Available Bike Stands,Actual Bikes\n"
        
        for station in STATIONS:
            text += str(station) + "," + str(now) + ","
            url = f"http://localhost:4502/api/search?station={station}&minutes={TIME_DELAY}"
            response = requests.get(url)
            json_data = response.json()
            status = int(json_data["status"])
            print(f"Getting URL {url} responses {status} => {json_data}")

            if (status == 200):
                available_bike_stands = json_data["data"]["available_bike_stands"]
                text += str(available_bike_stands) + ","
                available_bikes = json_data["data"]["available_bikes"]
                text += str(available_bikes) + "\n"
                time_arr.append(now_str)
            else:
                print(f"An error occurs while calling the API to predict - {json_data['status']}")
                text += "\n"
            #time.sleep(20)

        f.write(text)

def replaceLine(path, line_no, content):
    print(f"Replacing line {line_no} of {path} to {content}")
    with open(path, "r") as f:
        lines = f.readlines()
    #if (lines[line_no].__contains__(str(station)) == True):
    lines[line_no] = lines[line_no].replace("\n", "") + "," + content + "\n"
    with open(path, "w") as f:
        f.writelines(lines)

start = time.time()
stop = start + 7 * 24 * 60 * 60     # start for 7 days
print("START: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)))
func()
while True:
    now = time.time()
    # stop the caller after 7 days
    if now >= stop:
        break
    scheduler.enter(TIME_DELAY * 60, 1, func, ())
    scheduler.run()