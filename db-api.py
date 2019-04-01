'''
    Author: Tam M Pham
    Created date: 05/03/2019
    Modified date: 05/03/2019
    Description:
        Create a web service api to query the available bike stands using Gradient Boosting model
'''

from flask import Flask, request
from flask import jsonify
from common import Common
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/search', methods=['GET'])
def search():
    if ("station" in request.args == False):
        return jsonify({"status": 500})
    else:
        station = request.args.get("station")
        if (isinstance(station, int) == False):
            station = int(station)

    if ("minutes" in request.args):
        minutes = request.args.get("minutes")
        if (isinstance(minutes, int) == False):
            minutes = int(minutes)
    else:
        minutes = 0

    print(f"Station: {station} {type(station)}, minutes: {minutes} {type(minutes)}")
    status_code = 404
    data = {}
    try:
        if (minutes > 0):
            print("Calling predict() with station and minutes")
            pred, old_bike_stands, curr_bike_stands, last_update = Common.predict(station, minutes)
        else:
            print("Calling predict() with station only")
            pred, old_bike_stands, curr_bike_stands, last_update = Common.predict(station)
        data = {
            "available_bikes": int(curr_bike_stands) - int(pred),\
            "available_bike_stands": int(pred),\
            "old_bike_stands": int(old_bike_stands), \
            "curr_bike_stands": int(curr_bike_stands), \
            "last_update": str(last_update)
        }
        status_code = 200
    except Exception as e:
        status_code = 500
        data = { "error" : e }
    result = {"status": status_code, "data": data}
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port='4502')