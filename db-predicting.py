import sys
from common import Common

station = 0
minutes = 0
# only call file without any inputs
if len(sys.argv) <= 1:
    print("The station number is required")
    sys.exit()
else:
    station = int(sys.argv[1])
    if len(sys.argv) == 3:
        minutes = int(sys.argv[2])

pred, old_bike_stands, curr_bike_stands = Common.predict(station, minutes)
print(f"{pred}, {old_bike_stands}, {curr_bike_stands}")
