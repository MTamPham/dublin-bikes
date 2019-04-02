# dublin-bikes-forecast

This is a web-based mobile application which make use of Gradient Boosting algorithm to predict the available stands at a station at a specific period of time.

### Data source
The station detail such as number, name, address, latitude, longitude are from Firebase Realtime Database (data structure can be found in ./dublinbikes-forecast-export.json)

### Run the application
**It is required to have NPM and Ionic CLi before running the application**

> $ npm install

> $ ionic serve

**Note:
1. dublin-bikes db-api must be started before retrieving result of any station.
2. The Firebase config should be changed in ./src/app/app.module.ts
**