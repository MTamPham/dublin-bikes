import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { RestProvider } from '../../providers/rest/rest';
import { Station } from '../../data-model/Station';
import { AngularFireDatabase } from '@angular/fire/database';
import { map } from 'rxjs/operators';

/*
  Generated class for the StationProvider provider.

  See https://angular.io/guide/dependency-injection for more info on providers
  and Angular DI.
*/
@Injectable()
export class StationProvider {

    // define list manually
    private stationList: Station[] = new Array();
    // define list from Firebase
    private stationListRef = this.db.list<Station>('stations', ref => ref.orderByChild('number'));

    constructor(public http: HttpClient, public restProvider: RestProvider, private db: AngularFireDatabase) {
        // station 1, 20 are not found in JCDecaux
        // new Station(1, "Clarendon Row", "Clarendon"),
        // new Station(20, "", "James Street East", "James Street East"),
        this.stationList.push(
                            new Station(2, "Blessington Street", "Blessington Street"),
                            new Station(3, "Bolton Street", "Bolton Street"),
                            new Station(4, "Greek Street", "Greek Street"),
                            new Station(5, "Charlemont Place", "Charlemont Place")
                            );
    }

    getAllStations() {
        return this.stationList;
    }

    getAllStationsOnFirebase() {
        return this.parseFirebaseRef(this.stationListRef);
    }

    searchByNumber(number: number) {
        return this.stationList.filter(function(obj, index, stationList) {
            if (obj.number == number) {
              return obj;    
            }
        });
    }

    searchByNumberOnFirebase(number: number) {
        var ref = this.db.list<Station>('stations', ref => ref.orderByChild('number').equalTo(number));
        return this.parseFirebaseRef(ref);
    }

    searchByAddress(address: string) {
        return this.stationList.filter(function(obj, index, stationList) {
            if (obj.address.toLowerCase().indexOf(address) > -1) {
              return obj;
            }
        }); 
    }

    searchByAddressOnFirebase(address: string) {
        var ref = this.db.list<Station>('stations', ref => ref.orderByChild('address').equalTo(address));
        return this.parseFirebaseRef(ref);
    }

    searchByGeoLocationOnFirebase(lat: number, lng: number) {
        var ref = this.db.list<Station>('stations', ref => ref.orderByChild('position').equalTo(lat + "," + lng));
        return this.parseFirebaseRef(ref);
    }

    async predict(stationNumber: number, timePeriod: number) {
        if (typeof timePeriod !== "string" && Number.isNaN(Number(timePeriod))) {
            timePeriod = 0;
        } else {
            timePeriod = Number(timePeriod);
        }

        var predictedStation: Station;
        await this.searchByNumberOnFirebase(stationNumber).subscribe((data) => {
            predictedStation = Object.assign(data[0]);
        });

        return this.restProvider.getDataFromAPIViaPromise(stationNumber, timePeriod).then(responseData => {
            //console.log(responseData);
            if (responseData != null && responseData.hasOwnProperty("status")) {
                if (responseData["status"] == 200) {
                    predictedStation.bikeStands = responseData["data"]["curr_bike_stands"];
                    predictedStation.availableStands = responseData["data"]["available_bike_stands"];
                    predictedStation.availableBikes = responseData["data"]["available_bikes"];
                }
                //console.log(predictedStation);
            }
            return predictedStation;
        });
    }

    parseFirebaseRef(ref) {
        return ref.snapshotChanges().pipe(map(changes => {
              return changes.map(c => ({
                key: c.payload.key, ...c.payload.val()
              }))
        }));        
    }

}
