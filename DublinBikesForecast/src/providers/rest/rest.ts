import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

/*
  Generated class for the RestProvider provider.

  See https://angular.io/guide/dependency-injection for more info on providers
  and Angular DI.
*/
@Injectable()
export class RestProvider {

    OUR_REST_API_URL = "http://localhost:4502/api/search";

    constructor(public http: HttpClient) {
        console.log('Hello RestProvider Provider');
    }

    getDataFromAPIViaPromise(stationNumber: number, timePeriod: number) {
        return new Promise(resolve => {
            let apiURL = "";

            if (timePeriod == 0) {
                apiURL = `${this.OUR_REST_API_URL}?station=${stationNumber}`;
            } else {
                apiURL = `${this.OUR_REST_API_URL}?station=${stationNumber}&minutes=${timePeriod}`;    
            }
            
            console.log(apiURL);
            this.http.get(apiURL).subscribe(data => {
                resolve(data);
            }, err => {
                console.log(err);
            });
        });
    }

}
