import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

/*
  Generated class for the CommonProvider provider.

  See https://angular.io/guide/dependency-injection for more info on providers
  and Angular DI.
*/
@Injectable()
export class CommonProvider {

  constructor(public http: HttpClient) {
    console.log('Hello CommonProvider Provider');
  }

  getPeriodList() {
      var array = new Array();
      for (var i = 0; i <= 60; i+=15) {
          if (i == 0) {
              array.push({
                  value: 0,
                  name: "Now"
              });
          } else {
              array.push({
                  value: i,
                  name: "Next " + i + " minutes"
              });
          }
      }
      return array;
  }

    equals = function( x, y ) {
        if ( x === y ) return true;
        // if both x and y are null or undefined and exactly the same

        if ( ! ( x instanceof Object ) || ! ( y instanceof Object ) ) return false;
        // if they are not strictly equal, they both need to be Objects

        if ( x.constructor !== y.constructor ) return false;
        // they must have the exact same prototype chain, the closest we can do is
        // test there constructor.

        for ( var p in x ) {
            if ( ! x.hasOwnProperty( p ) ) continue;
            // other properties were tested using x.constructor === y.constructor

           if ( ! y.hasOwnProperty( p ) ) return false;
           // allows to compare x[ p ] and y[ p ] when set to undefined

           if ( x[ p ] === y[ p ] ) continue;
           // if they have the same strict value or identity then they are equal

           if ( typeof( x[ p ] ) !== "object" ) return false;
           // Numbers, Strings, Functions, Booleans must be strictly equal

           //if ( ! Object.equals( x[ p ],  y[ p ] ) ) return false;
           // Objects and Arrays must be tested recursively
       }

        for ( p in y ) {
          if ( y.hasOwnProperty( p ) && ! x.hasOwnProperty( p ) ) return false;
          // allows x[ p ] to be set to undefined
        }
        return true;
    }
}
