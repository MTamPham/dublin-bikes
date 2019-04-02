import { Component } from '@angular/core';
import { IonicPage, NavController, NavParams } from 'ionic-angular';
import { StationProvider } from '../../providers/station/station';
import { DetailPage } from '../detail/detail';
import { GeoDetailPage } from '../geo-detail/geo-detail';

/**
 * Generated class for the HomePage page.
 *
 * See https://ionicframework.com/docs/components/#navigation for more info on
 * Ionic pages and navigation.
 */

@IonicPage()
@Component({
  selector: 'page-home',
  templateUrl: 'home.html',
})
export class HomePage {

    constructor(public navCtrl: NavController, public navParams: NavParams, public stationProvider: StationProvider) {
        
    }

    ionViewDidLoad() {
        console.log('ionViewDidLoad HomePage');
    }

    goToDetailPage(type: number) {
        this.navCtrl.setRoot(DetailPage, {type: type});
    }

    goToGeoDetailPage() {
        this.navCtrl.setRoot(GeoDetailPage);
    }
}
