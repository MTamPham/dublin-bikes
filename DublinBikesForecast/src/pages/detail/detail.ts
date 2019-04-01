import { Component } from '@angular/core';
import { IonicPage, NavController, NavParams, LoadingController } from 'ionic-angular';
import { Station } from '../../data-model/Station';
import { StationProvider } from '../../providers/station/station';
import { CommonProvider } from '../../providers/common/common';
import { Observable, of } from 'rxjs';
import { HomePage } from '../home/home';
import { GeoDetailPage } from '../geo-detail/geo-detail';

/**
 * Generated class for the DetailPage page.
 *
 * See https://ionicframework.com/docs/components/#navigation for more info on
 * Ionic pages and navigation.
 */

@IonicPage()
@Component({
  selector: 'page-detail',
  templateUrl: 'detail.html',
})
export class DetailPage {
    actionType: number;
    selectedStation: Station;
    stationList: Station[] = new Array();    // deprecated
    stationList$: Observable<Station[]>;
    periodList: any[] = new Array();
    period: any;
    placeholder: string;
    loading: any;
    navigateFromGeoLocationPage: boolean = false;

  constructor(public navCtrl: NavController, public navParams: NavParams, public commonProvider: CommonProvider,
              public stationProvider: StationProvider, public loadingController: LoadingController) {
        this.stationList = this.stationProvider.getAllStations();    // deprecated
        this.periodList = this.commonProvider.getPeriodList();
        this.actionType = navParams.get("type");
        switch(this.actionType) {
            case 0:
                this.placeholder = "Filter by station number";
                break;
            case 1:
                this.placeholder = "Filter by station address";
                break;
        }
        this.stationList$ = this.stationProvider.getAllStationsOnFirebase();
        this.selectedStation = navParams.get("selectedStation");
        if (this.selectedStation != null) {
            this.navigateFromGeoLocationPage = true;
            this.predict(this.selectedStation);
        }
  }

  public search(searchTerm: string) {
      switch(this.actionType) {
          case 0:
              this.searchByNumber(searchTerm);
              break;
          case 1:
              this.searchByAddress(searchTerm);
              break;
      }
  }

  private searchByNumber(searchTerm: string) {
      if (searchTerm.length <= 0) {
        this.stationList$ = this.stationProvider.getAllStationsOnFirebase();
        return;
      }

      var number = Number(searchTerm);
      this.stationList$ = this.stationProvider.searchByNumberOnFirebase(number);
  }

  private searchByAddress(searchTerm: string) {
      if (searchTerm.length <= 0) {
        this.stationList$ = this.stationProvider.getAllStationsOnFirebase();
        return;
      }

      var address = new String(searchTerm).toLowerCase();
      this.stationList$ = this.stationProvider.searchByAddressOnFirebase(address);  
  }

  private async predict(clickedStation: Station) {
      this.selectedStation = clickedStation;
    // initalize ion-loading
    this.loading = this.loadingController.create({
        content: 'Retrieving result...'
    });
    // present loading view
    this.loading.present();    
    // call API to get the result
    var response = await this.stationProvider.predict(clickedStation.number, this.period);
    // hide loading view
    this.loading.dismiss();
    // update station object
    this.stationList$.subscribe(
        stations => {
            let index = stations.findIndex(s => this.commonProvider.equals(s, clickedStation));
            stations[index] = { ...response};
            // turn an array into an observable
            this.stationList$ = of(stations);
        },
        error => console.warn("Error: " + error)        
    );
  }

  public goToHome() {
    this.navCtrl.setRoot(HomePage);
  }

  public goToGeoDetail() {
      this.navCtrl.setRoot(GeoDetailPage);
  }

  public changePeriod(event) {
      this.period = event;
      this.predict(this.selectedStation);
  }

  public onCancel(event) {
      this.stationList$ = this.stationProvider.getAllStationsOnFirebase();
  }

  ionViewDidLoad() {
    console.log('ionViewDidLoad DetailPage');
  }
}
