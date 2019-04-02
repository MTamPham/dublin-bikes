import { Component, ViewChild, ElementRef } from '@angular/core';
import { IonicPage, NavController, NavParams, LoadingController } from 'ionic-angular';
import { CommonProvider } from '../../providers/common/common';
import { StationProvider } from '../../providers/station/station';
import leaflet from 'leaflet';
import { Station } from '../../data-model/Station';
import { DetailPage } from '../detail/detail';
import { HomePage } from '../home/home';

/**
 * Generated class for the GeoDetailPage page.
 *
 * See https://ionicframework.com/docs/components/#navigation for more info on
 * Ionic pages and navigation.
 */

@IonicPage()
@Component({
  selector: 'page-geo-detail',
  templateUrl: 'geo-detail.html',
})
export class GeoDetailPage {
    @ViewChild('map') mapContainer: ElementRef;
    map: any;
    periodList: any[] = new Array();
    period: any;
    loading: any;

    static LeafIcon = leaflet.Icon.extend({
        options: {
            iconSize:     [40, 40],
        }
    });

  constructor(public navCtrl: NavController, public navParams: NavParams, public commonProvider: CommonProvider,
              public stationProvider: StationProvider, public loadingController: LoadingController) {
  }

  public async predict(clickedStation: Station) {
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
    return response;
  }

  private loadmap() {
    this.map = leaflet.map("map").fitWorld();
    leaflet.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attributions: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="http://mapbox.com">Mapbox</a>',
      maxZoom: 18
    }).addTo(this.map);
    this.map.locate({
      setView: true,
      maxZoom: 10
    }).on('locationfound', (e) => {
        let markerGroup = leaflet.featureGroup();
        let iconMarker = new GeoDetailPage.LeafIcon({ iconUrl: "/assets/icon/white-marker.png" });
        let marker: any = leaflet.marker([e.latitude, e.longitude], {icon: iconMarker});
        markerGroup.addLayer(marker);
        this.map.addLayer(markerGroup);
      }).on('locationerror', (err) => {
        alert(err.message);
    })

  }

  public changePeriod(event, newPeriod) {
      this.period = newPeriod.value;
  }

  public goToDetailPage(station) {
      this.navCtrl.setRoot(DetailPage, {type: 0, selectedStation: station});
  }

  public goToHome() {
    this.navCtrl.setRoot(HomePage);
  }

  ionViewDidLoad() {
    console.log('ionViewDidLoad GeoDetailPage');
  }

  ionViewDidEnter() {
      this.loadmap();
      this.periodList = this.commonProvider.getPeriodList();
      this.stationProvider.getAllStationsOnFirebase().subscribe(stations => {
          let stationList: Station[] = stations as Station[];
          let iconMarker = new GeoDetailPage.LeafIcon({ iconUrl: "/assets/icon/red-marker.png" });
          for (var i = 0; i < stationList.length; i++) {
            let station = stationList[i];
            if (station["open"] == false) {
                continue;
            }
            let position = station["position"];
            let lat = "0";
            let lng = "0";
            if (position != "") {
                lat = position.slice(0, position.indexOf(","));
                lng = position.slice(position.indexOf(",") + 1, position.length);
            }

            let marker: any = leaflet.marker([lat, lng], {icon: iconMarker}).on('click', (marker) => {
                // this.predict(station).then(result => {
                //     let html = "<b>Station" + result.number + "</b>";
                //     html += "<br>";
                //     html += "" + result.name + "<br>";
                //     html += "Available bikes: " + result.availableBikes + "<br>";
                //     html += "Free stands: " + result.availableStands + "<br>";
                //     html += "Total capacity: " + result.bikeStands + "<br>";
                //     html += "<a (href)='goToDetailPage(" + result.number + ")'>More prediction?</a>";
                //     marker.target.bindPopup(html).openPopup();
                // });

                this.goToDetailPage(station);
            });
            marker.addTo(this.map);
          }
      });
  }
}
