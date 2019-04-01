import { NgModule } from '@angular/core';
import { IonicPageModule } from 'ionic-angular';
import { GeoDetailPage } from './geo-detail';

@NgModule({
  declarations: [
    GeoDetailPage,
  ],
  imports: [
    IonicPageModule.forChild(GeoDetailPage),
  ],
})
export class GeoDetailPageModule {}
