;**********************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin


  g = 9.80665
  rho = 6371.220
  pi = 4. * atan(1.)

;;;;This script reads in the computed 1995-2100 anomaly    ;;;;;;;;;
;;;;file and the computed 1995-2014 climo mean, & outputs  ;;;;;;;;;
;;;;fields interpolated to the 8km ISMIP grid.             ;;;;;;;;;
;;;;The interpolation uses cubic splines based on the      ;;;;;;;;;
;;;;csagrid package.                                       ;;;;;;;;;

;;;;1. Read in anomlay and clim files.                     ;;;;;;;;;

  nc_infile1 = addfile("UKESM1-0-LL_anomaly_g6sulfur_1995-2100.nc", "r")
  lat = nc_infile1->lat
  lon = nc_infile1->lon
  time = nc_infile1->time
  year = nc_infile1->year
  var1 = nc_infile1->pr_anomaly
  var2 = nc_infile1->evspsbl_anomaly
  var3 = nc_infile1->mrro_anomaly
  var4 = nc_infile1->smb_anomaly
  var5 = nc_infile1->ts_anomaly

  nc_infile2 = addfile("UKESM1-0-LL_clim_g6sulfur_1995-2014.nc", "r")
  var1_climo = nc_infile2->pr_clim
  var2_climo = nc_infile2->evspsbl_clim
  var3_climo = nc_infile2->mrro_clim
  var4_climo = nc_infile2->smb_clim
  var5_climo = nc_infile2->ts_clim

  printVarSummary(var5)


;;;;2. To keep the output grid lat & lon points within the ;;;;;;;;;
;;;;   range of the native CMIP grids, add cyclical points ;;;;;;;;;
;;;;   at the edges of the native grid.                    ;;;;;;;;;

;;;;   For example, if the native grid is one-degree       ;;;;;;;;;
;;;;   longitude from 0 to 359, and the output grid        ;;;;;;;;;
;;;;   requires an interpolated value at 359.5, the        ;;;;;;;;;
;;;;   routine will have a problem, which is remedied by   ;;;;;;;;;
;;;;   cyclically extending the longitudes.                ;;;;;;;;;

;;;;   A special case arises when the model native grid    ;;;;;;;;;
;;;;   lacks a point at the pole, as the output grid       ;;;;;;;;;
;;;;   requires a value at 90S. In this case (shown here), ;;;;;;;;;
;;;;   the grid is extended by one in the latitudinal      ;;;;;;;;;
;;;;   direction, and a pole value is computed as          ;;;;;;;;;
;;;;   the average of the closest zone.                    ;;;;;;;;;



;;;;   Extend latitudes by one, for the pole.              ;;;;;;;;;

  latp1 = new( dimsizes(lat) + 1, typeof(lat))
  latp1(1::) = lat
  latp1(0) = -90.


;;;;   Cyclically extend longitudes.                       ;;;;;;;;;

  lonp1 = new( dimsizes(lon)+6, typeof(lon))
  lonp1(3:dimsizes(lon)+2) = lon
  lonp1(dimsizes(lon)+3) = lonp1(3) + 360.
  lonp1(dimsizes(lon)+4) = lonp1(4) + 360.
  lonp1(dimsizes(lon)+5) = lonp1(5) + 360.
  lonp1(0) = lonp1(dimsizes(lon)) - 360.
  lonp1(1) = lonp1(dimsizes(lon)+1) - 360.
  lonp1(2) = lonp1(dimsizes(lon)+2) - 360.

  latp1!0 = "latp1"
  latp1&latp1 = latp1
  lonp1!0 = "lonp1"
  lonp1&lonp1 = lonp1

  copy_VarAtts(lat, latp1)
  copy_VarAtts(lon, lonp1)
  print(lonp1)
  printVarSummary(latp1)
  printVarSummary(lonp1)

;;;;   Create indices for native lats and lons, with cyclics.;;;;;;;;;

  ii = ispan(0, dimsizes(lonp1) - 1, 1)
  ii!0 = "lonp1"
  ii&lonp1 = lonp1
  jj = ispan(0, dimsizes(latp1) - 1, 1)
  jj!0 = "latp1"
  jj&latp1 = latp1
  copy_VarCoords(lonp1, ii)
  printVarSummary(ii)
  printVarSummary(jj)

;;;;   Copy original fields into variables with cyclics.   ;;;;;;;;;
;;;;   Repeat for all variables - anoms and climo.         ;;;;;;;;;

  var1p1_climo = new( (/dimsizes(latp1), dimsizes(lonp1)/), typeof(var1_climo))
  print(dimsizes(var1_climo))
  print(dimsizes(var1p1_climo))
  var1p1_climo(1::,3:dimsizes(lon)+2) = var1_climo
  var1p1_climo(0,:) = avg(var1_climo(0,:))
  var1p1_climo(:,dimsizes(lon)+3) = var1p1_climo(:,3)
  var1p1_climo(:,dimsizes(lon)+4) = var1p1_climo(:,4)
  var1p1_climo(:,dimsizes(lon)+5) = var1p1_climo(:,5)
  var1p1_climo(:,0) = var1p1_climo(:,dimsizes(lon))
  var1p1_climo(:,1) = var1p1_climo(:,dimsizes(lon)+1)
  var1p1_climo(:,2) = var1p1_climo(:,dimsizes(lon)+2)
  var1p1_climo!0 = "latp1"
  var1p1_climo!1 = "lonp1"
  var1p1_climo&latp1 = latp1
  var1p1_climo&lonp1 = lonp1

  var2p1_climo = new( (/dimsizes(latp1), dimsizes(lonp1)/), typeof(var2_climo))
  var2p1_climo(1::,3:dimsizes(lon)+2) = var2_climo
  var2p1_climo(0,:) = avg(var2_climo(0,:))
  var2p1_climo(:,dimsizes(lon)+3) = var2p1_climo(:,3)
  var2p1_climo(:,dimsizes(lon)+4) = var2p1_climo(:,4)
  var2p1_climo(:,dimsizes(lon)+5) = var2p1_climo(:,5)
  var2p1_climo(:,0) = var2p1_climo(:,dimsizes(lon))
  var2p1_climo(:,1) = var2p1_climo(:,dimsizes(lon)+1)
  var2p1_climo(:,2) = var2p1_climo(:,dimsizes(lon)+2)
  copy_VarCoords(var1p1_climo, var2p1_climo)

  var3p1_climo = new( (/dimsizes(latp1), dimsizes(lonp1)/), typeof(var3_climo))
  var3p1_climo(1::,3:dimsizes(lon)+2) = var3_climo
  var3p1_climo(0,:) = avg(var3_climo(0,:))
  var3p1_climo(:,dimsizes(lon)+3) = var3p1_climo(:,3)
  var3p1_climo(:,dimsizes(lon)+4) = var3p1_climo(:,4)
  var3p1_climo(:,dimsizes(lon)+5) = var3p1_climo(:,5)
  var3p1_climo(:,0) = var3p1_climo(:,dimsizes(lon))
  var3p1_climo(:,1) = var3p1_climo(:,dimsizes(lon)+1)
  var3p1_climo(:,2) = var3p1_climo(:,dimsizes(lon)+2)
  copy_VarCoords(var1p1_climo, var3p1_climo)

  var4p1_climo = new( (/dimsizes(latp1), dimsizes(lonp1)/), typeof(var4_climo))
  var4p1_climo(1::,3:dimsizes(lon)+2) = var4_climo
  var4p1_climo(0,:) = avg(var4_climo(0,:))
  var4p1_climo(:,dimsizes(lon)+3) = var4p1_climo(:,3)
  var4p1_climo(:,dimsizes(lon)+4) = var4p1_climo(:,4)
  var4p1_climo(:,dimsizes(lon)+5) = var4p1_climo(:,5)
  var4p1_climo(:,0) = var4p1_climo(:,dimsizes(lon))
  var4p1_climo(:,1) = var4p1_climo(:,dimsizes(lon)+1)
  var4p1_climo(:,2) = var4p1_climo(:,dimsizes(lon)+2)
  copy_VarCoords(var1p1_climo, var4p1_climo)

  var5p1_climo = new( (/dimsizes(latp1), dimsizes(lonp1)/), typeof(var5_climo))
  var5p1_climo(1::,3:dimsizes(lon)+2) = var5_climo
  var5p1_climo(0,:) = avg(var5_climo(0,:))
  var5p1_climo(:,dimsizes(lon)+3) = var5p1_climo(:,3)
  var5p1_climo(:,dimsizes(lon)+4) = var5p1_climo(:,4)
  var5p1_climo(:,dimsizes(lon)+5) = var5p1_climo(:,5)
  var5p1_climo(:,0) = var5p1_climo(:,dimsizes(lon))
  var5p1_climo(:,1) = var5p1_climo(:,dimsizes(lon)+1)
  var5p1_climo(:,2) = var5p1_climo(:,dimsizes(lon)+2)
  copy_VarCoords(var1p1_climo, var5p1_climo)

  var1p1 = new( (/dimsizes(time), dimsizes(latp1), dimsizes(lonp1)/), typeof(var1))
  var1p1(:,1::,3:dimsizes(lon)+2) = var1
  var1p1(:,0,:) = avg(var1(:,0,:))
  var1p1(:,:,dimsizes(lon)+3) = var1p1(:,:,3)
  var1p1(:,:,dimsizes(lon)+4) = var1p1(:,:,4)
  var1p1(:,:,dimsizes(lon)+5) = var1p1(:,:,5)
  var1p1(:,:,0) = var1p1(:,:,dimsizes(lon))
  var1p1(:,:,1) = var1p1(:,:,dimsizes(lon)+1)
  var1p1(:,:,2) = var1p1(:,:,dimsizes(lon)+2)
  var1p1!0 = "time"
  var1p1!1 = "latp1"
  var1p1!2 = "lonp1"
  var1p1&time = time
  var1p1&latp1 = latp1
  var1p1&lonp1 = lonp1

  var2p1 = new( (/dimsizes(time), dimsizes(latp1), dimsizes(lonp1)/), typeof(var2))
  var2p1(:,1::,3:dimsizes(lon)+2) = var2
  var2p1(:,0,:) = avg(var2(:,0,:))
  var2p1(:,:,dimsizes(lon)+3) = var2p1(:,:,3)
  var2p1(:,:,dimsizes(lon)+4) = var2p1(:,:,4)
  var2p1(:,:,dimsizes(lon)+5) = var2p1(:,:,5)
  var2p1(:,:,0) = var2p1(:,:,dimsizes(lon))
  var2p1(:,:,1) = var2p1(:,:,dimsizes(lon)+1)
  var2p1(:,:,2) = var2p1(:,:,dimsizes(lon)+2)
  copy_VarCoords(var1p1, var2p1)

  var3p1 = new( (/dimsizes(time), dimsizes(latp1), dimsizes(lonp1)/), typeof(var3))
  var3p1(:,1::,3:dimsizes(lon)+2) = var3
  var3p1(:,0,:) = avg(var3(:,0,:))
  var3p1(:,:,dimsizes(lon)+3) = var3p1(:,:,3)
  var3p1(:,:,dimsizes(lon)+4) = var3p1(:,:,4)
  var3p1(:,:,dimsizes(lon)+5) = var3p1(:,:,5)
  var3p1(:,:,0) = var3p1(:,:,dimsizes(lon))
  var3p1(:,:,1) = var3p1(:,:,dimsizes(lon)+1)
  var3p1(:,:,2) = var3p1(:,:,dimsizes(lon)+2)
  copy_VarCoords(var1p1, var3p1)

  var4p1 = new( (/dimsizes(time), dimsizes(latp1), dimsizes(lonp1)/), typeof(var4))
  var4p1(:,1::,3:dimsizes(lon)+2) = var4
  var4p1(:,0,:) = avg(var4(:,0,:))
  var4p1(:,:,dimsizes(lon)+3) = var4p1(:,:,3)
  var4p1(:,:,dimsizes(lon)+4) = var4p1(:,:,4)
  var4p1(:,:,dimsizes(lon)+5) = var4p1(:,:,5)
  var4p1(:,:,0) = var4p1(:,:,dimsizes(lon))
  var4p1(:,:,1) = var4p1(:,:,dimsizes(lon)+1)
  var4p1(:,:,2) = var4p1(:,:,dimsizes(lon)+2)
  copy_VarCoords(var1p1, var4p1)

  var5p1 = new( (/dimsizes(time), dimsizes(latp1), dimsizes(lonp1)/), typeof(var5))
  var5p1(:,1::,3:dimsizes(lon)+2) = var5
  var5p1(:,0,:) = avg(var5(:,0,:))
  var5p1(:,:,dimsizes(lon)+3) = var5p1(:,:,3)
  var5p1(:,:,dimsizes(lon)+4) = var5p1(:,:,4)
  var5p1(:,:,dimsizes(lon)+5) = var5p1(:,:,5)
  var5p1(:,:,0) = var5p1(:,:,dimsizes(lon))
  var5p1(:,:,1) = var5p1(:,:,dimsizes(lon)+1)
  var5p1(:,:,2) = var5p1(:,:,dimsizes(lon)+2)
  copy_VarCoords(var1p1, var5p1)


;;;;3. Read in the output grid's lats & lons.              ;;;;;;;;;

  nc_infile7 = addfile("/work/n02/shared/madhkri_/practice_richard_NCL/pr_Amon_CCSM4_r1i1p1_199501-210012_8km_ISMIP6GRID.nc", "r")
  lat_out = nc_infile7->lat
  lon_out = nc_infile7->lon
  lon_out = where(lon_out.lt.0., lon_out + 360., lon_out)

  xx = ispan(0, 760, 1)
  yy = ispan(0, 760, 1)
  lat_out!0 = "y"
  lat_out!1 = "x"
  lat_out&y = yy
  lat_out&x = xx
  copy_VarCoords(lat_out, lon_out)

;;;;4. Create arrays for the output variables.             ;;;;;;;;;

  var1_out = new( (/ dimsizes(time), 761, 761/), typeof(var1), var1@_FillValue)
  var2_out = new( (/ dimsizes(time), 761, 761/), typeof(var2), var2@_FillValue)
  var3_out = new( (/ dimsizes(time), 761, 761/), typeof(var3), var3@_FillValue)
  var4_out = new( (/ dimsizes(time), 761, 761/), typeof(var4), var4@_FillValue)
  var5_out = new( (/ dimsizes(time), 761, 761/), typeof(var5), var5@_FillValue)
  var1_climo_out = new( (/ 761, 761/), typeof(var1_climo), var1_climo@_FillValue)
  var2_climo_out = new( (/ 761, 761/), typeof(var2_climo), var2_climo@_FillValue)
  var3_climo_out = new( (/ 761, 761/), typeof(var3_climo), var3_climo@_FillValue)
  var4_climo_out = new( (/ 761, 761/), typeof(var4_climo), var4_climo@_FillValue)
  var5_climo_out = new( (/ 761, 761/), typeof(var5_climo), var5_climo@_FillValue)

;;;;5. Create arrays for the spline knots.                 ;;;;;;;;;

  knots = (/7, 5/)
  var1_1d = new( (/dimsizes(time), 35/), typeof(var1), var1@_FillValue)
  var2_1d = new( (/dimsizes(time), 35/), typeof(var2), var2@_FillValue)
  var3_1d = new( (/dimsizes(time), 35/), typeof(var3), var3@_FillValue)
  var4_1d = new( (/dimsizes(time), 35/), typeof(var4), var4@_FillValue)
  var5_1d = new( (/dimsizes(time), 35/), typeof(var5), var5@_FillValue)


;;;;6. Loop through the output lats and lons.              ;;;;;;;;;

  do y = 0, 760
    print(y+" ")
    do x = 0, 760

;;;;7. Using the native grid indices, (cleverly) determine ;;;;;;;;;
;;;;   the nearest native grid points to the output lat &  ;;;;;;;;;
;;;;   lon location, and put them into a 1-d array.        ;;;;;;;;;

      jval = jj( {lat_out(y,x)} )
      jbeg = jval - 3
      jend = jval + 3
      if (jbeg.lt.0) then
        jbeg = 0
        jend = 6
      end if

      ival = ii( {lon_out(y,x)} )
      ibeg = ival - 2
      iend = ival + 2

;     print(x+" "+ibeg+" "+iend+" "+jbeg+" "+jend)

      anum = 0.
      adnm = 0.
      var1_climo_1d = ndtooned(var1p1_climo(jbeg:jend, ibeg:iend))
      var2_climo_1d = ndtooned(var2p1_climo(jbeg:jend, ibeg:iend))
      var3_climo_1d = ndtooned(var3p1_climo(jbeg:jend, ibeg:iend))
      var4_climo_1d = ndtooned(var4p1_climo(jbeg:jend, ibeg:iend))
      var5_climo_1d = ndtooned(var5p1_climo(jbeg:jend, ibeg:iend))
      do k = 0, dimsizes(time) - 1
        var1_1d(k,:) = ndtooned(var1p1(k,jbeg:jend, ibeg:iend))
        var2_1d(k,:) = ndtooned(var2p1(k,jbeg:jend, ibeg:iend))
        var3_1d(k,:) = ndtooned(var3p1(k,jbeg:jend, ibeg:iend))
        var4_1d(k,:) = ndtooned(var4p1(k,jbeg:jend, ibeg:iend))
        var5_1d(k,:) = ndtooned(var5p1(k,jbeg:jend, ibeg:iend))
      end do

      lat_in_1d = ndtooned( conform(var5p1_climo(jbeg:jend, ibeg:iend), latp1(jbeg:jend), 0) )
      lon_in_1d = ndtooned( conform(var5p1_climo(jbeg:jend, ibeg:iend), lonp1(ibeg:iend), 1) )


;;;;8. Apply the spline interpolation.                     ;;;;;;;;;

      var1_climo_out(y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var1_climo_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var2_climo_out(y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var2_climo_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var3_climo_out(y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var3_climo_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var4_climo_out(y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var4_climo_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var5_climo_out(y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var5_climo_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var1_out(:,y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var1_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var2_out(:,y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var2_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var3_out(:,y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var3_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var4_out(:,y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var4_1d, knots, lat_out(y,x), lon_out(y,x) ) )
      var5_out(:,y,x) = tofloat( csa2( lat_in_1d, lon_in_1d, var5_1d, knots, lat_out(y,x), lon_out(y,x) ) )
     

    end do
  end do

;;;;9. Attach coordinate attributes to the output variables.;;;;;;;;
;;;;   Clean up and output.                                 ;;;;;;;;


  copy_VarAtts(var1, var1_out)
  copy_VarAtts(var2, var2_out)
  copy_VarAtts(var3, var3_out)
  copy_VarAtts(var4, var4_out)
  copy_VarAtts(var5, var5_out)
  copy_VarAtts(var1_climo, var1_climo_out)
  copy_VarAtts(var2_climo, var2_climo_out)
  copy_VarAtts(var3_climo, var3_climo_out)
  copy_VarAtts(var4_climo, var4_climo_out)
  copy_VarAtts(var5_climo, var5_climo_out)
  copy_VarCoords(lat_out, var1_climo_out)
  copy_VarCoords(lat_out, var2_climo_out)
  copy_VarCoords(lat_out, var3_climo_out)
  copy_VarCoords(lat_out, var4_climo_out)
  copy_VarCoords(lat_out, var5_climo_out)
  var1_out!0 = "time"
  var1_out!1 = "y"
  var1_out!2 = "x"
  var1_out&time = time
  var1_out&y = yy
  var1_out&x = xx
  copy_VarCoords(var1_out, var2_out)
  copy_VarCoords(var1_out, var3_out)
  copy_VarCoords(var1_out, var4_out)
  copy_VarCoords(var1_out, var5_out)

  dim_names = (/ "time", "y",  "x" /)
  dim_sizes = (/ dimsizes(time), dimsizes(yy), dimsizes(xx) /)
  dimUnlim = (/ True, False , False  /)

  fout1 = addfile("UKESM1-0-LL_8km_anomaly_g6sulfur_1995-2100.nc", "c")
  filedimdef(fout1, dim_names, dim_sizes, dimUnlim)
  fout1->pr_anomaly=var1_out
  fout1->evspsbl_anomaly=var2_out
  fout1->mrro_anomaly=var3_out
  fout1->smb_anomaly=var4_out
  fout1->ts_anomaly=var5_out
  fout1->lat2d=lat_out
  fout1->lon2d=lon_out
  fout1->year=year

  dim_names2 = (/ "y",  "x" /)
  dim_sizes2 = (/ dimsizes(yy), dimsizes(xx) /)
  dimUnlim2 = (/ False , False  /)

  fout2 = addfile("UKESM1-0-LL_8km_clim_g6sulfur_1995-2014.nc", "c")
  filedimdef(fout2, dim_names2, dim_sizes2, dimUnlim2)
  fout2->pr_clim=var1_climo_out
  fout2->evspsbl_clim=var2_climo_out
  fout2->mrro_clim=var3_climo_out
  fout2->smb_clim=var4_climo_out
  fout2->ts_clim=var5_climo_out
  fout2->lat2d=lat_out
  fout2->lon2d=lon_out



end


