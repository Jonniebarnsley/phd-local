;**********************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin


  g = 9.80665
  rho = 6371.220
  pi = 4. * atan(1.)

;;;;This script reads in 1995-2100 CMIP files, computes    ;;;;;;;;;
;;;;1995-2014 climo mean, 9nd 1995-2100 anomalies on native;;;;;;;;;
;;;;grid, and outputs.                                     ;;;;;;;;;

;;;;1. Read in CMIP5 variable files & assign to variables. ;;;;;;;;;
;;;;;; The input files are a combination of historical & rcp. ;;;;;;
;;;;;; var1 is pr, var2 is evspsbl, var3 is mrro ;;;;;;;;;;;;;;;;;;;
;;;;;; var4 is mrros (if available/necessary), var5 is ts ;;;;;;;;;; 

  fili1 = systemfunc("ls /Volumes/LaCie/CMIP6/CMIP/NCAR/CESM2-WACCM/historical/r1i1p1f1/Amon/pr/gn/v20190415/pr_Amon_CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc")
  nc_infile1 = addfiles(fili1, "r")
  lat = nc_infile1[0]->lat
  lon = nc_infile1[0]->lon
  time1 = nc_infile1[:]->time
  var1 = nc_infile1[:]->pr

  fili2 = systemfunc("ls /Volumes/LaCie/CMIP6/CMIP/NCAR/CESM2-WACCM/historical/r1i1p1f1/Amon/evspsbl/gn/v20190227/evspsbl_Amon_CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc")
  nc_infile2 = addfiles(fili2, "r")
  var2 = nc_infile2[:]->evspsbl

  fili3 = systemfunc("ls /Volumes/LaCie/CMIP6/CMIP/NCAR/CESM2-WACCM/historical/r1i1p1f1/Lmon/mrro/gn/v20190227/mrro_Lmon_CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc")
  nc_infile3 = addfiles(fili3, "r")
  var3 = nc_infile3[:]->mrro
  var3 = where(ismissing(var3), 0., var3)

  fili5 = systemfunc("ls /Volumes/LaCie/CMIP6/CMIP/NCAR/CESM2-WACCM/historical/r1i1p1f1/Amon/tas/gn/v20190227/tas_Amon_CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc")
  nc_infile5 = addfiles(fili5, "r")
  var5 = nc_infile5[:]->ts

;;;;;; Retrieve dates from the time variable within each file.  ;;;;

  utc_date = cd_calendar(time1, 0)
  iyear = tointeger(utc_date(:,0))
  imonth = tointeger(utc_date(:,1))
  iyrmo = iyear * 100 + imonth

;;;;;; Convert variables to annual averages for years 1995-2100.;;;;
;;;;;; Those years were requested first, the 1950-1994 period   ;;;; 
;;;;;; came later, so that's how the script evolved.            ;;;; 
;;;;;; A second script is used to compute anomalies for         ;;;;
;;;;;; the earlier period, & eventually 2010-2300.              ;;;;
;;;;;; Some models use a Babylonian calendar, so                ;;;; 
;;;;;; assume months are equally weighted to compute            ;;;;
;;;;;; annual average - only recourse.                          ;;;;


  var1_yr = new( (/ 175, dimsizes(lat), dimsizes(lon) /), typeof(var1), var1@_FillValue)
  var2_yr = new( (/ 175, dimsizes(lat), dimsizes(lon) /), typeof(var2), var2@_FillValue)
  var3_yr = new( (/ 175, dimsizes(lat), dimsizes(lon) /), typeof(var3), var3@_FillValue)
  var5_yr = new( (/ 175, dimsizes(lat), dimsizes(lon) /), typeof(var5), var5@_FillValue)
  time = new( 175, typeof(time1), "No_FillValue")


  do k = 0, 174
    kyr = k + 1850 ;!
    idx = ind( iyear.eq.kyr)
    printVarSummary(idx)
    var1_yr(k,:,:) = dim_avg_n (var1(idx,:,:), 0)
    var2_yr(k,:,:) = dim_avg_n (var2(idx,:,:), 0)
    var3_yr(k,:,:) = dim_avg_n (var3(idx,:,:), 0)
    var5_yr(k,:,:) = dim_avg_n (var5(idx,:,:), 0)

    time(k) = time1(idx(0))
    delete(idx)
  end do

;;;;3. Construct a new time variable for the annual averages.   ;;;;
;;;;;; Assign the annual-average variables coordinate           ;;;; 
;;;;;; information - time, lat, lon                             ;;;; 

  copy_VarAtts( time1, time )
  print("Dimensions of time: " + dimsizes(time))
  print("Dimensions of time1: " + dimsizes(time1))
  var1_yr!0 = "time"
  var1_yr!1 = "lat"
  var1_yr!2 = "lon"
  var1_yr&time = time
  var1_yr&lat = lat
  var1_yr&lon = lon
  copy_VarCoords( var1_yr, var2_yr)
  copy_VarCoords( var1_yr, var3_yr)
  copy_VarCoords( var1_yr, var5_yr)
  copy_VarAtts( var1, var1_yr )
  copy_VarAtts( var2, var2_yr )
  copy_VarAtts( var3, var3_yr )
  copy_VarAtts( var5, var5_yr )
  printVarSummary(var1_yr)

;;;;4. Compute the 1995-2014 climo averages.                    ;;;;
;;;;   Construct new variable smb.                              ;;;;

  var1_climo = dim_avg_n_Wrap( var1_yr(0:19,:,:), 0)
  var2_climo = dim_avg_n_Wrap( var2_yr(0:19,:,:), 0)
  var3_climo = dim_avg_n_Wrap( var3_yr(0:19,:,:), 0)
  var5_climo = dim_avg_n_Wrap( var5_yr(0:19,:,:), 0)

  smb_climo = var1_climo - var2_climo - var3_climo
  copy_VarCoords(var1_climo, smb_climo)
  copy_VarAtts(var1_climo, smb_climo)
  smb_climo@standard_name = "smb_flux"
  smb_climo@long_name = "Surface_Mass_Balance"
  delete(smb_climo@original_name)

  printVarSummary(var1_climo)

;;;;5. Compute var anomalies using the 1995-2014 climo averages.;;;;

  var1_anom = var1_yr
  var2_anom = var2_yr
  var3_anom = var3_yr
  var5_anom = var5_yr

  var1_anom =  var1_yr - conform( var1_yr, var1_climo, (/1,2/))
  var2_anom =  var2_yr - conform( var2_yr, var2_climo, (/1,2/))
  var3_anom =  var3_yr - conform( var3_yr, var3_climo, (/1,2/))
  var5_anom =  var5_yr - conform( var5_yr, var5_climo, (/1,2/))
  smb_anom = var1_anom - var2_anom - var3_anom
  copy_VarCoords(var1_anom, smb_anom)
  copy_VarAtts(var1_anom, smb_anom)
  smb_anom@standard_name = "smb_flux"
  smb_anom@long_name = "Surface_Mass_Balance"
  delete(smb_anom@original_name)


  printVarSummary(var1_anom)


;;;;6. Clean up & output.                                       ;;;;

  utc_date_out = cd_calendar(time, 0)
  iyear_out = tointeger(utc_date_out(:,0))
  iyear_out!0 = "time"
  iyear_out&time = time
  iyear_out@standard_name = "year"
  iyear_out@long_name = "Year"

  dim_names = (/ "time", "lat",  "lon" /)
  dim_sizes = (/ dimsizes(time), dimsizes(lat), dimsizes(lon) /)
  dimUnlim = (/ True, False , False  /)

  fout1 = addfile("CESM2-WACCM_anomaly_historical_1850-2014.nc", "c")
  filedimdef(fout1, dim_names, dim_sizes, dimUnlim)
  fout1->pr_anomaly=var1_anom
  fout1->evspsbl_anomaly=var2_anom
  fout1->mrro_anomaly=var3_anom
  fout1->ts_anomaly=var5_anom
  fout1->smb_anomaly=smb_anom
  fout1->year=iyear_out

  dim_names2 = (/ "lat",  "lon" /)
  dim_sizes2 = (/ dimsizes(lat), dimsizes(lon) /)
  dimUnlim2 = (/ False , False  /)

  fout2 = addfile("CESM2-WACCM_clim_historical_1850-2014.nc", "c")
  filedimdef(fout2, dim_names2, dim_sizes2, dimUnlim2)
  fout2->pr_clim=var1_climo
  fout2->evspsbl_clim=var2_climo
  fout2->mrro_clim=var3_climo
  fout2->ts_clim=var5_climo
  fout2->smb_clim=smb_climo





end


