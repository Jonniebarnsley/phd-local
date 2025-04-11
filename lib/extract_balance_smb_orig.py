#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

compute SMB that starts the ice sheet with the corect dh/dt

SMB is something like a = dh/dt + div(uh)

"""

import sys
import os
import glob
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from amrfile import io as amrio
sys.path.append(os.getcwd())
from bisiclesIO import BisiclesData
from osgeo import gdal, osr
from osgeo.gdalconst import *
from netCDF4 import Dataset
#%%
def save_nc(x,y,z_dict,filename):
    nc = Dataset(filename,'w')
    xdim = nc.createDimension('x',size=len(x))
    ydim = nc.createDimension('y',size=len(y))

    xvar = nc.createVariable('x','f8',('x'))
    yvar = nc.createVariable('y','f8',('y'))
    for k in z_dict:
        var  = nc.createVariable(k,'f8',('y','x'))   
        var[:,:] = z_dict[k]
        print (filename, k, np.min(var[:,:]), np.max(var[:,:]))
                
    xvar[:] = x
    yvar[:] = y

    nc.close()
    
def read_raster(raster_path):
    """        
    Opens a tiff as specified by the user    
    Returns an array of the raster with co-oordinates
    """
    from osgeo import gdal,gdalconst  
    import numpy as np
    
    driver = gdal.GetDriverByName('Gtiff')
    driver.Register()
    src = gdal.Open(raster_path, gdalconst.GA_ReadOnly)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)

    
    data=src.ReadAsArray()
    print("Opened %s" %(raster_path))
    #print(src.GetMetadata())
    
    tol = 1.0e-6
    x = np.arange(ulx,lrx-tol, +xres)
    y = np.arange(lry,uly-tol, -yres)
    
    return x,y,np.flipud(data[:,:])*365 # m/day -> m/a



#flux divergence in model
plot_file = 'plot/plot.relaxed_state_test.1km.000000.2d.hdf5'
bike = BisiclesData(plot_file,level=0,iord=0)

uh = bike.xvel * bike.thk
vh = bike.yvel * bike.thk

n,m = bike.thk.shape

divuh = np.zeros((n,m))
dx = bike.x[1] - bike.x[0]

divuh[1:n-1,1:m-1]  = 0.5/dx *( 
    (uh[1:n-1,2:m] - uh[1:n-1,0:m-2]) + 
    (vh[2:n,1:m-1] - vh[0:n-2,1:m-1]))




#obs dhdt

xr, yr, dhdtr = read_raster('ais_dhdt_grounded_filt.tif')
xr -= (-3067750 - 4000)
yr -= (-3067750 - 4000)
dhdtr = np.where(np.isnan(dhdtr),0.0,dhdtr)/365.0
dhdt = RectBivariateSpline(yr, xr, dhdtr)(bike.y, bike.x)


#MAR SMB
amrid = amrio.load('antarctica.mon-SMB-MAR_ERA5-1980-2021.mean.smb.hdf5')
lo,hi = amrio.queryDomainCorners(amrid,0)
x,y,mar_smb = amrio.readBox2D(amrid, 0, lo, hi, 'SMB', 0)
amrio.free(amrid)



MAX_SMB = 2.0
MIN_SMB = -1.0

# set smb = divuh + dh/dt on grounded, MAR SMB on shelf
smb = np.where(bike.hab > 0, divuh + dhdt, mar_smb)
smb = np.where(smb < MIN_SMB, MIN_SMB, smb) 
smb = np.where(smb > MAX_SMB, MAX_SMB, smb) 

smb = gaussian_filter(smb, sigma=4) # smooth
#smb = np.where(smb > mar_smb, mar_smb, smb) #limit to < MAR SMB
smb = np.where(smb < 0.0, 0.0, smb) #no surface melting allowed


g = np.where(bike.hab > 0, 1, 0) # grounded mask
sle = dx*dx*1.0e-9 / 360 # convert m^3 to s.l.e


fig = plt.figure(figsize=(16,16))
ax1 = fig.add_subplot(2,2,1,aspect='equal')
ax2 = fig.add_subplot(2,2,2,aspect='equal')
ax3 = fig.add_subplot(2,2,3,aspect='equal')
ax4 = fig.add_subplot(2,2,4,aspect='equal')

ax1.pcolormesh(smb, vmin=-1,vmax=1,cmap='RdBu_r') 
ax2.pcolormesh(mar_smb, vmin=-1,vmax=1,cmap='RdBu_r') 
ax3.pcolormesh(g*(smb - divuh), vmin=-1,vmax=1,cmap='RdBu_r')
ax4.pcolormesh(dhdt , vmin=-1,vmax=1,cmap='RdBu_r')


print(f'smb - mar_smb  = {np.sum(smb-mar_smb)*sle} mm/a')

print(f'dh/dt  = {np.sum(g*(smb-divuh))*sle} mm/a')
print(f'dh/dt_mar  = {np.sum(g*(mar_smb-divuh))*sle} mm/a')

save_nc(x,y,{'SMB':smb},'antarctica.background.smb.nc')
