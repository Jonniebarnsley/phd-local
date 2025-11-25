import numpy as np
from osgeo import gdal, gdalconst  

def read_raster(raster_path):
    """        
    Opens a tiff as specified by the user    
    Returns an array of the raster with co-oordinates
    """
    
    driver = gdal.GetDriverByName('Gtiff')
    driver.Register()
    src = gdal.Open(raster_path, gdalconst.GA_ReadOnly)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)

    data=src.ReadAsArray()
    print("Opened %s" %(raster_path))
    
    tol = 1.0e-6
    x = np.arange(ulx, lrx-tol, +xres)
    y = np.arange(lry, uly-tol, -yres)
    
    return x, y, np.flipud(data[:,:])

