import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping

def clip_dataset_with_shapefile(ds: xr.Dataset, shape: gpd.GeoDataFrame, all_touched: bool = False) -> xr.Dataset:

    """
    Clip an xarray Dataset using a shapefile.

    Parameters:
    - ds: The input xarray Dataset.
    - shape: shapefile as a Geopandas geodataframe.
    - all_touched: whether to clip any cell that touches the shapefile edge

    Returns:
    - An xarray Dataset clipped by the shapefile.
    """
    
    # Convert the dataset to a rioxarray object (if it's not already)
    ds_rio = ds.rio.write_crs(shape.crs.to_string())
        
    # Clip the dataset
    # from rio.clip documentation:
    # all_touched : boolean, optional
    #     If True, all pixels touched by geometries will be burned in.  If
    #     false, only pixels whose center is within the polygon or that
    #     are selected by Bresenham's line algorithm will be burned in.
    ds_clipped = ds_rio.rio.clip(shape.geometry.apply(mapping), all_touched=all_touched)

    # Interpolate back onto the original grid to avoid cropping done by rio.clip
    # Must first take the transpose because the Rignot gdf data is, for some reason,
    # described in (y, x) coordinates instead of (x, y).
    clipped_on_full_grid = ds_clipped.T.interp(coords=ds.coords, method='nearest')

    return clipped_on_full_grid
