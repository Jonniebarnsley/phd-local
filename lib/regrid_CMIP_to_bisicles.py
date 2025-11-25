#!/bin/python

"""
Regrids CMIP data from a lat-lon grid to the BISICLES 8km (768 x 768) South Polar
stereographic grid using pyremap.

Usage: python regrid_CMIP_to_bisicles.py <input_file> <output_file> [--method <method>] [--annual]

--- Options ---
method : Regridding method to use, bilinear or conserve (default: bilinear)
annual : If set, compute annual means before regridding

author: Jonnie
date: November 2025
"""

import os
import argparse
import numpy as np
import xarray as xr
from xarray import Dataset
from pathlib import Path
from pyremap import LatLonGridDescriptor, Remapper
from pyremap.polar import get_polar_descriptor_from_file

# local imports
from bisicles_defaults import GRID_8KM

# ignore warnings of multiple fill values when reading netcdfs
import warnings
from xarray import SerializationWarning
warnings.filterwarnings("ignore", category=SerializationWarning)


DATA_HOME = Path(os.environ.get('DATA_HOME', '~/data/')) # mapping files get saved here
BISICLES_FILE = Path("/Users/jonniebarnsley/code/phd/local/data/bisicles_grid_pole_centered.nc")

class Regridder:
    """
    Regridder class to handle regridding of CMIP data onto the BISICLES grid.
    
    Regridding requires a number of steps:
    
     1. Create a temporary file with a cyclic point / pole coordinate added.
     2. Regrid temporary file onto a South polar stereographic grid.
     3. Relabel the coordinates to match the BISICLES setup.
     4. Save the regridded dataset to a specified output file.
     5. Clean up the temporary and log files created during the process.

    THe regridder class handles all the above whilst saving useful attributes for use across
    functions - for example, a list of temporary files.
    """

    def __init__(self, method: str = "bilinear", annual_mean: bool = False):
        self.method = method
        self.calc_annual_mean = annual_mean
        self.tmp_files = []  # Track all temporary files for cleanup
        self.encoding_specs = {
            'zlib': True,
            'complevel': 4,
            '_FillValue': -9999,
        }

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - clean up temporary / log files"""
        self.clean_up()

    def regrid_CMIP_to_bisicles(self, infile: Path) -> Dataset:
        """Regrids any 2-D CMIP data on a lat-lon grid and return a dataset on the 
        BISICLES 8km-resolution 768x768 South Polar Steregraphic grid"""
        
        # The easiest way to use pyremap is to have two netcdfs that are on the source
        # and destination grids. However, some CMIP data requires adding a cyclic point
        # or pole coordinate, which changes the native grid. We therefore need to create 
        # an intermediate file with these changes made for pyremap to work from.
        tmp_filepath = self._make_tmp_file(infile)

        # Now regrid onto a 768x768 8km grid with the origin on the South pole.
        ds = self._regrid(tmp_filepath)
        
        # Redefine grid coordinates such that the origin is in the bottom left corner.
        xs, ys = GRID_8KM
        ds = ds.assign_coords(x=xs, y=ys)
        
        # Update attributes
        ds.attrs.update({
            "grid": "BISICLES 8km (768 x 768) South Polar Stereographic",
            "grid_label": "8km",
            "nominal_resolution": "8km"
        })
        return ds

    def _make_tmp_file(self, infile: Path) -> Path:
        """Makes a temporary intermediate netCDF file from the raw CMIP data. Returns
        the path to the temporary file."""

        tmp_filepath = get_tmp_filepath(infile)
        self.tmp_files.append(tmp_filepath)  # Track for cleanup
        if tmp_filepath.exists():
            print(f"Using existing temporary file: {tmp_filepath.name}")
            return tmp_filepath
        
        ds = xr.open_dataset(infile)
        # If the longitude range is less than 360 degrees, add a cyclic point
        lonDim = ds.lon.dims[0]
        lonRange = ds.lon[-1].values - ds.lon[0].values
        if np.abs(lonRange - 360.0) > 1e-10:
            ds = add_cyclic_point(ds, lonDim)

        # If the latitude does not extend to the pole, plug the pole hole
        latDim = ds.lat.dims[0]
        minlat = ds[latDim].min().values
        if minlat > -90.0:
            ds = plug_pole_hole(ds, latDim)

        # If requested, compute annual means
        if self.calc_annual_mean:
            ds = annual_means(ds)
        
        ds.to_netcdf(tmp_filepath)
        return tmp_filepath

    def _regrid(self, file: Path) -> Dataset:
        """Regrid a lat-lon file and return a dataset on a 768x768 8km South polar 
        stereographic grid with the origin centered over the South pole."""

        remapper = self._build_remapper(file, BISICLES_FILE)
        dsIn = xr.open_dataset(file)
        # Fill NaNs in mrro before regridding
        if "mrro" in dsIn.data_vars:
            dsIn["mrro"] = dsIn["mrro"].fillna(0)
        dsOut = remapper.remap_numpy(dsIn, renormalization_threshold=0.1)
        return dsOut
 
    def _build_remapper(self, src_grid: Path, dst_grid: Path) -> Remapper:
        """Builds and returns a pyremap Remapper object given a source lat-lon grid and destination 
        South polar stereographic grid."""

        # Create the source and destination grid descriptors
        inDescriptor = LatLonGridDescriptor.read(
            str(src_grid),
            lat_var_name='lat',
            lon_var_name='lon'
            )
        inDescriptor.regional = True
        outDescriptor = get_polar_descriptor_from_file(
            str(dst_grid),
            projection='antarctic'
            )
        
        # Create the remapper
        mapping = get_mapping_filepath(src_grid)
        remapper = Remapper(
            map_filename=str(mapping),
            src_descriptor=inDescriptor,
            dst_descriptor=outDescriptor,
            method=self.method
            )
        if not mapping.exists():
            mapping.parent.mkdir(parents=True, exist_ok=True)
            remapper.build_map()
        return remapper

    def save(self, ds: Dataset, filepath: Path) -> None:
        """Saves a dataset to a specified filepath with encoding specifications."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        for var in ds.data_vars:
            ds[var].encoding.update(self.encoding_specs)
        ds.to_netcdf(filepath)

    def clean_up(self):
        """Deletes temporary and log files created during regridding."""
        # Clean up all tracked temporary files
        for tmp_file in self.tmp_files:
            if tmp_file.exists():
                tmp_file.unlink()
        
        # Clean up ESMF log file
        log = Path.cwd() / "PET0.RegridWeightGen.Log"
        if log.exists():
            log.unlink()

# Helper functions
def get_mapping_filepath(infile: Path) -> Path:
    """Generates filepath for the pyremap mapping file based on the dataset attributes."""

    mapping_dir = DATA_HOME / "regridding"
    with xr.open_dataset(infile) as ds:
        model = ds.attrs['source_id']
        var = ds.attrs['variable_id']
    return mapping_dir / f"map_{model}_{var}_to_bisicles.nc"

def get_tmp_filepath(infile: Path) -> Path:
    """Generates a temporary file path in the same directory as the input for
    intermediate data storage."""

    infile = infile.resolve()
    tmp_filename = "{}_{}".format(infile.stem, "tmp.nc")
    return infile.parent / tmp_filename

def add_cyclic_point(ds: Dataset, lonDim: str = 'lon') -> Dataset:
    """Add a cyclic point to a dataset along the specified longitude dimension"""

    nLon = ds.sizes[lonDim]
    lonIndices = xr.DataArray(np.append(np.arange(nLon), [0]), dims=('newLon',))
    ds.load()
    ds = ds.isel({lonDim: lonIndices})
    ds = ds.swap_dims({'newLon': lonDim})
    return ds

def plug_pole_hole(ds: Dataset, latDim: str = 'lat') -> Dataset:
    """Fill in the pole hole in a lat-lon dataset by copying data from the most
    Southerly row."""

    minlat = ds[latDim].min().values
    southmost = ds.sel({latDim: minlat}) # Southmost row of data
    pole = southmost.expand_dims({latDim: [-90.0]}) # copy data to pole
    ds_with_pole = xr.concat([pole, ds], dim=latDim)
    ds_with_pole = ds_with_pole.sortby(latDim)
    ds_with_pole[latDim].attrs = ds[latDim].attrs # copy attributes for pyremap
    return ds_with_pole

def annual_means(ds: Dataset) -> Dataset:
    """Calculate annual means from a monthly dataset with dimension 'time'."""

    annual = ds.groupby('time.year').mean('time')
    annual = annual.rename({'year': 'time'}) # keep time dimension name consistent
    table_id = annual.attrs.get('table_id', "")
    annual.attrs.update({
        'frequency': 'yr',
        'table_id': table_id.replace('mon', 'yr')
        })
    for var in annual.data_vars:
        annual[var].attrs.update({'frequency': 'yr'})
        annual[var].attrs.pop('mipTable', None)
        annual[var].attrs.pop('prov', None)
    return annual

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path, help="path to CMIP file for regridding")
    parser.add_argument("outfile", type=Path, help="path to output file on BISICLES grid")
    parser.add_argument("--method", type=str, default="bilinear", 
                        help="regridding method to use (default: bilinear)")
    parser.add_argument("--annual", action="store_true", 
                        help="compute annual means before regridding")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.outfile.exists():
        print(f"{args.outfile} already exists.")
        return
    with Regridder(method=args.method, annual_mean=args.annual) as rg:
        ds = rg.regrid_CMIP_to_bisicles(args.infile)
        rg.save(ds, args.outfile)

if __name__ == "__main__":
    main()