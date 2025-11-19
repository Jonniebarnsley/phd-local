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
BISICLES_FILE = "/Users/jonniebarnsley/code/phd/local/data/bisicles_grid_pole_centered.nc"

class Regridder:
    """
    Regridder class to handle regridding of CMIP data onto the BISICLES grid.
    
    Regridding requires a number of steps:
    
     1. Create a temporary file with a cyclic point added.
     2. Regrid temporary file onto a South polar stereographic grid.
     3. Relabel the coordinates to match the BISICLES setup.
     4. Save the regridded dataset to a specified output file.
     5. Clean up the temporary and log files created during the process.

    THe regridder class handles all the above whilst saving useful attributes for use across
    functions - for example, information about the dataset such as model or scenario.
    """

    def __init__(self, infile: Path, method: str = "bilinear", annual_mean: bool = False):
        self.infile = infile
        self.method = method
        self.calc_annual_mean = annual_mean
        self.encoding_specs = {
            'zlib': True,
            'complevel': 4,
            '_FillValue': -9999,
        }
        self._setup()

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - clean up temporary / log files"""
        self.clean_up()

    def _setup(self) -> None:
        """Initialize regridder by loading metadata and setting up file paths."""
        if not self.infile.exists():
            raise FileNotFoundError(f"Input file not found: {self.infile}")
        
        # Load dataset attributes
        with xr.open_dataset(self.infile) as ds:
            self.attrs = ds.attrs
        
        # Validate required attributes
        required_attrs = ['source_id', 'variable_id']
        missing = [attr for attr in required_attrs if attr not in self.attrs]
        if missing:
            raise ValueError(f"Input file missing required attributes: {missing}")

        # Get filepaths for both the pyremap mapping file and intermediate tmp file
        self.mapping = self._get_mapping_filepath()
        self.tmp_file = self._get_tmp_filepath()

    def regrid_CMIP_to_bisicles(self) -> Dataset:
        """Regrids any 2-D CMIP data from a lat-lon grid and return a dataset on the 
        BISICLES 8km-resolution 768x768 South Polar Steregraphic grid"""
        # The easiest way to use pyremap is to have twp netcdfs that are on the source
        # and destination grids. However, some CMIP data requires adding a cyclic point,
        # which changes the native grid. We therefore need to create an intermediate file 
        # with a cyclic point added (if necessary) for pyremap to work from.
        self._make_tmp_file()
        if not self.tmp_file.exists():
            raise FileNotFoundError(f"Temporary file {self.tmp_file} not found.")

        # Now regrid onto a 768x768 8km grid with the origin on the South pole.
        ds = self._regrid(self.tmp_file)
        
        # Finally, redefine grid coordinates such that the origin is in the bottom left corner.
        xs, ys = GRID_8KM
        ds = ds.assign_coords(x=xs, y=ys)
        return ds

    def _make_tmp_file(self) -> None:
        """Makes a temporary intermediate netCDF file from the raw CMIP data"""
        if self.tmp_file.exists():
            print(f"Using existing temporary file: {self.tmp_file.name}")
            return
        ds = xr.open_dataset(self.infile)

        # If the longitude range is less than 360 degrees, add a cyclic point
        lonDim = ds.lon.dims[0]
        lonRange = ds.lon[-1].values - ds.lon[0].values
        if np.abs(lonRange - 360.0) > 1e-10:
            ds = add_cyclic_point(ds, lonDim)

        # If requested, compute annual means
        if self.calc_annual_mean:
            ds = annual_means(ds)
        
        ds.to_netcdf(self.tmp_file)

    def _regrid(self, file: Path) -> Dataset:
        """Regrid a lat-lon file and return a dataset on a 768x768 8km South polar 
        stereographic grid with the origin centered over the South pole."""
        remapper = self._build_remapper(file, BISICLES_FILE)
        dsIn = xr.open_dataset(file)
        dsIn = dsIn.fillna(0) # Fill NaNs before regridding
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
        remapper = Remapper(
            map_filename=str(self.mapping),
            src_descriptor=inDescriptor,
            dst_descriptor=outDescriptor,
            method=self.method
            )
        if not self.mapping.exists():
            remapper.build_map()
        return remapper
    
    def _get_mapping_filepath(self) -> Path:
        """Generates filepath for the pyremap mapping file based on the dataset attributes."""
        # Prepare directory for mapping files
        mapping_dir = DATA_HOME / "regridding"
        mapping_dir.mkdir(parents=True, exist_ok=True)

        model = self.attrs['source_id']
        var = self.attrs['variable_id']
        return mapping_dir / f"map_{model}_{var}_to_bisicles.nc"
    
    def _get_tmp_filepath(self) -> Path:
        """Generates a temporary file path in the same directory as the input for
        intermediate data storage."""
        infile = self.infile.resolve()
        tmp_filename = "{}_{}".format(infile.stem, "tmp.nc")
        return infile.parent / tmp_filename

    def save(self, ds: Dataset, filepath: Path) -> None:
        """Saves a dataset to a specified filepath with encoding specifications."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        for var in ds.data_vars:
            ds[var].encoding.update(self.encoding_specs)
        ds.to_netcdf(filepath)

    def clean_up(self):
        """Deletes temporary files created during regridding."""
        if self.tmp_file.exists():
            self.tmp_file.unlink()
        cwd = Path.cwd()
        log = cwd / "PET0.RegridWeightGen.Log"
        if log.exists():
            log.unlink()

def add_cyclic_point(ds: Dataset, lonDim: str) -> Dataset:
    """Add a cyclic point to a dataset along the specified longitude dimension"""
    nLon = ds.sizes[lonDim]
    lonIndices = xr.DataArray(np.append(np.arange(nLon), [0]), dims=('newLon',))
    ds.load()
    ds = ds.isel({lonDim: lonIndices})
    ds = ds.swap_dims({'newLon': lonDim})
    return ds

def annual_means(ds: Dataset) -> Dataset:
    """Calculate annual means from a dataset with a time dimension"""
    annual = ds.groupby('time.year').mean('time')
    annual = annual.rename({'year': 'time'}) # keep time dimension name consistent
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
    with Regridder(args.infile, method=args.method, annual_mean=args.annual) as rg:
        ds = rg.regrid_CMIP_to_bisicles()
        rg.save(ds, args.outfile)

if __name__ == "__main__":
    main()