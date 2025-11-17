#!/bin/python

# Regrids CMIP data from a lat-lon grid to the BISICLES 8km (768 x 768) South Polar
# stereographic grid using pyremap.

# Usage: python regrid_CMIP_to_bisicles.py <input_file> <output_file> [--method METHOD] [--annual]

# --- Options ---
# --method : Regridding method to use. bilinear or conserve (default: bilinear)
# --annual : If set, compute annual means before regridding

# author: Jonnie
# date: November 2025

import os
import argparse
import numpy as np
import xarray as xr
from xarray import Dataset
from pathlib import Path
from pyremap import LatLonGridDescriptor, Remapper
from pyremap.polar import get_polar_descriptor_from_file
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

# local imports
from bisicles_defaults import GRID_8KM

# ignore warnings of multiple fill values when reading netcdfs
import warnings
from xarray import SerializationWarning
warnings.filterwarnings("ignore", category=SerializationWarning)


DATA_HOME = Path(os.environ.get('DATA_HOME', '~/data/')) # mapping files get saved here
BISICLES_FILE = "/Users/jonniebarnsley/code/phd/local/data/bisicles_grid_pole_centered.nc"

class Regridder:
    def __init__(self, infile: Path, outfile: Path, method: str = "bilinear", annual_mean: bool = False):
        self.infile = infile
        self.outfile = outfile
        self.tmp_file = self.infile.resolve().parent / "{}_{}".format(self.infile.stem, "tmp.nc")
        self.method = method
        self.calc_annual_mean = annual_mean
        self.encoding_specs = {
            'zlib': True,
            'complevel': 4,
            '_FillValue': -9999,
        }

    def regrid_CMIP_to_bisicles(self) -> None:
        """Regrids any 2-D CMIP data from a lat-lon grid onto a BISICLES-ready South Polar Steregraphic grid."""
        if self.outfile.exists():
            print("{} already exists.".format(self.outfile))
            return
        # The easiest way to use pyremap is to have twp netcdfs that are on the source
        # and destination grids. However, some CMIP data requires adding a cyclic point,
        # which changes the native grid. We therefore need to create an intermediate file 
        # with a cyclic point added if necessary for pyremap to work from.
        self._make_intermediate_file()

        # Now use pyremap to regrid onto a BISICLES 768x768 8km grid.
        mapping_dir = DATA_HOME / "regridding"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        model = self._get_model_name()
        mapping = mapping_dir / f"{model}_to_bisicles_map.nc"
        self._regrid(mapping)

    def clean_up(self):
        """Deletes temporary files created during regridding."""
        if self.tmp_file.exists():
            self.tmp_file.unlink()
        cwd = Path(os.getcwd())
        log = cwd / "PET0.RegridWeightGen.Log"
        if log.exists():
            os.remove(log)

    def _regrid(self, mapping_file: Path) -> None:
        """Generic regridding function to regrid from any lat-lon grid to the BISICLES grid."""
        # Create the source and destination grid descriptors
        inDescriptor = LatLonGridDescriptor.read(
            str(self.tmp_file),
            lat_var_name='lat',
            lon_var_name='lon'
            )
        inDescriptor.regional = True
        outDescriptor = get_polar_descriptor_from_file(
            str(BISICLES_FILE),
            projection='antarctic'
            )
        
        # Create the remapper
        remapper = Remapper(
            map_filename=str(mapping_file),
            src_descriptor=inDescriptor,
            dst_descriptor=outDescriptor,
            method=self.method
            )
        if not mapping_file.exists():
            remapper.build_map()

        # Open the source dataset and regrid
        tmp = xr.open_dataset(self.tmp_file)
        tmp = tmp.fillna(0) # Fill NaNs in evspsbl and mrro with zeros

        timeslices = []
        ntime = tmp.sizes['time']
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), 
                      MofNCompleteColumn(), TimeElapsedColumn()) as progress:
            task = progress.add_task(f"[cyan]{self.outfile.name}", total=ntime)
            for tIndex in range(ntime):
                dsIn = tmp.isel(time=tIndex)
                dsOut = remapper.remap_numpy(dsIn, renormalization_threshold=0.1)
                timeslices.append(dsOut)
                progress.update(task, advance=1)
        ds = xr.concat(timeslices, dim='time', coords='minimal', join='override')
        
        # N.B. the BISICLES grid has to be pole-centered to be in South Polar Stereographic coordinates 
        # for pyremap, so we have to revert back to the standard BISICLES grid with (0,0) in the
        # bottom-left corner.
        xs, ys = GRID_8KM
        ds = ds.assign_coords(x=xs, y=ys)
        for var in ds.data_vars:
            ds[var].encoding.update(self.encoding_specs)
        ds.to_netcdf(self.outfile)

    def _make_intermediate_file(self) -> None:
        """Makes a temporary intermediate netCDF file from the raw CMIP data"""
        if self.tmp_file.exists():
            return
        ds = xr.open_dataset(self.infile)
        lonDim = ds.lon.dims[0]
        lonRange = ds.lon[-1].values - ds.lon[0].values
        # If the longitude range is less than 360 degrees, add a cyclic point
        if np.abs(lonRange - 360.0) > 1e-10:
            ds = add_cyclic_point(ds, lonDim)

        # Compute annual means to reduce file size
        if self.calc_annual_mean:
            ds = annual_means(ds)
        ds.to_netcdf(self.tmp_file)

    def _get_model_name(self) -> str:
        """Extracts the model name from the infile path"""
        parts = self.infile.stem.split('_')
        model_name = parts[2]
        return model_name

def add_cyclic_point(ds: Dataset, lonDim: str) -> Dataset:
    """Add a cyclic point to a dataset along the specified longitude dimension"""
    nLon = ds.sizes[lonDim]
    lonIndices = xr.DataArray(np.append(np.arange(nLon), [0]),
                                  dims=('newLon',))
    ds.load()
    ds = ds.isel({lonDim: lonIndices})
    ds = ds.swap_dims({'newLon': lonDim})
    return ds

def annual_means(ds: Dataset) -> Dataset:
    """Calculate annual means from a dataset with a time dimension"""
    annual = ds.groupby('time.year').mean('time')
    annual = annual.rename({'year': 'time'})
    return annual

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path, help="path to CMIP file for regridding")
    parser.add_argument("outfile", type=Path, help="path to output file on BISICLES grid")
    parser.add_argument("--method", type=str, default="bilinear", help="regridding method to use (default: bilinear)")
    parser.add_argument("--annual", action="store_true", help="compute annual means before regridding")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    rg = Regridder(args.infile, args.outfile, method=args.method, annual_mean=args.annual)
    try:
        rg.regrid_CMIP_to_bisicles()
    finally:
        rg.clean_up()

if __name__ == "__main__":
    main()