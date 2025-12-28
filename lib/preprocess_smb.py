#!/bin/python

"""
Calculates Antarctic SMB from raw CMIP data files. First calculates annual means and regrids
the data onto the BISICLES 8km 768x768 South Polar Stereographic grid. Then computes SMB using
the ISMIP6 formula:

SMB = pr - evspsbl - mrro

Usage: python preprocess_smb.py <input_dir1> <input_dir2> ... <output_dir>

Where input directories contain netcdfs with CMIP data for individual variables. This list must
include (at least) pr, evspsbl, and mrro, but may also include others (e.g., tas). The directories
can be in any order, but the output directory must be last.

author: Jonnie
date: November 2025
"""

import argparse
import xarray as xr
from xarray import Dataset
from pathlib import Path
from dask.diagnostics import ProgressBar

# local imports
from regrid_CMIP_to_bisicles import Regridder

def build_smb_dataset(data_dirs: list[Path]) -> Dataset:
    """Regrids all data from the specified directories and combines them into a single
    Dataset. Then computes SMB using the ISMIP6 formula."""

    print("Regridding data...")
    ds_list = [regrid_and_concat(dd) for dd in data_dirs]
    ds = xr.merge(ds_list, compat="equals")
    del ds.attrs["variable_id"]  # Remove variable_id from combined data attributes

    print("Calculating SMB...")
    ds['smb'] = ds['pr'] - ds['evspsbl'] - ds['mrro']
    ds['smb'].attrs = {
        "comment": "Calculated as smb = pr - evspsbl - mrro",
        "description": "Ice mass balance at the surface of the ice sheet.",
        "long_name": "Surface Mass Balance",
        "standard_name": "surface_mass_balance",
        "units": "kg m-2 s-1",
        "variable_id": "smb",
    }
    return ds

def regrid_and_concat(data_dir: Path) -> Dataset:
    """Regrids all CMIP data files in the specified directory and then concatenates them
    along the time dimension into a single Dataset."""

    ds_list = []
    with Regridder(method="bilinear", annual_mean=True) as rg:
        for file in sorted(data_dir.glob("*.nc")):
            print("   ", file.stem)
            regridded = rg.regrid_CMIP_to_bisicles(file)
            ds_list.append(regridded)
    ds = xr.concat(ds_list, dim='time')
    ds = ds.drop_vars(['time_bnds', 'time_bounds', 'lat_bnds', 'lon_bnds'], errors='ignore')
    return ds

def validate_variable_id(data_dir: Path) -> str:
    """Validates that all netCDF files in a directory have a single variable_id attribute.
    Returns that variable_id."""

    # Handle case where no netCDF files exist
    files = sorted(data_dir.glob("*.nc"))
    if len(files) == 0:
        raise FileNotFoundError(f"No netCDF files found in {data_dir}")
    
    # Extract variable_id attributes from each file
    variable_ids = set()
    for file in files:
        with xr.open_dataset(file) as ds:
            var = ds.attrs['variable_id']
            variable_ids.add(var)

    # Handle case where multiple variable_ids are found
    if len(variable_ids) > 1:
        raise ValueError(f"Multiple variable_id attributes found in files of {data_dir}: {variable_ids}")
    else:
        return var

def save_smb(ds: Dataset, output_dir: Path) -> None:
    """Saves the SMB Dataset to netCDF files in the specified output directory, organized
    by experiment and model."""

    # Update encoding for compression and fill values
    for var in ds.data_vars:
        ds[var].encoding.update({
            'zlib': True,
            'complevel': 5,
            'fill_value': -9999,
        })

    # Get output path
    experiment = ds.attrs.get('experiment_id', 'unknown_experiment')
    model = ds.attrs.get('source_id', 'unknown_model')
    nc_dir = output_dir / experiment
    nc_dir.mkdir(parents=True, exist_ok=True)

    nc_path = nc_dir / f"smb_{model}_{experiment}_8km.nc"
    print(f"Saving SMB dataset to {nc_path}")
    ds = ds.chunk({'x': 768, 'y': 768, 'time': 1})  # Chunk for efficient writing
    with ProgressBar():
        ds.to_netcdf(nc_path)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess CMIP SMB data by calculating annual means, regridding to "
                    "BISICLES 8km grid, and computing SMB."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input directories")
    parser.add_argument("output", type=Path, help="Output directory")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    print("Validating input directories...")
    # Validate input directories
    for data_dir in args.inputs:
        if not data_dir.is_dir():
            raise NotADirectoryError(f"{data_dir} is not a valid directory.")

    # Vlidate that directories include (at least) pr, evspsbl, mrro
    missing_vars = {'pr', 'evspsbl', 'mrro'}
    for data_dir in args.inputs:
        var = validate_variable_id(data_dir)
        missing_vars.discard(var)
    if len(missing_vars) > 0:
        raise ValueError(f"Missing directories for required variables: {missing_vars}")

    ds = build_smb_dataset(args.inputs)
    save_smb(ds, args.output)

if __name__ == "__main__":
    main()