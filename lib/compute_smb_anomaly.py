"""
Takes surface mass balance data created by preprocess_smb.py and computes the SMB anomaly
relative to the 1995-2014 climatology. Saves the SMB anomaly to netCDF files, one per year.
Units are converted from kg m-2 s-1 to m a-1 ice equivalent, ready for input to BISICLES.

Usage: python compute_smb_anomaly.py <historical_file> <scenario_file> <anomaly_dir>

Where:
    historical_file : path to netCDF file containing historical SMB data
    scenario_file   : path to netCDF file containing scenario SMB data
    anomaly_dir     : output directory to save SMB anomaly netCDF files

author: Jonnie
date: November 2025
"""

import argparse
import xarray as xr
from xarray import DataArray, Dataset
from pathlib import Path

# Local imports
from bisicles_defaults import ICE_DENSITY

# Compute SMB anomaly relative to climatology over these years
CLIM_START = 1995
CLIM_END = 2014

# Trim dataset to these years (must include climatology period)
DS_START = 1995
DS_END = 2300

def make_smb_anomaly(historical: Path, scenario: Path, anomaly_dir: Path) -> None:
    """Computes SMB anomaly from historical and scenario SMB data files and saves to netCDF files."""
    
    smb, attrs = _load_smb(historical, scenario)
    smb = smb.sel(time=slice(DS_START, DS_END)) # cut to desired years
    smb *= (60*60*24*365) / ICE_DENSITY  # convert from kg m-2 s-1 to m a-1

    # Compute climatology and anomaly
    clim = climatology(smb, start_year=1995, end_year=2014)
    smb_anomaly = smb - clim

    # Add variable attributes
    smb_anomaly.attrs.update({
        'long_name'     : "Surface Mass Balance Anomaly",
        'standard_name' : "smb_anomaly",
        'description'   : "Surface Mass Balance anomaly relative to 1995-2014 climatology",
        'units'         : "m a-1"
    })
    smb_anomaly.encoding.update({
        'zlib': True,
        'complevel': 4,
        'fill_value': -9999,
    })
    
    # Save to netCDF files - one for each year
    filestem = "smb_anomaly_{}_{}_{}".format(
        attrs['source_id'],
        attrs['experiment_id'],
        attrs['grid_label'],
    )
    outdir = anomaly_dir / attrs['experiment_id'] / attrs['source_id']
    outdir.mkdir(parents=True, exist_ok=True)
    save_smb_by_year(smb_anomaly, filestem, outdir)

def save_smb_by_year(smb_anomaly: DataArray, filestem: str, outdir: Path) -> None:
    """Saves SMB anomaly DataArray to netCDF files, one per year."""
    smb_anomaly = smb_anomaly.load()
    for year in smb_anomaly.time.values:
        smb_anomaly_file = outdir / f"{filestem}_{year}.nc"
        if smb_anomaly_file.exists():
            print(f"    {smb_anomaly_file} already exists.")
            continue
            
        smb_anomaly_yr = smb_anomaly.sel(time=year)
        ds = Dataset(
            {"smb_anomaly": smb_anomaly_yr},
            coords=smb_anomaly_yr.coords,
            attrs=smb_anomaly.attrs # copy attributes from original dataset
        )

        # Save the anomaly data
        print(smb_anomaly_file)
        try:
            ds.to_netcdf(smb_anomaly_file)
        except KeyboardInterrupt:
            if smb_anomaly_file.exists():
                smb_anomaly_file.unlink()  # Remove incomplete file
            print("Process interrupted. Incomplete file removed.")
            return

def _load_smb(historical: Path, scenario: Path) -> tuple[DataArray, dict]:
    """Loads historical and scenario SMB data from the specified files."""
    with xr.open_dataset(historical) as ds:
        hist_smb = ds.smb
    with xr.open_dataset(scenario) as ds:
        scen_smb = ds.smb
        attrs = ds.attrs
    total_smb = xr.concat([hist_smb, scen_smb], dim='time')
    return total_smb, attrs

def climatology(da: DataArray, start_year: int, end_year: int) -> DataArray:
    """Computes the climatology of SMB over the specified year range."""
    return da.sel(time=slice(start_year, end_year)).mean('time')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("historical_file", type=Path, help="path to historical SMB file")
    parser.add_argument("scenario_file", type=Path, help="path to scenario SMB file")
    parser.add_argument("anomaly_dir", type=Path, help="path to output directory for SMB anomalies")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.anomaly_dir.mkdir(parents=True, exist_ok=True)
    make_smb_anomaly(args.historical_file, args.scenario_file, args.anomaly_dir)

if __name__ == "__main__":
    main()