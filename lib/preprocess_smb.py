#!/bin/python

"""
Calculates Antarctic SMB from raw CMIP data files. First calculates annual means and regrids
the data onto the BISICLES 8km 768x768 South Polar Stereographic grid. Then computes SMB using
the ISMIP6 formula:

SMB = pr - evspsbl - mrro

Finally, it converts units from kg m^-2 s^-1 to m yr^-1 ice equivalent.

It assumes a specific data directory structure for the input CMIP data, as follows:
INPUT_DIR/
├── Amon/
│   ├── tas/
│   └── pr/
|   └── evspsbl/
├── Lmon/
|   └── mrro/
And then within each variable directory:
<variable>/gn/<version>/<files>.nc

Where gn indicates a native grid. It saves all regridded data to equivalent paths but with 
grid=bisicles-8km instead of gn and creates a new directory for smb under Lmon.

author: Jonnie
date: November 2025
"""

import argparse
import xarray as xr
from pathlib import Path

from regrid_CMIP_to_bisicles import Regridder
from bisicles_defaults import ICE_DENSITY

VARIABLES = ['tas', 'pr', 'evspsbl', 'mrro']
TABLE = {
    'tas': 'Ayr',
    'pr': 'Ayr',
    'evspsbl': 'Ayr',
    'mrro': 'Lyr',
}

class SMBProcessor:
    def __init__(self, input_dir: Path, version: str = None):
        self.input_dir = input_dir
        self.attrs = None
        if version:
            self.version = version
        else:
            self.version = self._check_version()
        print(f"Processing data from version: {self.version}")

    def regrid_all_data(self) -> None:
        """Regrids all CMIP data in the specified directory to the BISICLES 8km grid."""
        # Queue up all files for regridding
        files = []
        for var in VARIABLES:
            files += sorted(self.input_dir.glob(f"*/{var}/gn/*/*.nc"))
        
        # Check we found some files
        if len(files) == 0:
            raise FileNotFoundError(
                "No data files found in input directory, or perhaps the directory structure is not as "+\
                "expected. \nExpect a structure of <input_dir>/<table>/<variable>/gn/<version>/<files>.nc")
        
        # Regrid one at a time
        print("Regridding {} files...".format(len(files)))
        for infile in files:
            outfile = self._get_regridded_path(infile)
            outfile.parent.mkdir(parents=True, exist_ok=True)
            rg = Regridder(infile, outfile, method='conserve', annual_mean=True)
            rg.regrid_CMIP_to_bisicles()
            rg.clean_up()

    def compute_smb(self) -> None:
        """Computes SMB from regridded CMIP data and saves to new netcdf files."""
        
        # load the necessary climate data along with global attributes to copy
        pr, evspsbl, mrro, orig_attrs = self._load_input_data()

        smb = pr - evspsbl - mrro.fillna(0)
        smb_m_per_yr = smb / ICE_DENSITY * 31536000  # seconds in a year (no leap)

        # Choose which attributes to keep from the original data
        keep_attrs = ['activity_id', 'experiment_id', 'frequency', 'institution', 'institution_id', \
                      'mip_era', 'source_id', 'table_id', 'variant_info', 'variant_label']
        attrs = {key: value for key, value in orig_attrs.items() if key in keep_attrs}
        attrs.update({
            'grid': 'BISICLES 8km South Polar Stereographic',
            'grid_label': 'bisicles-8km',
        })

        # Create new dataset and apply attributes / encoding options
        ds = xr.Dataset(
            data_vars={'smb': smb_m_per_yr},
            coords=pr.coords,
            attrs=attrs
            )
        ds['smb'].attrs.update({
                'long_name': 'Surface Mass Balance',
                'description': 'Surface Mass Balance computed as pr - evspsbl - mrro',
                'units': 'm yr^-1 ice equivalent',
            })
        ds['smb'].encoding.update({
            'zlib': True,
            'complevel': 4,
            })
        
        # Generate output path and save
        model = attrs['source_id']
        experiment = attrs['experiment_id']
        variant = attrs['variant_label']
        start = ds.time.values[0]
        end = ds.time.values[-1]
        smb_filename = f"smb_Lyr_{model}_{experiment}_{variant}_bisicles-8km_{start}-{end}.nc"

        smb_dir = self.input_dir / 'Lyr' / 'smb' / 'bisicles-8km' / self.version
        smb_dir.mkdir(parents=True, exist_ok=True)
        
        outfilepath = smb_dir / smb_filename
        print("Saving SMB data to {}".format(outfilepath))
        ds.to_netcdf(outfilepath)

    def _get_regridded_path(self, file: Path):
        """Generates the output path for a regridded file based on the input file path."""
        version = file.parent.name
        try:
            var, table, model, experiment, ripf, grid, dates = file.stem.split('_')
        except ValueError:
            print(f"Unexpected filename format: {file.stem}")
            raise
        yr_table = table.replace("mon", "yr")  # monthly to yearly table since we are taking annual means
        new_stem = "_".join([var, yr_table, model, experiment, ripf, 'bisicles-8km', dates])
        regridded_path = f"{self.input_dir}/{yr_table}/{var}/bisicles-8km/{version}/{new_stem}.nc"
        return Path(regridded_path)  
    
    def _check_version(self) -> str:
        """Searches through all version subdirectories. If multiple or no versions are found, raises an error."""
        versions = {d.name for d in self.input_dir.glob('*/*/*/*') if d.is_dir()}
        if len(versions) == 0:
            raise ValueError(
                "No data found in input directory, or perhaps the directory structure is not as expected.\n"+\
                    "Expect a structure of <input_dir>/<table>/<variable>/<grid>/<version>/<files>.nc"
                    )
        if len(versions) > 1:
            raise ValueError(
                f"Multiple versions detected: {versions}. \nSpecify a single version using --version to avoid confusion.")
        else:
            return versions.pop()

    def _load_input_data(self) -> dict:
        """Opens data for pr, evspsbl, and mrro given an assume data structure."""

        data = {}
        for var in ['pr', 'evspsbl', 'mrro']:
            var_dir = self.input_dir / TABLE[var] / var / 'bisicles-8km' / self.version
            files = sorted(var_dir.glob("*.nc"))
            ds = xr.open_mfdataset(files, combine='by_coords')
            data[var] = ds[var]
        attrs = ds.attrs # global attributes should be shared across variables
        return data['pr'], data['evspsbl'], data['mrro'], attrs

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="path to input directory for processing")
    parser.add_argument("--version", type=str, default=None, help="specific model version to process")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    processor = SMBProcessor(args.directory, version=args.version)
    processor.regrid_all_data()
    processor.compute_smb()
    print("done")

if __name__ == "__main__":
    main()