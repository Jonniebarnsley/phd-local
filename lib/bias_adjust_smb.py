import argparse
import numpy as np
import xarray as xr
from pathlib import Path

# MAR SMB used in the inversion
MAR = xr.open_dataset('/Users/jonniebarnsley/data/MAR/antarctica.mon-SMB-MAR_ERA5-1980-2021.mean.smb.nc')
ZWALLY = xr.open_dataset('/Users/jonniebarnsley/data/masks/zwally_basins_8km.nc')
SMITH2020 = xr.open_dataset('/Users/jonniebarnsley/data/ICESat2/dhdt/ais_dhdt_grounded_filt_bisicles_8km.nc')

def bias_adjust_smb(background_smb, dhdt_mod, dhdt_obs):
    """Bias adjust a background SMB field based on the misfit between modelled and observed dhdt."""

    basin_nums = np.unique(ZWALLY.basins.values)
    misfit = np.zeros(ZWALLY.basins.shape)
    for basin in basin_nums:
        if basin == 0: # ice shelves / open ocean
            continue
        mask = ZWALLY.basins == basin
        dhdt_mod_basin = dhdt_mod.where(mask)
        dhdt_obs_basin = dhdt_obs.where(mask)
        
        mod_mean = dhdt_mod_basin.mean().item()
        obs_mean = dhdt_obs_basin.mean().item()
        delta = obs_mean - mod_mean
        misfit = np.where(mask, delta, misfit)

    bias_adjusted = background_smb + misfit

    return bias_adjusted

def build_parser():
    parser = argparse.ArgumentParser(
        description="Bias adjust MAR SMB data to match observed ice discharge."
    )
    parser.add_argument(
        "inversion_nc",
        type=Path,
        help="Path to inversion NetCDF file",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to save the bias-adjusted SMB NetCDF file.",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    inversion_ds = xr.open_dataset(args.inversion_nc)

    mar_smb = MAR.SMB
    dhdt_mod = inversion_ds.dThicknessdt.isel(time=-1)
    dhdt_obs = SMITH2020.dhdt_obs

    print("Performing bias adjustment using {}".format(args.inversion_nc))
    bias_adjusted_smb = bias_adjust_smb(mar_smb, dhdt_mod, dhdt_obs)

    output_ds = xr.Dataset(
        {"smb_bias_adjusted": bias_adjusted_smb})

    output_ds.to_netcdf(args.output_file)
    print(f"Bias-adjusted SMB saved to {args.output_file}")

if __name__ == "__main__":
    main()