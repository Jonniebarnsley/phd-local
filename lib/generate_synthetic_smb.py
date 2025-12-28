#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MAR SMB is too wet over East Antarctica and the peninsula, leading to erroneous mass gain in
# these regions under a control simulation. To fix this, we generate a synthetic SMB field such
# that dHdt at t=0 matches that from observations.

# To do this, we note that:
# dHdt = SMB + BMB - div.(uH)                 (1)

# For grounded ice, we assume BMB = 0, so (1) can be rearranged to give:
# SMB = dHdt + div.(uH)                       (2)        

# Note that we have both:
# SMB_mod = dHdt_mod + div.(uH)_mod           (3)
# SMB_obs = dHdt_obs + div.(uH)_obs           (4)

# but dHdt_mod != dHdt_obs when using MAR SMB. Define delta as the model-observation misfit:
# delta = dHdt_mod - dHdt_obs                 (5)

# Combining (3) and (5) gives:
# SMB_mod - delta = dHdt_obs + div.(uH)_mod   (6)

# Therefore define SMB_synthetic = SMB_mod - delta. Then (6) becomes:
# SMB_synthetic = dHdt_obs + div.(uH)_mod     (7)

# Comparing (7) with (2), we see that SMB_synthetic is the SMB field that, when used in the model,
# will give dHdt close to observations at t=0 and for grounded ice.

# For floating ice (BMB != 0), we simply use MAR SMB.

import argparse
import numpy as np
import xarray as xr

from pathlib import Path
from xarray import DataArray, Dataset
from scipy.ndimage import gaussian_filter

from bisicles_defaults import GRID_8KM

MAR = "/Users/jonniebarnsley/data/MAR/antarctica.mon-SMB-MAR_ERA5-1980-2021.mean.smb.nc"
DHDT_OBS = "/Users/jonniebarnsley/data/ICESat2/dhdt/ais_dhdt_grounded_filt_bisicles_8km.nc"

MAX_SMB = 2.0  # m/a
MIN_SMB = -1.0  # m/a

def load_MAR_smb():
    """Load surface mass balance (1980-2021 mean) from the MAR regional climate 
    model (Agosta et al., 2019)"""
    ds = xr.open_dataset(MAR)
    smb = ds.SMB.values
    return smb

def load_obs_dHdt():
    """Load rates of surface elevation change from ICESat-2 observations (Smith et al., 2020)"""
    ds = xr.open_dataset(DHDT_OBS)
    dhdt = ds.dhdt_obs.values
    return dhdt

def calculate_flux_divergence(ds):
    """Calculate flux divergence div.(uH) from ice velocities and thickness."""
    uh = ds.xVel * ds.thickness
    vh = ds.yVel * ds.thickness
    uh = uh.values # convert to numpy array
    vh = vh.values 
    
    x = ds.x.values
    dx = x[1] - x[0]

    n, m = ds.thickness.shape
    divuh = np.zeros((n, m))

    divuh[1:n-1, 1:m-1] = 0.5 / dx * (
        (uh[1:n-1, 2:m] - uh[1:n-1, 0:m-2]) +
        (vh[2:n, 1:m-1] - vh[0:n-2, 1:m-1])
    )
    return divuh

def get_grounded_mask(ds, ice_density=917, ocean_density=1027):
    """Calculate mask for grounded ice based on hydrostatic equilibrium."""
    thk = ds.thickness.values
    bed = ds.Z_base.values
    floatation_thickness = -bed * ocean_density / ice_density
    grounded_mask = thk > floatation_thickness
    return grounded_mask

def smooth_smb(smb, max_smb=MAX_SMB, min_smb=MIN_SMB, sigma=4):
    """Apply limits and Gaussian smoothing to SMB field."""
    # synthetic smb is very noisy because divuh_model is very noisy - would take many centuries for
    # model to smooth it out. So instead apply limits and a Gaussian filter to artificially smooth 
    # it, avoiding any weird artifacts arising from noisy smb fields.
    smb = np.where(smb < min_smb, min_smb, smb) 
    smb = np.where(smb > max_smb, max_smb, smb) 
    smb = gaussian_filter(smb, sigma=sigma)
    return smb

def generate_synthetic_smb(input: Path, output: Path, overwrite: bool = False):
    """
    Generate synthetic SMB field such that initial dHdt matches observations on grounded ice.
    """
    # Get flux divergence from last timestep of the relaxation
    bisicles_ds = xr.open_dataset(input)
    last_timestep = bisicles_ds.isel(time=-1)
    divuh = calculate_flux_divergence(last_timestep)

    obs_dhdt = load_obs_dHdt()
    mar = load_MAR_smb()
    grounded_mask = get_grounded_mask(last_timestep)

    # Synthetic SMB is:
    # - grounded: divuh_model + dhdt_obs
    # - floating: MAR
    smb = np.where(grounded_mask, divuh + obs_dhdt, mar)
    smb = np.nan_to_num(smb, nan=0.0)
    smb = smooth_smb(smb, sigma=4)

    # Save to output file
    xs, ys = GRID_8KM
    da = DataArray(
        smb,
        coords={'x': xs, 'y': ys},
        dims=['y', 'x']
    )
    ds = Dataset({'smb_synthetic': da})
    if output.exists() and overwrite:
        output.unlink()
    ds.to_netcdf(output)
    print(f"Synthetic SMB saved to {output}")

def build_parser():
    parser = argparse.ArgumentParser(description="Generate synthetic SMB data.")
    parser.add_argument("input", type=Path, help="Input Bisicles data file path.")
    parser.add_argument("output", type=Path, help="Output synthetic SMB file path.")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite output file if it exists.")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    generate_synthetic_smb(args.input, args.output, args.overwrite)

if __name__ == "__main__":
    main()