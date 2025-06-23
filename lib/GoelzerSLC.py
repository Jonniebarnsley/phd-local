import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from xarray import DataArray
import local.lib.utils as utils
from dask.distributed import Client, LocalCluster, default_client

def GoelzerSLC(
        thickness: DataArray, 
        z_base: DataArray,
        rho_ice: int|float = 918.,
        rho_ocean: int|float = 1028.,
        A_ocean: int|float = 3.625e14,
    ) -> DataArray:

    '''
    Calculates sea level contribution from ice sheet thickness and bed
    elevation. Sea level contribution is returned as a grid, with the value
    in each cell corresponding to sea level contribution from that model
    pixel. Methodology for SLC calculation follows Goelzer et al. (2020) 
    https://doi.org/10.5194/tc-14-833-2020

    inputs:
        - thickness: DataArray of ice sheet thickness
        - z_base: DataArray of bed elevation
        - rho_ice: density of ice (kg m^-3)
        - rho_ocean: density of seawater (kg m^-3)
        - A_ocean: surface area of the ocean in m^-2 (Gregory et al., 2019)
    returns: 
        - SLCgrid: A grid of Sea level contribution for each cell in DataArray
    '''

    # ensure that the dataarrays align and have standardised dimension names
    utils.checkAlignment(thickness, z_base)
    for da in (thickness, z_base):
        utils.checkDims(da, {'x', 'y', 'time'})
    
    # fill nans in thickness
    thickness = thickness.fillna(0)

    # get pixel width dx. Pixel area is dx^2 / k^2 where k is the area scale factor
    x = thickness.x
    dx = x[1] - x[0]    # CAUTION: only valid on regularly spaced grids

    # get scale factor. sng=-1 indicates Antarctic (+1 for Arctic)
    k = utils.scaleFactor(thickness, sgn=-1)

    # Apply mask to get only grounded ice
    groundedMask = (thickness > -z_base*rho_ocean/rho_ice) # floatation criteria
    groundedThickness = thickness.where(groundedMask).fillna(0)
    groundedZ_base = z_base.where(groundedMask).fillna(0)

    # Sea level contribution by change in Volume Above Floatation
    V_af = (groundedThickness + np.minimum(groundedZ_base, 0) * rho_ocean/rho_ice) * dx**2 / k**2
    SLC_af = - (V_af - V_af.isel(time=0)) * (rho_ice / rho_ocean) / A_ocean

    # Sea level contribution due to change in Potential Ocean Volume
    V_pov = np.maximum(-z_base, 0) * dx**2 / k**2
    SLC_pov = - (V_pov - V_pov.isel(time=0)) / A_ocean

    # Density correction, this time including floating ice
    rho_water = 1000. # density of freshwater (kg m^-3)
    V_den = thickness * (rho_ice/rho_water - rho_ice/rho_ocean) * dx**2 / k**2
    SLC_den = - (V_den - V_den.isel(time=0)) / A_ocean

    SLCgrid = SLC_af + SLC_pov + SLC_den
    SLCgrid.name = 'slc'

    return SLCgrid


def timeseriesByBasin(da: DataArray, mask: DataArray) -> DataArray:

    '''
    Iterates over basins in a given mask and generates timeseries for variable
    (e.g. sea level contribution) from each basin, which are returned as a DataArray 
    with basin as a dimension.

    inputs:
     - da:      DataArray of some variable with dims (x, y, time)
     - mask:    A basin mask as DataArray. i.e. a grid consisting of integers
                identifying certain basin regions in Antarctica.
    '''

    # ensure dataarray and mask have the appropriate dimensions and alignment
    utils.checkDims(mask, {'x', 'y'})
    utils.checkDims(da, {'x', 'y', 'time'})
    utils.checkAlignment(da.isel(time=0), mask)

    # initialise list of SLC timeseries for each basin
    timeseries = []
    basinIDs = list(map(int, np.unique(mask))) # unique basinID values in mask as int
    for basinID in basinIDs:
        masked = da.where(mask == basinID) # masks out specific basin
        ts = masked.sum(dim=['x', 'y']) # sums gridded data to timeseries
        timeseries.append(ts)

    # concatenate into a new dataarray with 'basin' dimension for output
    outfile = xr.concat(timeseries, dim='basin')
    outfile = outfile.assign_coords(basin=basinIDs)

    return outfile

def getEnsembleSLC(thkPath: Path, zbPath: Path, maskPath: Path=None, parallel: bool=False) -> DataArray:

    '''
    Iterates through directories of thickness and Z_base data for different 
    model runs in order to compile a single dataarray with timeseries of sea 
    level contribution for each ensemble member. Includes optional functionality 
    to mask out a specific basin before calculating sea level contribution.

    inputs:
        - thkPath:  Path object to thickness directory
        - zbPath:   Path object to Z_base directory
        - maskPath (optional):  Path object to mask netcdf
        - parallel (optional):  Boolean to indicate whether to use dask for parallel
                        computation. Default is False.
    output:
        - da:   DataArray of sea level contribution with dimensions time and
                (optionally) basin.
    '''

    # if parallel:
    #     # if no client exists, create a new one
    #     cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    #     client = Client(cluster)

    thkFilepaths = sorted(thkPath.glob('*.nc'))
    zbFilepaths = sorted(zbPath.glob('*.nc'))

    # check that the directories contain the same number of run netcdfs
    if len(thkFilepaths) != len(zbFilepaths):
        raise ValueError('Mismatched number of thickness and z_base files in directories')
    
    # initialise list of SLC timeseries for each run
    timeseries = []
    print('Thickness'.ljust(60), 'Z_base')
    for thkFilepath, zbFilepath in zip(thkFilepaths, zbFilepaths):
        
        print(thkFilepath.name.ljust(60), zbFilepath.name)

        # load datasets, in chunks if parallel computation is selected
        if parallel:
            thkFile = xr.open_dataset(thkFilepath, chunks='auto')
            zbFile = xr.open_dataset(zbFilepath, chunks='auto')
        else:
            thkFile = xr.open_dataset(thkFilepath)
            zbFile = xr.open_dataset(zbFilepath)

        thkFile = utils.forceNamingConvention(thkFile)
        zbFile = utils.forceNamingConvention(zbFile)
        thickness = thkFile.thickness
        Z_base = zbFile.Z_base

        # calculate sea level contribution using Heiko's method
        SLCgrid = GoelzerSLC(thickness, Z_base)

        # Apply mask if selected
        if maskPath is not None:
            with xr.open_dataset(maskPath) as file:
                mask = utils.forceNamingConvention(file).basin
            ts = timeseriesByBasin(SLCgrid, mask)
        else:
            ts = SLCgrid.sum(dim=['x', 'y'])

        timeseries.append(ts.compute())

    # close parallel client if it was used
    # if parallel:
    #     client.close()

    # concatenate into a single dataarray for output
    runLabels = range(1, len(timeseries)+1)
    da = xr.concat(timeseries, dim='run')
    da = da.assign_coords(run=runLabels)

    return da

def main(args) -> None:

    thkPath = Path(args.thickness)
    zbPath = Path(args.z_base)
    outPath = Path(args.outfile)
    maskPath = Path(args.mask) if args.mask else None

    if outPath.suffix != '.nc':
        raise TypeError('outfile must be in netcdf (.nc) format')
    
    # netcdf4 won't allow you to overwrite existing netcdfs, so need to delete
    # existing file if the overwrite option is called
    if args.overwrite:
        outPath.unlink(missing_ok=True)
    
    elif outPath.is_file():
        raise FileExistsError(
            f'{outPath.name} already exists and will not be overwritten by default. '
            'Use --overwrite if you would like to overwrite the file.')

    try:
        da = getEnsembleSLC(thkPath, zbPath, maskPath=maskPath, parallel=args.parallel)
    except Exception as e:
        print(f'Error: {e}')
        raise
    ds = da.to_dataset(name='slc')
    ds.to_netcdf(outPath)


if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser(
        description="Process inputs and optionally select a mask and basin"
        )

    # add arguments
    parser.add_argument("thickness", type=str, help="Path to thickness netcdfs") 
    parser.add_argument("z_base", type=str, help="Path to z_base netcdfs")
    parser.add_argument("outfile", type=str, help="Path to output netcdf")

    # add optional arguments
    parser.add_argument("--mask", type=str, help="Path to basin mask")
    parser.add_argument("--overwrite", action="store_true",
                        help = "Will overwrite outfile if it already exists")
    parser.add_argument("--parallel", action="store_true",
                        help = "Use dask to parallelise computation")

    args = parser.parse_args()
    main(args)
