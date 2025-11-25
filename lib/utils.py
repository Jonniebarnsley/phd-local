import numpy as np
from xarray import Dataset, DataArray
from lib.xy2ll import xy2ll
from math import pi

def round_sig_figs(x: float, sig: int=2) -> int | float:

    '''
    Rounds float <x> to <sig> significant figures.
    '''

    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

def forceNamingConvention(ds: Dataset) -> Dataset:

    '''
    Handles cases with Datasets where variables are labelled using multiple possible
    common names, e.g. 'time', 't', 'year', etc.. Currently only does this for time
    and basin but could be expanded to more variables and their common name variants.

    inputs:
        - ds: Dataset with any old variable names
    output:
        - ds with variables (hopefully) renamed to match standard naming conventions
    '''

    LookupVariants = {
        'time':     {'t', 'Time', 'year', 'Year'},
        'basin':    {'basins', 'Basin', 'Basins'}
    }

    for standard_name, variants in LookupVariants.items():
        wrong_vars = variants.intersection(ds.variables)
        if len(wrong_vars) > 0 and standard_name not in ds.variables:
            wrong_name = wrong_vars.pop()
            ds = ds.rename({wrong_name: standard_name})
    
    return ds

def checkDims(da: DataArray, dims: set) -> None:

    '''
    We often require that a dataarray includes certain dimensions (commonly x and y) in
    order for the script to function. This checks to ensure a dataarray includes a given 
    set of dimensions and raises a ValueError if it doesn't.
    '''

    if not dims.issubset(da.dims):
        raise ValueError(
            f'{da.name} requires dimensions {dims} but has dimensions {da.dims}'
        )
    

def checkAlignment(A: DataArray, B: DataArray) -> None:

    '''
    It's common that two dataarrays may come with different resolutions, projections, 
    coordinate grids, etc. It's important that we check dataarrays properly align before
    applying operations to them, e.g. sea level calculation. This function makes the
    appropriate checks and raises a ValueError if any fail.
    '''

    if A.dims != B.dims:
        raise ValueError(
            f'{A.name} and {B.name} have different dimensions: {A.dims} and {B.dims}'
        )

    if A.shape != B.shape:
        raise ValueError(
            f'{A.name} and {B.shape} have different shapes: {A.shape} and {B.shape}'
        )

    for dim in A.dims:
        if not all(A[dim] == B[dim]):
            raise ValueError(
                f'{A.name} and {B.name} do not align along axis {dim}. '
                f'{A.name}: {float(A[dim].min())} to {float(A[dim].max())}. '
                f'{B.name}: {float(B[dim].min())} to {float(B[dim].max())}'
            )
        
def scaleFactor(da: DataArray, sgn: int) -> DataArray:

    '''
    Calculates the area scale factor for a DataArray on a Polar Stereographic
    grid.

    Inputs:
        - da: DataArray with dimensions [x, y, ...]
        - sgn: integer indicating the hemisphere.
            +1 if North Pole
            -1 if South Pole
    Returns:
        - DataArray for k, the area scale factor (Geolzer et al., 2020)
    '''

    checkDims(da, {'x', 'y'})
    x = da.x
    y = da.y

    # centre origin on the pole if not already
    xs = x - x.mean()
    ys = y - y.mean()
 
    lat, lon = xy2ll(xs, ys, sgn)
    k = 2/(1+np.sin(sgn*lat*2*pi/360))

    return k