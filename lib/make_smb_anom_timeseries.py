import xarray as xr
from pathlib import Path
from bisicles_defaults import GRID_8KM
xs, ys = GRID_8KM

smb_anom_dir = Path("/Volumes/LaCie/forcings/smb_anomaly/")
ts_dir = smb_anom_dir.parent / "smb_anomaly_timeseries_antarctic_mask"

bedmachine = xr.open_dataset("/Users/jonniebarnsley/data/BedMachine/BedMachine_antarctica_v3_bisicles_1km.nc")
ice = (bedmachine.mask != 0)
coarse = ice.coarsen(x=8, y=8).max()
coarse = coarse.assign_coords(x=xs, y=ys)

for scenario in smb_anom_dir.iterdir():
    if not scenario.is_dir():
        continue
    for model in scenario.iterdir():
        if not model.is_dir():
            continue
        print(scenario.name, model.name)
        smb_anom_files = sorted(model.glob("smb_anomaly_*.nc"))
        smb_anom_ds = xr.open_mfdataset(
            smb_anom_files,
            combine='nested',
            concat_dim='time'
        )
        smb_anom_ds = smb_anom_ds.where(coarse)
        smb_anom_ts = smb_anom_ds.mean(dim=['x', 'y'])
        # Save combined time series
        outdir = ts_dir / scenario.name
        outdir.mkdir(parents=True, exist_ok=True)
        out_file = outdir / f"smb_anomaly_{model.name}_{scenario.name}_timeseries.nc"
        print(f"Saving combined time series to {out_file}")
        smb_anom_ts.to_netcdf(out_file)