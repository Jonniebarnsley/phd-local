import argparse
import xarray as xr
from xarray import DataArray, Dataset
from pathlib import Path
    

def climatology(smb: DataArray, start_year: int, end_year: int) -> DataArray:
    """Computes the climatology of SMB over the specified year range."""
    return smb.sel(time=slice(start_year, end_year)).mean('time')

def _load_data(scenario_dir: Path, historical_files: Path) -> Dataset:
    """Loads scenario and historical SMB data from given files."""

    historical_files = sorted(historical_files.glob("*.nc"))
    if len(historical_files) == 0:
        raise FileNotFoundError(f"No historical SMB files found in {historical_files}")

    scenario_files = sorted(scenario_dir.glob("*.nc"))
    if len(scenario_files) == 0:
        raise FileNotFoundError(f"No scenario SMB files found in {scenario_dir}")
    
    return xr.open_mfdataset(historical_files + scenario_files, combine='by_coords')

def make_smb_anomaly(historical_dir: Path, scenario_dir: Path, smb_anomaly_dir: Path) -> None:
    
    data = _load_data(scenario_dir, historical_dir)
    smb = data.smb
    smb = smb.sel(time=slice(1995, 2300))  # limit to years with full data

    # Compute climatology and anomaly
    clim = climatology(smb, start_year=1995, end_year=2014)
    smb_anomaly = smb - clim

    # Add variable attributes
    smb_anomaly.attrs['long_name'] = "Surface Mass Balance Anomaly"
    smb_anomaly.attrs['description'] = "Surface Mass Balance anomaly relative to 1995-2014 climatology"
    smb_anomaly.attrs['units'] = "m a^-1"

    smb_anomaly.encoding.update({
        'zlib': True,
        'complevel': 4,
    })
    
    # Load all data into memory at once
    smb_anomaly = smb_anomaly.load()
    
    for year in smb_anomaly.time.values:

        smb_anomaly_file = _get_anomaly_output_dir(scenario_dir) / "{}_{}.nc".format(
            _get_anomaly_filestem(data),
            int(year)
        )
        if smb_anomaly_file.exists():
            print(f"    {smb_anomaly_file} already exists.")
            continue
            
        smb_anomaly_yr = smb_anomaly.sel(time=year)
        ds = xr.Dataset(
            {"smb_anomaly": smb_anomaly_yr},
            coords=smb_anomaly_yr.coords,  # Use coords from the slice, not full dataset
            attrs=data.attrs # copy attributes from original dataset
        )
    
        # Save the anomaly data
        print(smb_anomaly_file)
        try:
            ds.to_netcdf(smb_anomaly_file)
        except KeyboardInterrupt:
            if smb_anomaly_file.exists():
                smb_anomaly_file.unlink()  # Remove incomplete file
            print("Process interrupted. Incomplete file removed.")
            raise

def _find_data_directories(CMIP6_dir: Path) -> list[Path]:
    """Finds all scenario and historical SMB data directories under the base directory."""
    CMIP_dir = CMIP6_dir / "CMIP"
    for modelling_group in CMIP_dir.iterdir():
        if not modelling_group.is_dir():
            continue

        # Look for smb directory for historical experiment
        # Should be under <model>/historical/<variant>/Lyr/smb/bisicles-8km/<version>/
        valid_historical_dirs = [d for d in modelling_group.glob("*/historical/*/Lyr/smb/bisicles-8km/*")\
                                  if d.is_dir()]
        
        if len(valid_historical_dirs) == 0:
            raise FileNotFoundError(f"No historical SMB directory found for {modelling_group.name}")
        elif len(valid_historical_dirs) > 1:
            raise ValueError(f"Multiple historical SMB directories found for {modelling_group.name}:"+\
                             f" {valid_historical_dirs}. Please ensure only one version exists.")
        else:
            historical_dir = valid_historical_dirs[0]
        
        # Look for equivalent directories for each scenario experiment
        scenario_modellin_group = CMIP6_dir / "ScenarioMIP" / modelling_group.name
        scenario_dirs = [d for d in scenario_modellin_group.glob("*/*/*/Lyr/smb/bisicles-8km/*")\
                               if d.is_dir()]
        directory_strs = [str(d) for d in scenario_dirs]
        print("Calculating SMB anomaly for the following scenarios:", "\n".join(directory_strs), sep="\n")

    return historical_dir, scenario_dirs

def _get_anomaly_output_dir(scenario_dir: Path) -> Path:
    """Generates the output path for the SMB anomaly file based on the scenario directory."""
    version = scenario_dir.name  # Get the version (last directory in path)
    anomaly_dir = scenario_dir.parent.parent.parent / "smb_anomaly" / "bisicles-8km" / version
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    return anomaly_dir

def _get_anomaly_filestem(anomaly_ds: Path) -> str:
    """Generates the output file stem for the SMB anomaly dataset."""
    model = anomaly_ds.attrs['source_id']
    experiment = anomaly_ds.attrs['experiment_id']
    smb_anomaly_filename = f"smb_anomaly_{model}_{experiment}_8km"
    return smb_anomaly_filename

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("CMIP6", type=Path, help="path to CMIP6 base directory")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    historical_dir, scenario_dirs = _find_data_directories(args.CMIP6)
    for scenario_dir in scenario_dirs:
        anomaly_dir = _get_anomaly_output_dir(scenario_dir)
        make_smb_anomaly(historical_dir, scenario_dir, anomaly_dir)

if __name__ == "__main__":
    main()