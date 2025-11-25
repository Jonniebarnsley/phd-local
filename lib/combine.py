from netCDF4 import Dataset
from pathlib import Path

# Directories
vars = ['thickness', 'dThicknessdt', 'xVel', 'yVel', 'Z_base', 'Z_surface']
outdir = Path('combined')
outdir.mkdir(exist_ok=True)

anchor = 'thickness'  # base directory to match filenames

for f in sorted(Path(anchor).glob('*.nc')):
    fname = f.name
    # Extract prefix/suffix around the variable name
    prefix = fname.split(f"_{anchor}_")[0]
    suffix = fname.split(f"_{anchor}_")[1]
    outfile = outdir / f"{prefix}_{suffix}"
    if outfile.exists():
        continue
    # Copy dimensions and coords from the first file
    with Dataset(f, 'r') as src, Dataset(outfile, 'w') as dst:
        # Copy global attributes
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        # Copy dimensions
        for name, dim in src.dimensions.items():
            dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))
        # Copy all variables
        for vname, var in src.variables.items():
            out_var = dst.createVariable(vname, var.datatype, var.dimensions)
            out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            out_var[:] = var[:]

    # Append variables from other directories
    with Dataset(outfile, 'a') as dst:
        for v in vars:
            if v == anchor:
                continue
            src_path = Path(v) / f"{prefix}_{v}_{suffix}"
            if not src_path.exists():
                print(f"⚠️ Missing: {src_path}")
                continue
            with Dataset(src_path, 'r') as src:
                for vname, var in src.variables.items():
                    new_vname = f"{v}_{vname}" if vname in dst.variables else vname
                    out_var = dst.createVariable(new_vname, var.datatype, var.dimensions)
                    out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    out_var[:] = var[:]

    print(f"✅ Combined: {outfile}")
