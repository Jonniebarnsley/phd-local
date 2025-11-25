import argparse
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from xarray import Dataset, DataArray
from matplotlib.animation import FuncAnimation, FFMpegWriter
#from local.lib.utils import forceNamingConvention

def animate(da: DataArray, **kw) -> FuncAnimation:

    time=da.time

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    init = da.isel(time=0)
    im = ax.pcolormesh(init, **kw)
    fig.colorbar(im)

    def update(frame):

        data = da.sel(time=frame)

        # plot data
        ax.clear()
        im = data.plot(ax=ax, **kw)
        ax.set_title(f'year = {frame}')

        return im

    animation = FuncAnimation(
        fig, 
        update, 
        frames = time.values, 
        blit=False
        )
    
    plt.close(fig)
    
    return animation

def main(args) -> None:

    netcdf = Path(args.netcdf)
    variable = args.variable
    outfile = Path(args.outfile)
    cmap = args.cmap if args.cmap else 'Blues'

    if outfile.is_file() and not args.overwrite:
        print(f'{outfile.name} already exists')
        return

    ds = xr.open_dataset(netcdf)
    #ds = forceNamingConvention(ds)
    animation = animate(ds[variable], cmap=cmap)
    writervideo = FFMpegWriter(fps=30, bitrate=5000)
    animation.save(outfile, writer=writervideo, dpi=300)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process inputs and options"
        )
    
    # add arguments
    parser.add_argument("netcdf", type=str, help="netcdf to turn into an mp4") 
    parser.add_argument("variable", type=str, help="variable to extract from netcdf")
    parser.add_argument("outfile", type=str, help="save path for output mp4")

    # add optional arguments
    parser.add_argument("--cmap", type=int, help="colormap for pcolormesh")
    parser.add_argument("--overwrite", action="store_true", 
                        help="Will overwrite outfile if it already exists")

    args = parser.parse_args()
    main(args)