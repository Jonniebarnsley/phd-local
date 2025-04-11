import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import gaussian_kde

def get_sci_notation(ax, which) -> str:

    '''
    Calculates the appropriate scientific notation to label an axis. Returns a string like x10^n,
    where n is determined from the data.

    inputs:
        - ax: a matplotlib axes object
        - which: x or y
    '''

    if which=='x':   
        maxval = max(ax.get_xticks())
    elif which=='y':
        maxval = max(ax.get_yticks())
    else:
        raise ValueError("which must be either 'x' or 'y'")

    exponent = np.floor(np.log10(maxval)).astype(int)
    notation = f'$\\times \\, 10^{{{exponent}}}$'

    return notation

def plot_density(x, ax, label=None) -> None:

    '''
    Calculates a kernel density estimate for a parameter and plots it on a specified axis.

    inputs:
        - x     : parameter values
        - ax    : axis on which to plot the kernel density estimate
        - label : parameter name to label the axis
    '''
    
    ax.set_xticks([])
    ax.set_yticks([])

    # calculate density     
    kde = gaussian_kde(x, bw_method=0.25)
    xkde = np.linspace(min(x), max(x), 1000)
    ykde = kde(xkde)
    ymax = max(ykde)

    # plot density
    ax.plot(xkde, ykde, c='black', lw=0.8)
    ax.set_ylim((0, ymax*1.8))
    ax.set_xlim((min(x), max(x)))
    ax.text(0.5, 0.8, label, transform=ax.transAxes, ha='center', va='center')

def plot(df: pd.DataFrame, ticks=None, labels=None, groupby=None, **kwargs) -> plt.figure:

    '''
    Creates a pairs plot for a perturbed parameter ensemble with kernel density estimates
    along the diagonal.

    inputs:
        - df: Pandas dataframe containing the PPE samples
        - ticks: dictionary with the dataframe headers as keys. e.g.
            ticks = {
                'gamma0'    : [0, 2e5, 4e5],
                'UMV'       : [0, 1e21]
            }
        - labels: list of alternative names for parameters.
            e.g. labels = ['$\gamma_0$', 'UMV']
        - groupby: header or mask with which to group the data by when plotting
        - **kwargs: key word arguments inherited from plt.scatter. e.g. 
            kw = {
                'edgecolor' : 'black',
                'facecolor' : 'none',
                'size'      : 5,
                'linewidth' : 0.5
            }
    '''

    # labels are column names if not otherwise specified
    cols = df.columns
    if not labels:
        labels=cols

    # plot setup
    N = len(cols)
    fig, axes = plt.subplots(ncols=N, nrows=N, figsize=(7, 7), dpi=600)

    for i, j in product(range(N), range(N)):
        ax = axes[i][j]
        x = df[cols[i]]
        y = df[cols[j]]

        # upper panel
        if i<j:
            ax.set_axis_off()
            continue
        
        # diagonal panel
        elif i==j:
            label = labels[i]
            plot_density(x, ax=ax, label=label)
            continue

        # lower panel
        if groupby is None:
            ax.scatter(y, x, **kwargs)
        else:
            for _, subdf in df.groupby(groupby):
                x = subdf[cols[i]]
                y = subdf[cols[j]]
                ax.scatter(y, x, **kwargs)

        # customise xticks
        if i<N-1:
            ax.set_xticks([])
        elif ticks:
            ax.set_xticks(ticks[cols[j]])
            ax.get_xaxis().get_offset_text().set_visible(False)
            ax.set_xlabel(get_sci_notation(ax, which='x'))
        
        # customise yticks
        if j>0:
            ax.set_yticks([])
        elif ticks:
            ax.set_yticks(ticks[cols[i]])
            ax.get_yaxis().get_offset_text().set_visible(False)
            ax.set_ylabel(get_sci_notation(ax, which='y'))
            ax.tick_params(axis='y', labelrotation=90)

        # formatting applicable to all subplots
        ax.set_aspect(1/ax.get_data_ratio())
        ax.ticklabel_format(style='sci', scilimits=(-2,2))

    fig.subplots_adjust(hspace=0.15, wspace=0.15)

    return fig