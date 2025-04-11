from GPy.models.gp_regression import GPRegression
from gpflow.models.gpr import GPR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_rectangle_dimensions(N: int) -> tuple[int, int]:

    a = int(np.floor(np.sqrt(N)))
    b = int(np.ceil(N/a))

    return a, b

def main_effects_gpy(model: GPRegression, names: str=None, **kw) -> pd.DataFrame:

    _, N = model.X.shape
    num_testpoints = 1000

    X_plot = np.linspace(0, 1, num_testpoints)
    a, b = get_rectangle_dimensions(N)

    fig, axes = plt.subplots(nrows=a, ncols=b, dpi=150)

    for i in range(N):

        ax = axes.flatten()[i]
        input = np.full((num_testpoints, N), 0.5)
        input[:,i] = X_plot

        Y_mean, Y_var = model.predict(input)

        ax.plot(X_plot, Y_mean)
        ax.fill_between(X_plot, 
                 (Y_mean - 1.96 * np.sqrt(Y_var)).flatten(), 
                 (Y_mean + 1.96 * np.sqrt(Y_var)).flatten(), 
                 alpha=0.2, label='95% confidence interval')
        ax.set_ylim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])

        if names:
            ax.set_title(names[i], size=8)

    L = len(axes.flatten())
    if N < L:
        for i in range(N, L):
            axes.flatten()[i].set_axis_off()
