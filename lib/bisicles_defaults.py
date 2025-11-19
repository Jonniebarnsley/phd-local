import numpy as np

coords_8km = np.arange(4e3, 6.144e6, 8e3)
GRID_8KM = (coords_8km, coords_8km) # 768x768 8km BISICLES grid

ICE_DENSITY = 917.0  # kg m^-3
OCEAN_DENSITY = 1027.0  # kg m^-3