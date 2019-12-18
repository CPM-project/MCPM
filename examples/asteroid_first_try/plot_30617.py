"""
plot LC of asteroid 30617
"""
import numpy as np
from matplotlib import pyplot as plt


file_name = "lc_30617.dat"

(time, flux) = np.loadtxt(file_name, unpack=True)

time -= 2450000.

# First plot:
plt.plot(time, flux, 'ro')
plt.show()

# There are many outliers, remove them:
mask = (flux > 0.) & (flux < 550.)
time = time[mask]
flux = flux[mask]
plt.plot(time, flux, 'ro')
plt.show()

# By hand measure slopes:
slope_1 = 9.5  # [flux/day]
slope_2 = -9.2
time_0 = 7562.
# and correct for them:
mask = (time < time_0)
flux[mask] -= slope_1 * (time[mask] - time_0)
mask = (time > time_0)
flux[mask] -= slope_2 * (time[mask] - time_0)
plt.plot(time, flux, 'ro')
plt.show()

# Phase the data:
period = 0.1311
phase = (time % period) / period
plt.plot(phase, flux, 'ro')
plt.show()
