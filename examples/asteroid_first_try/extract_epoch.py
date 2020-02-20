"""
Extract specific epochs from each of many files. The argument is a two-column
file: 1) names of files, and 2) corresponding epochs to be extracted.
"""
import sys
import os
import numpy as np


def get_index_nearest(array, value):
    """
    find the index of the closest value in an array
    """
    return (np.abs(array - value)).argmin()


max_time_diff = 0.005  # days; = 7.2 min.
dt = 2450000.

with open(sys.argv[1]) as in_file:
    in_data = in_file.readlines()

out_epoch = []
out_flux = []
for line in in_data:
    file_name = line.split()[0]
    ref_epoch = float(line.split()[1]) - dt
    if not os.path.isfile(file_name):
        continue
    (time, flux) = np.loadtxt(file_name, unpack=True, usecols=(0, 1))
    index = get_index_nearest(time, ref_epoch)
    if np.abs(time[index] - ref_epoch) > max_time_diff:
        continue
    out_epoch.append(time[index] + dt)
    out_flux.append(flux[index])

for (epoch, flux) in zip(out_epoch, out_flux):
    print("{:.5f} {:.3f}".format(epoch, flux))
