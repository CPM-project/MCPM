import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import os
from pymultinest.solve import solve
from astropy.coordinates import SkyCoord
from astropy import units as u

import MulensModel as MM
from MCPM import utils
from MCPM.cpmfitsource import CpmFitSource
from MCPM.minimizer import Minimizer


# data files:
files = ['phot_01_corr_sel1.dat', 'phot_CFHT_01_g.dat', 'phot_CFHT_01_i.dat', 'phot_CFHT_01_r.dat']
files_fmt = ['mag', 'flux', 'flux', 'flux']
ephemeris_file = 'K2_ephemeris_01.dat'

# info on fitting:
parameters_to_fit = ['pi_E_N', 'pi_E_E', "u_0", "t_0", "t_E", "f_s_sat"]
# The order is forced by MultiNest, because it runs clustering on
# the first N parameters (N=3 in our case - see n_modes below)
t_0_par = 2457573.5
t_0 = 2457573.4
u_0 = 0.45
t_E = 2.1
pi_E_N = 0.
pi_E_E = 0.
f_s_sat = 150.
starting = [pi_E_N, pi_E_E, u_0, t_0, t_E, f_s_sat]

# MN settings:
mn_min = np.array([-1.5, -1.5, -0.8, t_0-0.3, 1., 1.])
mn_max = np.array([1.5, 1.5, 0.8, t_0+0.3, 4., 400.])
n_modes = len(['pi_E_N', 'pi_E_E', "u_0"])
n_live_points = 1000
max_iter = np.int64(1.e5) # This makes it run for long, but reasonable time.
# If you don't want to set a limit, then make it 0.
dir_out = "chains/"
file_prefix = __file__.split(".py")[0] + "-"
file_all_prefix = file_prefix + "all_models" # This file will have all models calculated.

# MCPM settings
channel = 52
campaign = 92
ra = 270.865792
dec = -28.373194
ra_unit = u.deg
dec_unit = u.deg
half_size = 2
n_select = 10
l2 = 10**6.5
t_0_sat = t_0
u_0_sat = 0.4
sat_sigma = 25. # somehow arbitrary value

# End of settings.
###################################################################

# read datasets
datasets = []
coords = MM.Coordinates(SkyCoord(ra, dec, unit=(ra_unit, dec_unit)))
for (file_, fmt) in zip(files, files_fmt):
    data = MM.MulensData(file_name=file_, add_2450000=True, phot_fmt=fmt, 
            coords=coords)
    datasets.append(data)
    
# prepare cpm_source: 
cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, channel=channel)
cpm_source.get_predictor_matrix()
cpm_source.set_l2_l2_per_pixel(l2=l2)
cpm_source.set_pixels_square(half_size)
cpm_source.select_highest_prf_sum_pixels(n_select)

# satellite dataset
model_1 = utils.pspl_model(t_0_sat, u_0_sat, t_E, f_s_sat, cpm_source.pixel_time)
# We have to run some model to get the cpm_source.residuals_mask
cpm_source.run_cpm(model_1)
mask = cpm_source.residuals_mask
sat_time = cpm_source.pixel_time[mask] + 2450000.
data = MM.MulensData([sat_time, 0.*sat_time, sat_time*0.+sat_sigma], 
            phot_fmt='flux', ephemerides_file=ephemeris_file)
datasets.append(data)

# initiate model, event, minimizer
n_params = len(starting)
model = MM.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E, 
        'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E, 
        't_0_par': t_0_par}) 
event = MM.Event(datasets=datasets, model=model)
minimizer = Minimizer(event, parameters_to_fit, cpm_source)
minimizer.file_all_models = file_all_prefix
minimizer.set_chi2_0(np.sum([len(d.time) for d in datasets]))

# MultiNest fit:
if not os.path.exists(dir_out):
    os.mkdir(dir_out)
minimizer.set_MN_cube(mn_min, mn_max)
result_2 = solve(LogLikelihood=minimizer.ln_like, 
    Prior=minimizer.transform_MN_cube, 
    n_dims=n_params,
    max_iter=max_iter,
    n_clustering_params=n_modes,
    n_live_points=n_live_points, 
    outputfiles_basename=dir_out+file_prefix,
    resume=False)
print('parameter values:')
for name, col in zip(parameters_to_fit, result_2['samples'].transpose()):
    print('{:10s} : {:.4f} +- {:.4f}'.format(name, col.mean(), col.std()))
minimizer.print_min_chi2()
print()
minimizer.reset_min_chi2()   
minimizer.close_file_all_models()
