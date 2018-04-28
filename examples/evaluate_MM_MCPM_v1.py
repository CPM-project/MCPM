import numpy as np
import os
import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import configparser
import matplotlib.pyplot as plt

import MulensModel as MM

from MCPM import utils
from MCPM.cpmfitsource import CpmFitSource
from MCPM.minimizer import Minimizer

import read_config


if len(sys.argv) != 2:
    raise ValueError('Exactly one argument needed - cfg file')
config_file = sys.argv[1]

config = configparser.ConfigParser()
config.optionxform = str
config.read(config_file)

# Read general options:
out = read_config.read_general_options(config)
(skycoord, methods, file_all_models) = out[:3]
(files, files_formats, parameters_fixed) = out[3:]

# Read models:
out = read_config.read_models(config)
(parameter_values, model_ids) = out[:2]
(plot_files, txt_files, parameters_to_fit) = out[2:]

# Read MCPM options:
MCPM_options = read_config.read_MCPM_options(config)

# End of settings.
###################################################################

# read datasets
datasets = []
if skycoord is not None:
    coords = MM.Coordinates(skycoord)
else:
    coords = None
for (file_, fmt) in zip(files, files_formats):
    data = MM.MulensData(file_name=file_, add_2450000=True, phot_fmt=fmt, 
            coords=coords)
    datasets.append(data)
    
# satellite datasets
cpm_sources = []
for campaign in MCPM_options['campaigns']:
    cpm_source = CpmFitSource(ra=skycoord.ra.deg, dec=skycoord.dec.deg, 
                campaign=campaign, channel=MCPM_options['channel'])
    cpm_source.get_predictor_matrix(**MCPM_options['predictor_matrix'])
    cpm_source.set_l2_l2_per_pixel(l2=MCPM_options['l2'], 
                l2_per_pixel=MCPM_options['l2_per_pixel'])
    cpm_source.set_pixels_square(MCPM_options['half_size'])
    cpm_source.select_highest_prf_sum_pixels(MCPM_options['n_select'])

    cpm_sources.append(cpm_source)

# initiate model
model_begin = parameter_values[0]
parameters = {key: value for (key, value) in zip(parameters_to_fit, model_begin)}
parameters.update(parameters_fixed)
model = MM.Model(parameters)
if methods is not None:
    model.set_magnification_methods(methods)
for cpm_source in cpm_sources:
    sat_model = utils.pspl_model(parameters['t_0'], parameters['u_0'], 
            parameters['t_E'], parameters['f_s_sat'], cpm_source.pixel_time)
    cpm_source.run_cpm(sat_model)

    if 'train_mask_time_limit' in MCPM_options:
        mask_1 = np.zeros_like(cpm_source.pixel_time, dtype=bool)
        mask_2 = ~np.isnan(cpm_source.pixel_time)
        limit = MCPM_options['train_mask_time_limit'] - 2450000.
        mask_1[mask_2] = (cpm_source.pixel_time[mask_2] < limit)
        if np.sum(mask_1) == 0:
            raise ValueError('value of train_mask_time_limit results in 0 ' +
                'epochs in training mask')
        cpm_source.set_train_mask(mask_1)

    mask = cpm_source.residuals_mask
    sat_time = cpm_source.pixel_time[mask] + 2450000.
    sat_sigma = sat_time * 0. + MCPM_options['sat_sigma']
    data = MM.MulensData([sat_time, 0.*sat_time, sat_sigma],
            phot_fmt='flux', ephemerides_file=MCPM_options['ephemeris_file'])
    datasets.append(data)
    
# initiate event    
event = MM.Event(datasets=datasets, model=model)
params = parameters_to_fit[:]
minimizer = Minimizer(event, params, cpm_sources)
if 'coeffs_fits_in' in MCPM_options:
    minimizer.read_coeffs_from_fits(MCPM_options['coeffs_fits_in'])
if 'coeffs_fits_out' in MCPM_options:
    raise ValueError("coeffs_fits_out cannot be set in this program")
    
# main loop:
for (values, name, plot_file, txt_file) in zip(parameter_values, model_ids, plot_files, txt_files):
    for (key, value) in zip(parameters_to_fit, values):
        setattr(model.parameters, key, value)
    print(name, minimizer.chi2_fun(values))
    if txt_file is not None:
        (y_, y_mask) = cpm_source.prf_photometry()
        y = y_[y_mask]
        #else:
            #minimizer.set_satellite_data(values)
            #minimizer.chi2_fun(values)
            #y = minimizer.event.datasets[-1].flux#[y_mask]
        x = cpm_source.pixel_time[y_mask]
        np.savetxt(txt_file, np.array([x, y]).T)
    if plot_file is not None:
        minimizer.set_satellite_data(values)
        minimizer.standard_plot(7530., 7573., [14.65, 13.6], title=name)
        plt.savefig(plot_file)
        plt.close()
        print("{:} file saved".format(plot_file))
    if len(datasets) > 1:
        for i in range(len(datasets)):
            print(i, event.get_chi2_for_dataset(i))
    print()
