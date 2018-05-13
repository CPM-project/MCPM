import numpy as np
import os
import sys
import emcee
from astropy.coordinates import SkyCoord
from astropy import units as u
import configparser

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

# Read EMCEE options:
out = read_config.read_EMCEE_options(config)
(starting_mean, starting_sigma, parameters_to_fit) = out[:3]
(min_values, max_values, emcee_settings) = out[3:]

# Read MCPM options:
MCPM_options = read_config.read_MCPM_options(config)

# End of settings.
###################################################################
n_params = len(parameters_to_fit)
if file_all_models is None:
    file_all_models = os.path.splitext(config_file)[0] + ".models"

# read datasets
datasets = []
if skycoord is not None:
    coords = MM.Coordinates(skycoord)
else:
    coords = None
if files is not None:
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
parameters = {key: value for (key, value) in zip(parameters_to_fit, starting_mean)}
parameters.update(parameters_fixed)
model = MM.Model(parameters)
if methods is not None:
    model.set_magnification_methods(methods)

for cpm_source in cpm_sources:
    sat_model = utils.pspl_model(parameters['t_0'], parameters['u_0'], 
            parameters['t_E'], parameters['f_s_sat'], cpm_source.pixel_time)
    cpm_source.run_cpm(sat_model)
    
    utils.apply_limit_time(cpm_source, MCPM_options)

    mask = cpm_source.residuals_mask
    if 'mask_model_epochs' in MCPM_options:
        mask *= utils.mask_nearest_epochs(cpm_source.pixel_time+2450000., 
            MCPM_options['mask_model_epochs'])
    sat_time = cpm_source.pixel_time[mask] + 2450000.
    #sat_sigma = sat_time * 0. + MCPM_options['sat_sigma']
    sat_sigma = np.sqrt(np.sum(np.array([err[mask] for err in cpm_source.pixel_flux_err])**2, axis=0))
    if 'sat_sigma_scale' in MCPM_options:
        sat_sigma *= MCPM_options['sat_sigma_scale']
    data = MM.MulensData([sat_time, 0.*sat_time, sat_sigma],
            phot_fmt='flux', ephemerides_file=MCPM_options['ephemeris_file'])
    datasets.append(data)

# initiate event and minimizer
event = MM.Event(datasets=datasets, model=model, coords=skycoord)
params = parameters_to_fit[:]
minimizer = Minimizer(event, params, cpm_sources)
minimizer.file_all_models = file_all_models
minimizer.set_chi2_0()
if 'coeffs_fits_in' in MCPM_options:
    minimizer.read_coeffs_from_fits(MCPM_options['coeffs_fits_in'])
if 'coeffs_fits_out' in MCPM_options:
    minimizer.start_coeffs_cache()

if 'mask_model_epochs' in MCPM_options:
    minimizer.model_masks[0] = utils.mask_nearest_epochs(
        cpm_sources[0].pixel_time+2450000., MCPM_options['mask_model_epochs'])

# EMCEE fit:
print("EMCEE walkers, steps, burn: {:} {:} {:}".format(
    emcee_settings['n_walkers'], emcee_settings['n_steps'], 
    emcee_settings['n_burn']))
minimizer.set_prior_boundaries(min_values, max_values)
starting = [starting_mean + starting_sigma * np.random.randn(n_params)
            for i in range(emcee_settings['n_walkers'])]
for start_ in starting:
    if minimizer.ln_prior(start_) <= -float('inf'):
        raise ValueError('starting point is not in prior')
sampler = emcee.EnsembleSampler(
    emcee_settings['n_walkers'], n_params, minimizer.ln_prob)
# run:
sampler.run_mcmc(starting, emcee_settings['n_steps'])

# cleanup and close minimizer:
samples = sampler.chain[:, emcee_settings['n_burn']:, :].reshape((-1, n_params))
blob_sampler = np.transpose(np.array(sampler.blobs), axes=(1, 0, 2))
blob_samples = blob_sampler[:, emcee_settings['n_burn']:, :].reshape((-1, n_fluxes))
if 'coeffs_fits_out' in MCPM_options:
    minimizer.set_pixel_coeffs_from_samples(samples)
    minimizer.save_coeffs_to_fits(MCPM_options['coeffs_fits_out'])
    minimizer.stop_coeffs_cache()
minimizer.close_file_all_models()

# output
print("Mean acceptance fraction: {:.4f} +- {:.4f}".format(
    np.mean(sampler.acceptance_fraction),
    np.std(sampler.acceptance_fraction)))
results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        zip(*np.percentile(samples, [16, 50, 84], axis=0)))
for (param, r) in zip(parameters_to_fit, results):
    print('{:7s} : {:.4f} {:.4f} {:.4f}'.format(param, *r))
n_fluxes = blob_sampels.shape[-1]
blob_results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(blob_samples, [16, 50, 84], axis=0)))
flux_name = ['S', 'B']
for (i, r) in zip(range(n_fluxes), blob_results):
    print('flux_{:}_{:} : {:.4f} {:.4f} {:.4f}'.format(flux_name[i%2], i//2+1, *r))
print('Best model:')
minimizer.print_min_chi2()
