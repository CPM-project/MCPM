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
(files, files_formats, files_kwargs, parameters_fixed) = out[3:]

# Read EMCEE options:
out = read_config.read_EMCEE_options(config)
(starting_mean, starting_sigma, parameters_to_fit) = out[:3]
(min_values, max_values, emcee_settings) = out[3:]

# Read MCPM options:
MCPM_options = read_config.read_MCPM_options(config)

# other constraints:
other_constraints = read_config.read_other_constraints(config)

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
    for (file_, fmt, kwargs) in zip(files, files_formats, files_kwargs):
        data = MM.MulensData(file_name=file_, add_2450000=True, phot_fmt=fmt,
                coords=coords, **kwargs)
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
parameters_ = {**parameters}
for p in ['f_s_sat', 'q_f', 'log_q_f']:
    parameters_.pop(p, None)
model = MM.Model(parameters_, coords=skycoord)
if methods is not None:
    model.set_magnification_methods(methods)

for cpm_source in cpm_sources:
    if model.n_sources == 2:
        if 'log_q_f' in parameters:
            q_f = 10**parameters['log_q_f']
        else:
            q_f = parameters['q_f']
        model.set_source_flux_ratio(q_f)
    times = cpm_source.pixel_time + 2450000.
    times[np.isnan(times)] = np.mean(times[~np.isnan(times)])
    cpm_source.run_cpm(parameters['f_s_sat'] * model.magnification(times))
    
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
if 'sat_sigma_scale' in MCPM_options:
    minimizer.sigma_scale = MCPM_options['sat_sigma_scale']
if 'color_constraint' in MCPM_options:
    cc = 'color_constraint'
    ref_dataset = files.index(MCPM_options[cc][0])
    if len(MCPM_options[cc]) == 3:
        ref_mag = MM.utils.MAG_ZEROPOINT
    else:
        ref_mag = MCPM_options[cc][1]
        
    if len(MCPM_options[cc]) in [3, 4]:
        minimizer.add_color_constraint(ref_dataset, ref_mag, 
            MCPM_options[cc][-2], MCPM_options[cc][-1])
    elif len(MCPM_options[cc]) in [5, 6, 7, 8]:
        minimizer.add_full_color_constraint(ref_dataset,
            files.index(MCPM_options[cc][1]), files.index(MCPM_options[cc][2]), 
            *MCPM_options[cc][3:])
    #def add_full_color_constraint(self,
            #ref_dataset_0, ref_dataset_1, ref_dataset_2, 
            #polynomial, sigma, ref_zero_point_0=MAG_ZEROPOINT, 
            #ref_zero_point_1=MAG_ZEROPOINT, ref_zero_point_2=MAG_ZEROPOINT):    
    else:
        raise ValueError('wrong size of "color_constraint" option')
key = 'min_blending_flux'
if key in other_constraints:
    index = files.index(other_constraints[key][0])
    other_constraints['min_blending_flux'] = [datasets[index], other_constraints[key][1]]
minimizer.other_constraints = other_constraints

key = 'no_blending_files'
if key in MCPM_options:
    indexes = [files.index(f) for f in MCPM_options['no_blending_files']]
    for ind in indexes:
        minimizer.fit_blending[ind] = False

if 'mask_model_epochs' in MCPM_options:
    for i in range(minimizer.n_sat):
        minimizer.model_masks[i] = utils.mask_nearest_epochs(
                cpm_sources[i].pixel_time+2450000.,
                MCPM_options['mask_model_epochs'])

# EMCEE fit:
print("EMCEE walkers, steps, burn: {:} {:} {:}".format(
    emcee_settings['n_walkers'], emcee_settings['n_steps'], 
    emcee_settings['n_burn']))
minimizer.set_prior_boundaries(min_values, max_values)
starting = [starting_mean + starting_sigma * np.random.randn(n_params)
            for i in range(emcee_settings['n_walkers'])]
for start_ in starting:
    if minimizer.ln_prior(start_) <= -float('inf'):
        raise ValueError('starting point is not in prior\n', start_)
sampler = emcee.EnsembleSampler(
    emcee_settings['n_walkers'], n_params, minimizer.ln_prob)
acceptance_fractions = []
# run:
#sampler.run_mcmc(starting, emcee_settings['n_steps'])
for results in sampler.sample(starting, iterations=emcee_settings['n_steps']):
    acceptance_fractions.append(np.mean(sampler.acceptance_fraction))

# cleanup and close minimizer:
out_name = emcee_settings.get('file_acceptance_fractions', None)
if out_name is not None:
    with open(out_name, 'w') as file_out:
        file_out.write('\n'.join([str(af) for af in acceptance_fractions]))
samples = sampler.chain[:, emcee_settings['n_burn']:, :].reshape((-1, n_params))
blob_sampler = np.transpose(np.array(sampler.blobs), axes=(1, 0, 2))
n_fluxes = blob_sampler.shape[-1]
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
blob_results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(blob_samples, [16, 50, 84], axis=0)))
flux_name = ['S', 'B']
for (i, r) in zip(range(n_fluxes), blob_results):
    print('flux_{:}_{:} : {:.4f} {:.4f} {:.4f}'.format(flux_name[i%2], i//2+1, *r))
print('Best model:')
minimizer.print_min_chi2()

