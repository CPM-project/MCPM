import numpy as np
import os
import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import configparser
from pymultinest.solve import solve
from pymultinest.analyse import Analyzer

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
read_config.check_sections_in_config(config)

# Read general options:
out = read_config.read_general_options(config)
(skycoord, methods, file_all_models) = out[:3]
(files, files_formats, files_kwargs, parameters_fixed) = out[3:]

# Read MN options:
out = read_config.read_MultiNest_options(config, config_file)
(mn_min, mn_max, parameters_to_fit, MN_args) = out

# Read MCPM options:
MCPM_options = read_config.read_MCPM_options(config)

# other constraints:
other_constraints = read_config.read_other_constraints(config)

# End of settings.
###################################################################
n_params = len(parameters_to_fit)
config_file_root = os.path.splitext(config_file)[0]
if file_all_models is None:
    file_all_models = config_file_root + ".models"

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
    cpm_source = CpmFitSource(
        ra=skycoord.ra.deg, dec=skycoord.dec.deg,
        campaign=campaign, channel=MCPM_options['channel'])
    cpm_source.get_predictor_matrix(**MCPM_options['predictor_matrix'])
    cpm_source.set_l2_l2_per_pixel(l2=MCPM_options['l2'],
                                   l2_per_pixel=MCPM_options['l2_per_pixel'])
    cpm_source.set_pixels_square(MCPM_options['half_size'])
    cpm_source.select_highest_prf_sum_pixels(MCPM_options['n_select'])

    cpm_sources.append(cpm_source)

# initiate model
zip_ = zip(parameters_to_fit, mn_min, mn_max)
parameters = {key: (v1+v2)/2. for (key, v1, v2) in zip_}
parameters.update(parameters_fixed)
parameters_ = {**parameters}
for param in list(parameters_.keys()).copy():
    if (param == 'f_s_sat' or param[:3] == 'q_f' or param[:7] == 'log_q_f'):
        parameters_.pop(param)
model = MM.Model(parameters_, coords=coords)
for (m_key, m_value) in methods.items():
    model.set_magnification_methods(m_value, m_key)

for cpm_source in cpm_sources:
    times = cpm_source.pixel_time + 2450000.
    times[np.isnan(times)] = np.mean(times[~np.isnan(times)])
    if model.n_sources == 1:
        model_magnification = model.magnification(times)
    else:
        if ('log_q_f' in parameters) or ('q_f' in parameters):
            if 'log_q_f' in parameters:
                q_f = 10**parameters['log_q_f']
            else:
                q_f = parameters['q_f']
            model.set_source_flux_ratio(q_f)
            model_magnification = model.magnification(times)
        else:
            model_magnification = model.magnification(
                times, separate=True)[0]  # This is very simple solution.
    cpm_source.run_cpm(parameters['f_s_sat'] * model_magnification)

    utils.apply_limit_time(cpm_source, MCPM_options)

    mask = cpm_source.residuals_mask
    if 'mask_model_epochs' in MCPM_options:
        mask *= utils.mask_nearest_epochs(
            cpm_source.pixel_time+2450000., MCPM_options['mask_model_epochs'])
    sat_time = cpm_source.pixel_time[mask] + 2450000.
    # sat_sigma = sat_time * 0. + MCPM_options['sat_sigma']
    sat_sigma = np.sqrt(np.sum(
        np.array([err[mask] for err in cpm_source.pixel_flux_err])**2, axis=0))
    if 'sat_sigma_scale' in MCPM_options:
        sat_sigma *= MCPM_options['sat_sigma_scale']
    data = MM.MulensData(
        [sat_time, 0.*sat_time, sat_sigma],
        phot_fmt='flux', ephemerides_file=MCPM_options['ephemeris_file'],
        bandpass="K2")
    datasets.append(data)

# initiate event and minimizer
event = MM.Event(datasets=datasets, model=model)
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
        minimizer.add_color_constraint(
            ref_dataset, ref_mag,
            MCPM_options[cc][-2], MCPM_options[cc][-1])
    elif len(MCPM_options[cc]) in [5, 6, 7, 8]:
        minimizer.add_full_color_constraint(
            ref_dataset,
            files.index(MCPM_options[cc][1]), files.index(MCPM_options[cc][2]),
            *MCPM_options[cc][3:])
    else:
        raise ValueError('wrong size of "color_constraint" option')
key = 'min_blending_flux'
if key in other_constraints:
    index = files.index(other_constraints[key][0])
    other_constraints[key] = [datasets[index], other_constraints[key][1]]
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

# MN fit:
dir_out = os.path.dirname(MN_args['outputfiles_basename'])
if not os.path.exists(dir_out):
    os.mkdir(dir_out)
minimizer.set_MN_cube(mn_min, mn_max)
# HERE
MN_args['resume'] = True
import redirect_stdout
#with open(os.devnull, 'w') as null:
with redirect_stdout.stdout_redirect_2():
        result = solve(
            LogLikelihood=minimizer.ln_like,
            Prior=minimizer.transform_MN_cube,
            **MN_args)
# XXX minimizer.close_file_all_models()

# Analyze output:
analyzer = Analyzer(
    n_params=MN_args['n_dims'],
    outputfiles_basename=MN_args['outputfiles_basename'], verbose=False)
stats = analyzer.get_stats()
print("=====")
print()
print("Number of modes found: {:}".format(len(stats['modes'])))
msg = "Log-eveidence: {:.4f} +- {:.4f}"
log_evidence = stats['nested sampling global log-evidence']
log_evidence_err = stats['nested sampling global log-evidence error']
print(msg.format(log_evidence, log_evidence_err))
print('parameter values:')
for (name, values) in zip(parameters_to_fit, stats['marginals']):
    median = values['median']
    sigma = (values['1sigma'][1] - values['1sigma'][0]) / 2.
    print('{:10s} : {:.4f} +- {:.4f}'.format(name, median, sigma))
