import numpy as np
import os
import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import configparser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# Read models:
out = read_config.read_models(config)
(parameter_values, model_ids, plot_files, txt_files) = out[:4]
(txt_files_prf_phot, txt_models, parameters_to_fit) = out[4:7]
(plot_epochs, plot_epochs_type) = out[7:]
plot_settings = read_config.read_plot_settings(config)

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
    if 'n_select' in MCPM_options:
        cpm_source.select_highest_prf_sum_pixels(MCPM_options['n_select'])

    cpm_sources.append(cpm_source)

# initiate model
model_begin = parameter_values[0]
parameters = {key: value for (key, value) in zip(parameters_to_fit, model_begin)}
parameters.update(parameters_fixed)
parameters_ = {**parameters}
for param in list(parameters_.keys()).copy():
    if (param == 'f_s_sat' or param[:3] == 'q_f' or param[:7] == 'log_q_f'):
        parameters_.pop(param)
model = MM.Model(parameters_, coords=coords)
#for band in {d.bandpass for d in datasets}:
#    model.set_limb_coeff_gamma(band, 0.)
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
                times, separate=True)[0] # This is very simple solution.
    cpm_source.run_cpm(parameters['f_s_sat'] * model_magnification)

    utils.apply_limit_time(cpm_source, MCPM_options)

    mask = cpm_source.residuals_mask
    if 'mask_model_epochs' in MCPM_options:
        mask *= utils.mask_nearest_epochs(cpm_source.pixel_time+2450000., MCPM_options['mask_model_epochs'])
    sat_time = cpm_source.pixel_time[mask] + 2450000.
    #sat_sigma = sat_time * 0. + MCPM_options['sat_sigma']
    sat_sigma = np.sqrt(np.sum(np.array([err[mask] for err in cpm_source.pixel_flux_err])**2, axis=0))
    if 'sat_sigma_scale' in MCPM_options:
        sat_sigma *= MCPM_options['sat_sigma_scale']
    data = MM.MulensData([sat_time, 0.*sat_time, sat_sigma],
            phot_fmt='flux', ephemerides_file=MCPM_options['ephemeris_file'],
            bandpass="K2", coords=coords)
    datasets.append(data)

# initiate event
event = MM.Event(datasets=datasets, model=model)
params = parameters_to_fit[:]
minimizer = Minimizer(event, params, cpm_sources)
if 'coeffs_fits_in' in MCPM_options:
    minimizer.read_coeffs_from_fits(MCPM_options['coeffs_fits_in'])
if 'coeffs_fits_out' in MCPM_options:
    raise ValueError("coeffs_fits_out cannot be set in this program")
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

key = 'no_blending_files'
if key in MCPM_options:
    indexes = [files.index(f) for f in MCPM_options['no_blending_files']]
    for ind in indexes:
        minimizer.fit_blending[ind] = False

if 'mask_model_epochs' in MCPM_options:
    minimizer.model_masks[0] = utils.mask_nearest_epochs(
        cpm_sources[0].pixel_time+2450000., MCPM_options['mask_model_epochs'])

# main loop:
zipped = zip(parameter_values, model_ids, plot_files, txt_files, 
             txt_files_prf_phot, txt_models, plot_epochs)
for zip_single in zipped:
    (values, name, plot_file) = zip_single[:3]
    (txt_file, txt_file_prf_phot, txt_model, plot_epochs_) = zip_single[3:]
    minimizer.set_parameters(values)
    chi2 = minimizer.chi2_fun(values)
    print(name, chi2)
    minimizer.set_satellite_data(values)

    if txt_file_prf_phot is not None:
        (y, y_mask) = cpm_source.prf_photometry()
        x = cpm_source.pixel_time[y_mask]
        err = cpm_source.all_pixels_flux_err * MCPM_options['sat_sigma_scale']
        np.savetxt(txt_file_prf_phot, np.array([x, y[y_mask], err[y_mask]]).T)
    if txt_file is not None:
        y_mask = cpm_source.residuals_mask
        x = cpm_source.pixel_time[y_mask]
        y = minimizer.event.datasets[-1].flux
        y_err = cpm_source.all_pixels_flux_err[y_mask]
        y_err *= MCPM_options['sat_sigma_scale']
        y_model = minimizer._sat_models[0][y_mask]  # XXX we should not use private property
        np.savetxt(txt_file, np.array([x, y, y_err, y-y_model]).T)
    if txt_model is not None:
        y_mask = cpm_source.residuals_mask
        x = cpm_source.pixel_time[y_mask]
        y_model = minimizer._sat_models[0][y_mask]  # XXX we should not use private property
        np.savetxt(txt_model, np.array([x, y_model]).T)
    if plot_file is not None:
        minimizer.set_satellite_data(values)
        if 'xlim' in plot_settings:
            (t_beg, t_end) = plot_settings.pop('xlim')
        elif 91 in MCPM_options['campaigns'] and 92 in MCPM_options['campaigns']:
            (t_beg, t_end) = (7500.3, 7573.5)
        elif 91 in MCPM_options['campaigns']:
            (t_beg, t_end) = (7500.3, 7528.0)
        elif 92 in MCPM_options['campaigns']:
            (t_beg, t_end) = (7530., 7573.5)
        else:
            (t_beg, t_end) = (7425., 7670.)
        ylim = plot_settings.pop('ylim', None)
        ylim_residuals = plot_settings.pop('ylim_residuals', None)
        adjust = dict(left=0.09, right=0.995, bottom=0.08, top=0.995)
        if len(plot_settings) == 0:
            minimizer.very_standard_plot(t_beg, t_end, ylim, title=name)
        else:
            minimizer.standard_plot(t_beg, t_end, ylim, title=name,
                                    **plot_settings)
            if 'fluxes_y_axis' in plot_settings:
                adjust['right'] = 0.895
        if ylim_residuals is not None:
            plt.ylim(*ylim_residuals)
            if ylim_residuals[0] > 0.1 and -ylim_residuals[1] > 0.1:
                fmt = ticker.FormatStrFormatter('%0.1f')
                plt.gca().yaxis.set_major_formatter(fmt)
        plt.xlabel("BJD-2450000")
        plt.subplots_adjust(**adjust)
        if len(plot_file) == 0:
            plt.show()
        else:
            plt.savefig(plot_file, dpi=400)
            print("{:} file saved".format(plot_file))
        plt.close()
    if plot_epochs_ is not None:
        if minimizer.n_sat != 1:
            raise ValueError('.n_sat != 1 is not implemented')
        for epoch in plot_epochs_:
            args = [epoch - 2450000.]
            if plot_epochs_type is not None:
                args.append(plot_epochs_type)
            minimizer.cpm_sources[0].plot_image(*args)
            plt.show()
    if len(datasets) > 1:
        print("Non-K2 datasets (i, chi2, F_s, F_b):")
        for (i, (dat, fb)) in enumerate(zip(datasets, minimizer.fit_blending)):
            chi2_data = event.get_chi2_for_dataset(i, fit_blending=fb)
            print(i, chi2_data, event.fit.flux_of_sources(dat)[0],
                  event.fit.blending_flux(dat))
            if i < len(files):
                chi2 -= chi2_data
        print("-----")
        print("K2 chi2: ", chi2)
    if len(cpm_sources) > 0:
        print("Satellite t_0, u_0, A_max:")
        print(*minimizer.satellite_maximum())
    print()

