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
from MCPM.pixellensingmodel import PixelLensingModel
from MCPM.pixellensingevent import PixelLensingEvent

import read_config


def fit_MM_MCPM_EMCEE(
        files, files_formats, files_kwargs, skycoord, methods, MCPM_options,
        starting_settings, parameters_to_fit, parameters_fixed,
        min_values, max_values, emcee_settings, other_constraints,
        file_all_models, data_add_245=True):
    """
    Fit the microlensing (MulensModel) and K2 photometry (MCPM) using
    EMCEE method. The input is complicated - please see code below and
    in read_config.py to find out details.
    """
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
            data = MM.MulensData(file_name=file_, add_2450000=data_add_245,
                                 phot_fmt=fmt, coords=coords, **kwargs)
            datasets.append(data)

    # satellite datasets
    cpm_sources = []
    for campaign in MCPM_options['campaigns']:
        cpm_source = CpmFitSource(
            ra=skycoord.ra.deg, dec=skycoord.dec.deg,
            campaign=campaign, channel=MCPM_options['channel'])
        cpm_source.get_predictor_matrix(**MCPM_options['predictor_matrix'])
        cpm_source.set_l2_l2_per_pixel(
            l2=MCPM_options['l2'], l2_per_pixel=MCPM_options['l2_per_pixel'])
        cpm_source.set_pixels_square(MCPM_options['half_size'])
        cpm_source.select_highest_prf_sum_pixels(MCPM_options['n_select'])

        cpm_sources.append(cpm_source)

    # initiate model
    starting = utils.generate_random_points(
        starting_settings, parameters_to_fit, emcee_settings['n_walkers'])
    zip_ = zip(parameters_to_fit, starting[0])
    parameters = {key: value for (key, value) in zip_}
    parameters.update(parameters_fixed)
    parameters_ = {**parameters}
    for param in list(parameters_.keys()).copy():
        if param == 'f_s_sat' or param[:3] == 'q_f' or param[:7] == 'log_q_f':
            parameters_.pop(param)
    try:
        model = MM.Model(parameters_, coords=coords)
    except KeyError:
        model = PixelLensingModel(parameters_, coords=coords)
    for (m_key, m_value) in methods.items():
        model.set_magnification_methods(m_value, m_key)

    for cpm_source in cpm_sources:
        times = cpm_source.pixel_time + 2450000.
        times[np.isnan(times)] = np.mean(times[~np.isnan(times)])
        if model.n_sources == 1:
            if isinstance(model, MM.Model):
                model_flux = ((model.magnification(times)-1.) *
                              parameters['f_s_sat'])
            else:
                model_flux = model.flux_difference(times)
        else:
            if not isinstance(model, MM.Model):
                raise NotImplementedError('not yet coded for pixel lensing')
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
            model_flux = (model_magnification - 1.) * parameters['f_s_sat']
        cpm_source.run_cpm(model_flux)

        utils.apply_limit_time(cpm_source, MCPM_options)

        mask = cpm_source.residuals_mask
        if 'mask_model_epochs' in MCPM_options:
            mask *= utils.mask_nearest_epochs(
                cpm_source.pixel_time+2450000.,
                MCPM_options['mask_model_epochs'])
        sat_time = cpm_source.pixel_time[mask] + 2450000.
        # sat_sigma = sat_time * 0. + MCPM_options['sat_sigma']
        sat_sigma = np.sqrt(np.sum(
            np.array([err[mask] for err in cpm_source.pixel_flux_err])**2,
            axis=0))
        if 'sat_sigma_scale' in MCPM_options:
            sat_sigma *= MCPM_options['sat_sigma_scale']
        data = MM.MulensData(
            [sat_time, 0.*sat_time, sat_sigma],
            phot_fmt='flux', ephemerides_file=MCPM_options['ephemeris_file'],
            bandpass="K2")
        datasets.append(data)

    # initiate event and minimizer
    if isinstance(model, MM.Model):
        event = MM.Event(datasets=datasets, model=model)
    else:
        event = PixelLensingEvent(datasets=datasets, model=model)
    params = parameters_to_fit[:]
    minimizer = Minimizer(event, params, cpm_sources)
    minimizer.file_all_models = file_all_models
    minimizer.set_chi2_0()
    if 'f_s_sat' in parameters_fixed:
        minimizer.set_satellite_source_flux(parameters_fixed['f_s_sat'])
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
                files.index(MCPM_options[cc][1]),
                files.index(MCPM_options[cc][2]),
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

    # EMCEE fit:
    print("EMCEE walkers, steps, burn: {:} {:} {:}".format(
        emcee_settings['n_walkers'], emcee_settings['n_steps'],
        emcee_settings['n_burn']))
    minimizer.set_prior_boundaries(min_values, max_values)
    for start_ in starting:
        if minimizer.ln_prior(start_) <= -float('inf'):
            raise ValueError('starting point is not in prior:\n' + str(start_))
    sampler = emcee.EnsembleSampler(
        emcee_settings['n_walkers'], n_params, minimizer.ln_prob)
    acceptance_fractions = []
    # run:
    # sampler.run_mcmc(starting, emcee_settings['n_steps'])
    for _ in sampler.sample(starting, iterations=emcee_settings['n_steps']):
        acceptance_fractions.append(np.mean(sampler.acceptance_fraction))

    # cleanup and close minimizer:
    out_name = emcee_settings.get('file_acceptance_fractions', None)
    if out_name is not None:
        if len(out_name) == 0:
            out_name = config_file_root + ".accept"
        data_save = [str(i+1) + " " + str(af)
                     for (i, af) in enumerate(acceptance_fractions)]
        with open(out_name, 'w') as file_out:
            file_out.write('\n'.join(data_save))
    n_burn = emcee_settings['n_burn']
    samples = sampler.chain[:, n_burn:, :].reshape((-1, n_params))
    blob_sampler = np.transpose(np.array(sampler.blobs), axes=(1, 0, 2))
    n_fluxes = blob_sampler.shape[-1]
    if 'coeffs_fits_out' in MCPM_options:
        minimizer.set_pixel_coeffs_from_samples(samples)
        minimizer.save_coeffs_to_fits(MCPM_options['coeffs_fits_out'])
        minimizer.stop_coeffs_cache()
    minimizer.close_file_all_models()

    # output
    print("Mean acceptance fraction: {:.4f} +- {:.4f}".format(
        np.mean(sampler.acceptance_fraction),
        np.std(sampler.acceptance_fraction)))
    zip_ = zip(*np.percentile(samples, [16, 50, 84], axis=0))
    results = map(lambda v: (v[1], v[2]-v[1], v[0]-v[1]), zip_)
    for (param, r) in zip(parameters_to_fit, results):
        print('{:7s} : {:.4f} {:+.4f} {:+.4f}'.format(param, *r))
    if n_fluxes > 0:
        blob_samples = blob_sampler[:, n_burn:, :].reshape((-1, n_fluxes))
        percentiles = np.percentile(blob_samples, [16, 50, 84], axis=0)
        blob_results = map(
            lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*percentiles))
        msg = 'flux_{:}_{:} : {:.4f} {:.4f} {:.4f}'
        for (i, r) in zip(range(n_fluxes), blob_results):
            print(msg.format(['S', 'B'][i % 2], i//2+1, *r))
    if 'file_posterior' in emcee_settings:
        if len(emcee_settings['file_posterior']) == 0:
            emcee_settings['file_posterior'] = config_file_root + ".posterior"
        all_samples = np.concatenate((samples, blob_samples), axis=1)
        np.save(emcee_settings['file_posterior'], all_samples)
    print('Best model:')
    minimizer.print_min_chi2()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - cfg file')
    config_file = sys.argv[1]

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)
    read_config.check_sections_in_config(config)
    print('Configuration file:', config_file)

    # Read general options:
    out = read_config.read_general_options(config)
    (skycoord, methods, file_all_models) = out[:3]
    (files, files_formats, files_kwargs, parameters_fixed) = out[3:]
    # Read EMCEE options:
    out = read_config.read_EMCEE_options(config)
    (starting_settings, parameters_to_fit) = out[:2]
    (min_values, max_values, emcee_settings) = out[2:]
    # Read MCPM options and other constraints:
    MCPM_options = read_config.read_MCPM_options(config)
    other_constraints = read_config.read_other_constraints(config)

    # Main function:
    fit_MM_MCPM_EMCEE(
        files, files_formats, files_kwargs, skycoord, methods, MCPM_options,
        starting_settings, parameters_to_fit, parameters_fixed,
        min_values, max_values, emcee_settings, other_constraints,
        file_all_models)
