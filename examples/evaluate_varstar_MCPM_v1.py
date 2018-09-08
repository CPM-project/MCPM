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
from MCPM.minimizervariablestar import MinimizerVariableStar

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
(_, _, _, parameters_fixed) = out[3:]

# Read models:
out = read_config.read_models(config)
(parameter_values, model_ids, plot_files) = out[:3]
(txt_files, txt_files_prf_phot, txt_models, parameters_to_fit) = out[3:]

# Read MCPM options:
MCPM_options = read_config.read_MCPM_options(config)

# End of settings.
###################################################################

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

(model_time, model_value) = np.loadtxt(MCPM_options['model_file'], unpack=True)

for cpm_source in cpm_sources:
    time = cpm_source.pixel_time + 2450000.
    time[np.isnan(time)] = 2457530.
    sat_model = utils.scale_model(parameters['t_0'],
        parameters['width_ratio'], parameters['flux'], 
        time, model_time, model_value)
    cpm_source.run_cpm(sat_model)

    utils.apply_limit_time(cpm_source, MCPM_options)

minimizer = MinimizerVariableStar(parameters_to_fit[:], cpm_sources)
minimizer.model_time = model_time
minimizer.model_value = model_value
if 'coeffs_fits_in' in MCPM_options:
    minimizer.read_coeffs_from_fits(MCPM_options['coeffs_fits_in'])
if 'coeffs_fits_out' in MCPM_options:
    raise ValueError("coeffs_fits_out cannot be set in this program")
if 'magnitude_constraint' in MCPM_options:
    (ref_mag, ref_mag_sigma) = MCPM_options['magnitude_constraint']
    minimizer.add_magnitude_constraint(ref_mag, ref_mag_sigma)
if 'sat_sigma_scale' in MCPM_options:
    minimizer.sigma_scale = MCPM_options['sat_sigma_scale']
minimizer.parameters.update(parameters_fixed)

if 'mask_model_epochs' in MCPM_options:
    minimizer.model_masks[0] = utils.mask_nearest_epochs(
        cpm_sources[0].pixel_time+2450000., MCPM_options['mask_model_epochs'])

# main loop:
zipped = zip(parameter_values, model_ids, plot_files, txt_files,
             txt_files_prf_phot, txt_models)
for zip_single in zipped:
    (values, name, plot_file) = zip_single[:3]
    (txt_file, txt_file_prf_phot, txt_model) = zip_single[3:]

    minimizer.set_parameters(values)

    print(name, minimizer.chi2_fun(values))

    if txt_file_prf_phot is not None:
        (y, y_mask) = cpm_source.prf_photometry()
        x = cpm_source.pixel_time[y_mask]
        np.savetxt(txt_file_prf_phot, np.array([x, y[y_mask]]).T)
            #minimizer.set_satellite_data(values)
            #y = minimizer.event.datasets[-1].flux#[y_mask]    
    if txt_file is not None:
        y_mask = cpm_source.residuals_mask
        x = cpm_source.pixel_time[y_mask]
        y = cpm_source.residuals[y_mask]
        y += utils.scale_model(values[0], values[2], values[1], x+2450000., 
                model_time, model_value)
        np.savetxt(txt_file, np.array([x, y]).T)
    if txt_model is not None:
        y_mask = cpm_source.residuals_mask
        x = cpm_source.pixel_time[y_mask]
        y = minimizer._sat_models[0][y_mask]
        np.savetxt(txt_model, np.array([x, y]).T)
    if plot_file is not None:
        raise NotImplementedError("plot_file")
        # get_satellite_data() instead below; also minimizer.standard_plot() is not implemented
        minimizer.set_satellite_data(values)
        if 91 in MCPM_options['campaigns'] and 92 in MCPM_options['campaigns']:
            (t_beg, t_end) = (7501., 7573.5)
        elif 91 in MCPM_options['campaigns']:
            (t_beg, t_end) = (7501., 7528.5)
        elif 92 in MCPM_options['campaigns']:
            (t_beg, t_end) = (7530., 7573.5)
        else:
            (t_beg, t_end) = (7425., 7670.)
        minimizer.standard_plot(t_beg, t_end, [19., 13.], title=name)
        if len(plot_file) == 0:
            plt.show()
        else:
            plt.savefig(plot_file)
            print("{:} file saved".format(plot_file))
        plt.close()
    print()
