import os
import warnings
import configparser
from astropy.coordinates import SkyCoord
from astropy import units as u


# Contains:
#  check_sections_in_config()
#  read_general_options()
#  read_MultiNest_options()
#  read_EMCEE_options()
#  read_MCPM_options()
#  read_other_constraints()
#  read_models()
#  read_plot_settings()

def check_sections_in_config(config):
    """
    Check names of the sections - if there is one that is not parsed by this
    file, then warning is raised.

    Parameters:
        config - configparser.ConfigParser instance
    """
    allowed = [
        'event', 'file_names', 'parameters_fixed', 'MultiNest_ranges',
        'MultiNest', 'EMCEE_starting_mean_sigma', 'EMCEE_min_values',
        'EMCEE_max_values', 'EMCEE_settings', 'MCPM', 'other_constraints',
        'models_1_line', 'plot_files', 'txt_files', 'txt_files_prf_phot',
        'txt_models', 'plot_settings']
    difference = set(config.sections()) - set(allowed)
    if len(difference) > 0:
        txt = ("\nThere are unexpected sections in config file (they will " +
               "be ignored):\n\n{:}\n".format(difference))
        warnings.warn(txt, UserWarning)

def read_general_options(config):
    """
    parses general information about the event
    
    Parameters:
        config - configparser.ConfigParser instance
    
    Returns:
        out_skycoord - SkyCoord or None - event coordinates
        methods - list or None - floats and str that specify which method
            to use when
        file_all_models - str or None - name of file where all calculated
            models will be printed
        files - list - list of names of files to be read
        files_formats - list - list of "mag" or "flux"
        parameters_fixed - dict - parameters to be kept fixed during fitting
    """
    if not isinstance(config, configparser.ConfigParser):
        raise TypeError('read_general_options() option of wrong type')
    
    # General event properties:
    section = 'event'
    try:
        ra = config.getfloat(section, 'RA')
        dec = config.getfloat(section, 'Dec')
        out_skycoord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    except:
        out_skycoord = None
    file_all_models = None
    if config.has_option(section, 'file_all_models'):
        file_all_models = config.get(section, 'file_all_models')

    # Methods:
    keys = {'methods': None, 'methods_1': 1, 'methods_2': 2}
    methods = {}
    for (key, v) in keys.items():
        if config.has_option(section, key):
            methods[v] = _parse_methods(config.get(section, key).split())

    # data files:
    section = 'file_names'
    if section not in config:
        files = None
        files_formats = None
        files_kwargs = None
    else:
        info = [[var, config.get(section, var).split()] for var in config[section]]
        for info_ in info:
            if len(info_[1]) not in [2, 4]:
                msg = ('Wrong input in cfg file:\n{:}\nFiles require ' +
                    'additional parameter: "mag" or "flux"')
                raise ValueError(msg.format(config.get(section, info_[0])))
        files = [x[1][0] for x in info]
        files_formats = [x[1][1] for x in info]
        files_kwargs = [{}] * len(files)
        plot_properties_keys = ['label', 'color', 'show_errorbars', 'show_bad',
                                'marker', 'fmt', 'markersize', 'ms', 's']
        for i in range(len(files)):
            if len(info[i][1]) == 4:
                if info[i][1][2] in plot_properties_keys:
                    files_kwargs[i] = {'plot_properties': {info[i][1][2]: info[i][1][3]} }
                else:
                    files_kwargs[i] = {info[i][1][2]: info[i][1][3]}
        formats = set(files_formats)-set(["mag", "flux"])
        if len(formats) > 0:
            raise ValueError('wrong file formats: {:}'.format(formats))

    # fixed parameters:
    section = 'parameters_fixed'
    parameters_fixed = {}
    if section in config.sections():
        for var in config[section]:
            parameters_fixed[var] = config.getfloat(section, var)

    out = (out_skycoord, methods, file_all_models, files, files_formats, 
            files_kwargs, parameters_fixed)
    return out

def _parse_methods(methods):
    """
    change odd elements of the list to floats
    """
    for i in range(0, len(methods), 2):
        try:
            methods[i] = float(methods[i])
        except ValueError:
            print("Parsing methods failed - expected float, got ", 
                methods[i])
            raise
    return methods

def read_MultiNest_options(config, config_file, dir_out="chains"):
    """
    parses MultiNest options
    
    Parameters:
        config - configparser.ConfigParser instance
        config_file - str - name of the the cfg file used to for default
            file name root
        dir_out - str - default value of output directory
        
    Returns:
        ranges_min - list - minimal values of parameters
        ranges_max - list - maximal values of parameters
        parameters_to_fit - list - corresponding names of parameters
        MN_args - dict - arguments to be passed to MultiNest
    """
    # MultiNest ranges:
    section = 'MultiNest_ranges'
    info = [[var, config.get(section, var).split()] for var in config[section]]
    for info_ in info:
        if len(info_[1]) != 2:
            raise ValueError('Wrong input in cfg file:\n{:}'.format(
                    config.get(section, info_[0])))
    ranges_min = [float(x[1][0]) for x in info]
    ranges_max = [float(x[1][1]) for x in info]
    parameters_to_fit = [x[0] for x in info]
    
    # MultiNest settings:
    section = 'MultiNest'
    dir_out = "chains"
    split_ = os.path.splitext(config_file)
    MN_args = {
        'n_dims': len(parameters_to_fit),
        'importance_nested_sampling': False, 
        'multimodal': True,
        'n_live_points': 500,
        'outputfiles_basename': os.path.join(dir_out, split_[0] + "_"), 
        'resume': False}
    if section in config.sections():
        int_variables = ["n_params", "n_clustering_params", "n_live_points", 
                        "seed", "max_iter"]
        for var in config[section]:
            if var in int_variables:
                value = config.getint(section, var)
            elif var in ["evidence_tolerance"]:
                value = config.getfloat(section, var)
            elif var in ["outputfiles_basename"]:
                value = config.get(section, var)
            else:
                raise ValueError(('unrecognized MultiNest parameter: {:}\n' + 
                    '(note that some parameters are not included in ' + 
                    'the parser yet)').format(var))
            MN_args[var] = value

    return (ranges_min, ranges_max, parameters_to_fit, MN_args)

def read_EMCEE_options(config):
    """
    parses EMCEE options
    
    Parameters:
        config - configparser.ConfigParser instance
        
    Returns:
        starting - dict - specifies PDFs for starting values of parameters
        parameters_to_fit - list - corresponding names of parameters
        min_values - dict - prior minimum values
        max_values - dict - prior maximum values
        emcee_settings - dict - a few parameters need for EMCEE
    """
    # mean and sigma for start
    section = 'EMCEE_starting_mean_sigma'
    parameters_to_fit = [var for var in config[section]]
    starting = {}
    for param in parameters_to_fit:
        words = config.get(section, param).split()
        if len(words) < 2:
            msg = 'Wrong input in cfg file:\n{:}'
            raise ValueError(msg.format(config.get(section, param)))
        if len(words) == 2:
            words.append('gauss')
        starting[param] = [float(words[0]), float(words[1])] + words[2:]

    # prior min an max values
    min_values = {}
    section = 'EMCEE_min_values'
    if section in config.sections():
        for var in config[section]:
            min_values[var] = config.getfloat(section, var)

    max_values = {}
    section = 'EMCEE_max_values'
    if section in config.sections():
        for var in config[section]:
            max_values[var] = config.getfloat(section, var)

    # EMCEE settings
    emcee_settings = {
        'n_walkers': 4 * len(parameters_to_fit),
        'n_steps': 1000,
        'n_burn': 50}
    section = 'EMCEE_settings'
    if section in config.sections():
        for var in config[section]:
            if var in ['file_acceptance_fractions', 'file_posterior']:
                emcee_settings[var] = config.get(section, var)
                if os.path.isfile(emcee_settings[var]):
                    raise FileExistsError(emcee_settings[var])
            else:
                emcee_settings[var] = config.getint(section, var)
    if emcee_settings['n_steps'] < emcee_settings['n_burn']:
        raise ValueError(("This doesn't make sense:\nn_steps = {:}\n" +
            "n_burn = {:}").format(emcee_settings['n_steps'], 
            emcee_settings['n_burn']))

    out = (starting, parameters_to_fit, min_values,
            max_values, emcee_settings)
    return out

def read_MCPM_options(config):
    """
    parses MCPM options

    Parameters:
        config - configparser.ConfigParser instance

    Returns:
        mcpm_options - dict - gives all the options
    """
    section = 'MCPM'

    if section not in config.sections():
        return dict(campaigns=[])

    mcpm_options = {'half_size': 2, 'l2_per_pixel': None, 'l2': None}

    mcpm_options['ephemeris_file'] = config.get(section, 'ephemeris_file')
    if 'model_file' in config[section]:
        mcpm_options['model_file'] = config.get(section, 'model_file')
    mcpm_options['channel'] = config.getint(section, 'channel')
    mcpm_options['campaigns'] = [
            int(var) for var in config.get(section, 'campaigns').split()]
    if 'half_size' in config[section]:
        mcpm_options['half_size'] = config.getint(section, 'half_size')
    mcpm_options['n_select'] = config.getint(section, 'n_select')
    if 'sat_sigma' in config[section]:
        raise KeyError("Upppsss... It seems you're trying to used old config "
            + "with new code. This version uses 'sat_sigma_scale', not "
            + "'sat_sigma'")
    if 'sat_sigma_scale' in config[section]:
        mcpm_options['sat_sigma_scale'] = config.getfloat(section, 'sat_sigma_scale')

    if 'l2_per_pixel' in config[section] and 'l2' in config[section]:
        raise ValueError('l2 and l2_per_pixel cannot both be set')
    if 'l2_per_pixel' in config[section]:
        mcpm_options['l2_per_pixel'] = config.getfloat(section, 'l2_per_pixel')
    elif 'l2' in config[section]:
        mcpm_options['l2'] = config.getfloat(section, 'l2')
    else:
        raise ValueError('l2 or l2_per_pixel must be set')
    if 'train_mask_time_limit' in config[section]:
        raise KeyError("Upppsss... It seems you're trying to used old config "
            + "with new code. Try train_mask_begin or train_mask_end instead "
            + "of train_mask_time_limit")
    if 'train_mask_begin' in config[section]:
        mcpm_options['train_mask_begin'] = config.getfloat(section, 
            'train_mask_begin')
    if 'train_mask_end' in config[section]:
        mcpm_options['train_mask_end'] = config.getfloat(section, 
            'train_mask_end')
    if 'mask_model_epochs' in config[section]:
        mcpm_options['mask_model_epochs'] = [float(txt) 
                for txt in config.get(section, 'mask_model_epochs').split()]

    predictor = {}
    if 'n_pix_predictor_matrix' in config[section]:
        predictor['n_pixel'] = config.getint(section, 'n_pix_predictor_matrix')
    key = 'n_pca_components'
    if key in config[section]:
        predictor[key] = config.getint(section, key)
    key = 'selected_pixels_file'
    if key in config[section]:
        predictor[key] = config.get(section, key)
    mcpm_options['predictor_matrix'] = predictor
    
    if 'color_constraint' in config[section]:
        tt = config.get(section, 'color_constraint').split()
        if len(tt) in [3, 4]:
            mcpm_options['color_constraint'] = [tt[0]] + [float(t) for t in tt[1:]]
        elif len(tt) == 7:
            mcpm_options['color_constraint'] = tt[:3] + [[float(t) for t in tt[3:6]]] + [float(t) for t in tt[6:]]
        else:
            raise ValueError(
                'Wrong length of color_constraint in MCPM section of config')
    if 'magnitude_constraint' in config[section]:
        tt = config.get(section, 'magnitude_constraint').split()
        if len(tt) == 2:
            mcpm_options['magnitude_constraint'] = [float(tt[0]), float(tt[1])]
        else:
            raise ValueError('Wrong length of magnitude_constraint in ' +
                             'MCPM section of config')

    if 'coeffs_fits_out' in config[section] and 'coeffs_fits_in' in config[section]:
        raise ValueError('coeffs_fits_out and coeffs_fits_in cannot both be set')
    if 'coeffs_fits_out' in config[section]:
        mcpm_options['coeffs_fits_out'] = config.get(section, 'coeffs_fits_out').split()
        if len(mcpm_options['coeffs_fits_out']) != len(mcpm_options['campaigns']):
            raise ValueError('incompatible length of coeffs_fits_out and campaigns')
        for file_name in mcpm_options['coeffs_fits_out']:
            if os.path.isfile(file_name):
                raise ValueError('file {:} already exists'.format(file_name))

    if 'coeffs_fits_in' in config[section]:
        mcpm_options['coeffs_fits_in'] = config.get(section, 'coeffs_fits_in').split()
        if len(mcpm_options['coeffs_fits_in']) != len(mcpm_options['campaigns']):
            raise ValueError('incompatible length of coeffs_fits_in and campaigns')
        for file_name in mcpm_options['coeffs_fits_in']:
            if not os.path.isfile(file_name):
                raise ValueError('file {:} does not exist'.format(file_name))

    if 'no_blending_files' in config[section]:
        warnings.warn("Option no_blending_files may not work currently - check Minimizer.chi2_fun()")
        mcpm_options['no_blending_files'] = config.get(section, 'no_blending_files').split()

    return mcpm_options

def read_other_constraints(config):
    """
    parses more complicated constrains

    Parameters:
        config - configparser.ConfigParser instance

    Returns:
       options - dict - specifies constraints
    """
    section = 'other_constraints'
    options = dict()

    if section not in config.sections():
        return options

    for var in config[section]:
        if var == 't_0':
            value = config.get(section, var)
            if value == '1 < 2':
                options['t_0'] = 't_0_1 < t_0_2'
            elif value == '1 > 2':
                options['t_0'] = 't_0_1 > t_0_2'
            else:
                raise ValueError('unknown value: ' + value)
        elif var == 'min_blending_flux':
            value = config.get(section, var).split()
            if len(value) != 2:
                raise ValueError('wrong keyword length: {:}'.format(value))
            options[var] = [value[0], float(value[1])]
        else:
            raise KeyError('unregognized keyword: ' + var)
    return options

def read_models(config):
    """
    parses a set of models

    Parameters:
        config - configparser.ConfigParser instance

    Returns:
        parameter_values - list - list of lists of floats
        model_ids - list - names of models
        plot_files - list - names of files where to make plots or None
        txt_files - list - names of files where data will be saved
        txt_files_prf_phot - list - names of files where PRF-like photometry 
                                    will be saved
        txt_models - list - names of files where model flxues will be saved
        parameters_to_fit - list - corresponding names of parameters
    """
    parameter_values = []
    model_ids = []
    parameters_to_fit = []

    section = 'models_1_line'
    if section in config.sections():
        for var in config[section]:
            if var == "parameters":
                parameters_to_fit = config.get(section, var).split()
            else:
                model_ids.append(var)
                split_ = config.get(section, var).split()
                parameter_values.append([float(value) for value in split_])
        for model in parameter_values:
            if len(model) != len(parameters_to_fit):
                msg = ('error in reading models: got {:} parameters instead' +
                    'of {:}\n').format(len(model), len(parameters_to_fit))
                raise ValueError(msg, " ".join([str(x) for x in model]))

    section = 'plot_files'
    plot_files = [None] * len(model_ids)
    if section in config.sections():
        for var in config[section]:
            plot_files[model_ids.index(var)] = config.get(section, var)

    section = 'txt_files'
    txt_files = [None] * len(model_ids)
    if section in config.sections():
        for var in config[section]:
            txt_files[model_ids.index(var)] = config.get(section, var)

    section = 'txt_files_prf_phot'
    txt_files_prf_phot = [None] * len(model_ids)
    if section in config.sections():
        for var in config[section]:
            txt_files_prf_phot[model_ids.index(var)] = config.get(section, var)

    section = 'txt_models'
    txt_models = [None] * len(model_ids)
    if section in config.sections():
        for var in config[section]:
            txt_models[model_ids.index(var)] = config.get(section, var)

    return (parameter_values, model_ids, plot_files, txt_files, 
            txt_files_prf_phot, txt_models, parameters_to_fit)

def read_plot_settings(config):
    """
    Read settings that govern plotting. Most importantly, we read y axis
    limits here.

    The keys read are: "ylim", "ylim_residuals", "color_list", "label_list",
    "legend_order", "alpha_list", "line_width".

    Parameters:
        config - configparser.ConfigParser instance

    Returns:
        plot_settings: *dict*
            Settings read.
    """
    plot_settings = dict()

    section = 'plot_settings'
    if section not in config.sections():
        return plot_settings

    keys_comma_split = ["label_list"]
    keys_str = ["color_list"] + keys_comma_split
    keys_float = ["alpha_list", "ylim", "ylim_residuals"]
    keys_int = ["legend_order", "fluxes_y_axis"]
    keys = keys_str + keys_float + keys_int
    for key in keys:
        if key in config[section]:
            split = None
            if key in keys_comma_split:
                split = ", "
            plot_settings[key] = config.get(section, key).split(split)
            if key in keys_float:
                plot_settings[key] = [float(val) for val in plot_settings[key]]
            elif key in keys_int:
                plot_settings[key] = [int(val) for val in plot_settings[key]]

    if 'line_width' in config[section]:
        plot_settings['line_width'] = config.getfloat(section, 'line_width')

    return plot_settings

