import os
import configparser
from astropy.coordinates import SkyCoord
from astropy import units as u


# Contains:
#  read_general_options()
#  read_MultiNest_options()
#  read_EMCEE_options()
#  read_MCPM_options()
#  read_models()

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
    methods = None
    if config.has_option(section, 'methods'):
        methods = config.get(section, 'methods').split()
        for i in range(0, len(methods), 2):
            try:
                methods[i] = float(methods[i])
            except ValueError:
                print("Parsing methods failed - expected float, got ", 
                    methods[i])
                raise
    file_all_models = None
    if config.has_option(section, 'file_all_models'):
        file_all_models = config.get(section, 'file_all_models')

    # data files:
    section = 'file_names'
    if section not in config:
        files = None
        files_formats = None
    else:
        info = [[var, config.get(section, var).split()] for var in config[section]]
        for info_ in info:
            if len(info_[1]) != 2:
                msg = ('Wrong input in cfg file:\n{:}\nFiles require ' +
                    'additional parameter: "mag" or "flux"')
                raise ValueError(msg.format(config.get(section, info_[0])))
        files = [x[1][0] for x in info]
        files_formats = [x[1][1] for x in info]
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
            parameters_fixed)
    return out

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
        starting_mean - list - mean starting values of parameters
        starting_sigma - list - sigma for starting values of parameters
        parameters_to_fit - list - corresponding names of parameters
        min_values - dict - prior minimum values
        max_values - dict - prior maximum values
        emcee_settings - dict - a few parameters need for EMCEE
    """
    # mean and sigma for start
    section = 'EMCEE_starting_mean_sigma'
    info = [[var, config.get(section, var).split()] for var in config[section]]
    for info_ in info:
        if len(info_[1]) != 2:
            raise ValueError('Wrong input in cfg file:\n{:}'.format(config.get(
                    section, info_[0])))
    starting_mean = [float(x[1][0]) for x in info]
    starting_sigma = [float(x[1][1]) for x in info]
    parameters_to_fit = [x[0] for x in info]

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
            emcee_settings[var] = config.getint(section, var)
    if emcee_settings['n_steps'] < emcee_settings['n_burn']:
        raise ValueError(("This doesn't make sense:\nn_steps = {:}\n" +
            "n_burn = {:}").format(emcee_settings['n_steps'], 
            emcee_settings['n_burn']))

    out = (starting_mean, starting_sigma, parameters_to_fit, min_values, 
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

    return mcpm_options

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

