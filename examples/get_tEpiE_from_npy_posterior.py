"""
Read the .npy file with posterior, extract columns contain tE and parallax,
bin those and print to text file.
"""
import sys
import os
import numpy as np
import configparser

import read_config


if len(sys.argv) != 2:
    raise ValueError('Exactly one argument needed - cfg file')
config_file = sys.argv[1]

parameters = ['t_E', 'pi_E_N', 'pi_E_E']

config = configparser.ConfigParser()
config.optionxform = str
config.read(config_file)
read_config.check_sections_in_config(config)

MCPM_options = read_config.read_MCPM_options(config, check_fits_files=False)

out = read_config.read_EMCEE_options(config)
parameters_to_fit = out[1]
emcee_settings = out[4]
if 'file_posterior' not in emcee_settings:
    raise ValueError('file_posterior missing in EMCEE settings')
if len(emcee_settings['file_posterior']) == 0:
    config_file_root = os.path.splitext(config_file)[0]
    emcee_settings['file_posterior'] = config_file_root + ".posterior.npy"

data = np.load(emcee_settings['file_posterior'])

indexes = [parameters_to_fit.index(p) for p in parameters]

selected = np.array([data[:, index] for index in indexes]).T

(unique, counts) = np.unique(selected, return_counts=True, axis=0)

for (u, c) in zip(unique, counts):
    print(*u.tolist(), c)

