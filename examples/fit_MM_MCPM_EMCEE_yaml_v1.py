"""
A simple script for running fit_MM_MCPM_EMCEE_v1.py using input from
a YAML file.
"""
import yaml
import sys
import os
from astropy.coordinates import SkyCoord
from astropy import units as u

from fit_MM_MCPM_EMCEE_v1 import fit_MM_MCPM_EMCEE


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - yaml file')

    file_in = sys.argv[1]

    with open(file_in) as in_file:
        settings = yaml.safe_load(in_file)

    settings['parameters_to_fit'] = [*settings['starting_settings']]
    ra = settings.pop("RA")
    dec = settings.pop("Dec")
    settings['skycoord'] = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    settings['config_file_root'] = os.path.splitext(file_in)[0]
    settings['emcee_settings']['n_temps'] = 1
    settings['emcee_settings']['PTSampler'] = False

    fit_MM_MCPM_EMCEE(**settings)

