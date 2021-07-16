"""
A simple script for running evaluate_MM_MCPM_v1.py using input from
a YAML file.
"""
import yaml
import sys
import os
from astropy.coordinates import SkyCoord
from astropy import units as u

from evaluate_MM_MCPM_v1 import evaluate_MM_MCPM


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - yaml file')

    file_in = sys.argv[1]

    with open(file_in) as in_file:
        settings = yaml.safe_load(in_file)

    ra = settings.pop("RA")
    dec = settings.pop("Dec")
    settings['skycoord'] = SkyCoord(ra, dec, unit=(u.deg, u.deg))

    evaluate_MM_MCPM(**settings)

