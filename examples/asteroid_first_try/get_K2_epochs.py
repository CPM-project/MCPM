"""
extract time vector for K2
"""
import sys
import yaml
import numpy as np

import K2fov

from MCPM.cpmfitsource import CpmFitSource


def get_time_mask(ra, dec, campaign):
    """for given ra & dec (deg) and campaign extract time vector and mask"""
    campaign_id = campaign
    if campaign > 20:
        campaign_id = int(campaign / 10.)

    field_info = K2fov.fields.getFieldInfo(campaign_id)
    fovRoll_deg = K2fov.fov.getFovAngleFromSpacecraftRoll(field_info["roll"])
    field = K2fov.fov.KeplerFov(
        field_info["ra"], field_info["dec"], fovRoll_deg)
    channel = int(field.pickAChannel(ra, dec) + 0.5)

    cpm_source = CpmFitSource(ra=ra, dec=dec,
                              campaign=campaign, channel=channel)

    cpm_source.set_pixels_square(0)

    mask = cpm_source.pixel_mask[0]
    time = cpm_source.pixel_time

    return (time, mask)


def write_time(ra, dec, campaign, file_out, apply_mask=False):
    """dump the time vector"""
    (time, mask) = get_time_mask(ra=ra, dec=dec, campaign=campaign)

    if apply_mask:
        time = time[mask]

    time += 2450000.

    time = time[np.isfinite(time)]

    np.savetxt(file_out, time.T, fmt='%.5f')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('single YAML file needed as inline argument')

    with open(sys.argv[1], 'r') as data:
        settings = yaml.safe_load(data)

    write_time(**settings)
