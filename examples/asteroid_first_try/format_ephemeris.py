"""
Cleans ephemeris from points definitely outside superstamp,
interpolates, and extracts exact time stamp.
"""
import numpy as np
import sys

from clean_1 import select_in_superstamp_approx
from get_K2_epochs import get_time_mask
from get_CCR import get_CCR
from interpolate_ephemeris import interpolate_ephemeris


def guess_campaign(time):
    """
    guess campaign (*int*: 91 or 92) based on time vector
    """
    if np.all(time > 2457530.) and np.all(time < 2457575.):
        return 92
    elif np.all(time > 2457500.) and np.all(time < 2457530.):
        return 91
    else:
        raise ValueError('campaign unknown')


def get_index_nearest(array, value):
    """
    find the index of the closest value in an array
    """
    return (np.abs(array - value)).argmin()


def get_in_superstamp_mask(ra, dec):
    """
    XXX
    """
    ok_channels = [30, 31, 49, 52]

    # approximate trimming to superstamp area
    mask_1 = select_in_superstamp_approx(ra, dec)
    if np.sum(mask_1) == 0:
        raise ValueError('no epochs on silicon - stage 1')
    ra_1 = ra[mask_1]
    dec_1 = dec[mask_1]
    where = np.where(mask_1)[0]

    # full trimming to superstamp area
    (on_silicon, channel, _, _) = get_CCR(ra_1, dec_1)
    mask_2 = np.zeros(len(ra), dtype=bool)
    for (index, on, channel_) in zip(where, on_silicon, channel):
        if on and channel_ in ok_channels:
            mask_2[index] = True
    if np.sum(mask_2) == 0:
        raise ValueError('no epochs on silicon - stage 2')

    return mask_2


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('one parameter needed - file with ephemeris')

    data = np.loadtxt(sys.argv[1], unpack=True)
    time = data[0]
    ra = data[1]
    dec = data[2]

    campaign = guess_campaign(time)

    # trimming to superstamp area
    mask_1 = get_in_superstamp_mask(ra, dec)
    where_1 = np.where(mask_1)[0]

    # select epoch/position in the middle and get K2 epochs
    i_middle = where_1[len(where_1)//2]
    ra_middle = ra[i_middle]
    dec_middle = dec[i_middle]

# XXX - line below fails on ephem_C9a_v1/264.ephem
    (time_3, mask_3) = get_time_mask(ra_middle, dec_middle, campaign)
    mask_ = np.logical_not(np.isnan(time_3))
    time_3 = time_3[mask_] + 2450000.
    mask_3 = mask_3[mask_]
    if np.sum(mask_3) == 0:
        raise ValueError('no epochs on silicon - stage 3')

    # interpolate ephemeris
    (ra_3, dec_3) = interpolate_ephemeris(time, ra, dec, time_3)

    # trim interpolated ephmeris
    mask_4 = get_in_superstamp_mask(ra_3, dec_3)
    time_4 = time_3[mask_4]
    ra_4 = ra_3[mask_4]
    dec_4 = dec_3[mask_4]

    # check exact epochs and masks for interpolated ephemeris
    time_5 = []
    for (time_, ra_, dec_) in zip(time_4, ra_4, dec_4):
        try:
            (time_tmp, mask_tmp) = get_time_mask(ra_, dec_, campaign)
        except Exception:
            pass
        else:
            mask_ = np.logical_not(np.isnan(time_tmp))
            time_tmp = time_tmp[mask_] + 2450000.
            index = get_index_nearest(time_tmp, time_)
            if mask_tmp[mask_][index]:
                time_5.append(time_tmp[index])

    # final interpolation:
    (ra_5, dec_5) = interpolate_ephemeris(time, ra, dec, np.array(time_5))

    for (time_, ra_, dec_) in zip(time_5, ra_5, dec_5):
        print("{:.5f} {:.6f} {:.6f}".format(time_, ra_, dec_))
