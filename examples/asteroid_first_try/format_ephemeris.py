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
    Take numpy arrays of ra & dec (both in deg) and check XXX
    XXX - seems now it only checks if it's in superstamp CHANNELS
    Result is a mask that indicates which coords
    """
    ok_channels = [30, 31, 32, 49, 52]

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
    where_1 = np.where(mask_1)[0].tolist()

    # get time vector for single coordinates, starting from
    # the middle of the masked vector
    i_middle = len(where_1)//2
    indexes = where_1[i_middle:] + where_1[:i_middle]
    for index in indexes:
        try:
            (time_3, mask_3) = get_time_mask(ra[index], dec[index], campaign)
        except Exception:  # i.e., the pixel is not in a superstamp
            if index == indexes[-1]:
                raise ValueError('no epochs in superstamp - stage 3')
        else:
            break
    mask_ = np.logical_not(np.isnan(time_3))
    time_3 = time_3[mask_] + 2450000.
    mask_3 = mask_3[mask_]
    if np.sum(mask_3) == 0:
        raise ValueError('no epochs on silicon - stage 4')

    # interpolate ephemeris
    (ra_3, dec_3) = interpolate_ephemeris(time, ra, dec, time_3)

    # trim interpolated ephemeris
    mask_4 = get_in_superstamp_mask(ra_3, dec_3)
    time_4 = time_3[mask_4]
    ra_4 = ra_3[mask_4]
    dec_4 = dec_3[mask_4]

    # check exact epochs and masks for interpolated ephemeris
    time_5 = []
    previous_epic = None
    for (time_, ra_, dec_) in zip(time_4, ra_4, dec_4):
        # epic = ...
        # if epic is None:
        #     continue
        # if previous_epic != epic:
        # XXX - AND NO INDENT BLOCK try BELOW:
        try:
            # XXX this is probably the slowest part - can be speed up if we
            # use the times from previous coords provided that they're in
            # the same TPF - check it using
            # TpfRectangles.get_epic_id_for_pixel(x, y)
            # XXX - make sure results are the same
            (time_tmp, mask_tmp) = get_time_mask(ra_, dec_, campaign)
        except Exception:
            continue
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
