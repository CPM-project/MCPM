"""
interpolate ephemeris to different time stamps
"""
import numpy as np
from scipy import interpolate


def interpolate_ephemeris(time, ra, dec, requested_time):
    """
    Interpolate provided ephemeris to requested time vector
    """
    kwargs = dict(fill_value='extrapolate', kind='quadratic')
    fun_ra = interpolate.interp1d(time, ra, **kwargs)
    ra_out = fun_ra(requested_time)
    fun_dec = interpolate.interp1d(time, dec, **kwargs)
    dec_out = fun_dec(requested_time)
    return (ra_out, dec_out)


if __name__ == '__main__':
    file_1 = 'ephem_C9b_v1/30617.ephem'
    file_2 = 'ephem_C9b_v1/30617_time_tpf.dat'

    (e_time, e_ra, e_dec, _) = np.loadtxt(file_1, unpack=True)
    time = np.loadtxt(file_2, unpack=True)

    (ra, dec) = interpolate_ephemeris(e_time, e_ra, e_dec, time)

    for (time_, ra_, dec_) in zip(time, ra, dec):
        print("{:.5f} {:.6f} {:.6f}".format(time_, ra_, dec_))
