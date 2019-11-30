"""
interpolate ephemeris to different time stamps
"""
import numpy as np
from scipy import interpolate


file_1 = 'ephem_C9b_v1/30617.ephem'
file_2 = 'ephem_C9b_v1/30617_time_tpf.dat'

(e_time, e_ra, e_dec, _) = np.loadtxt(file_1, unpack=True)
time = np.loadtxt(file_2, unpack=True)

kwargs = dict(fill_value='extrapolate', kind='quadratic')
fun_ra = interpolate.interp1d(e_time, e_ra, **kwargs)
ra = fun_ra(time)
fun_dec = interpolate.interp1d(e_time, e_dec, **kwargs)
dec = fun_dec(time)

for (time_, ra_, dec_) in zip(time, ra, dec):
    print("{:.5f} {:.6f} {:.6f}".format(time_, ra_, dec_))
