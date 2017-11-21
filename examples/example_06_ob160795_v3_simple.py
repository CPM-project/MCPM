import numpy as np
from scipy.optimize import minimize
from scipy.stats import sigmaclip
from os import path
import matplotlib.pyplot as plt
import sys

from MCPM import utils
from MCPM.cpmfitsource import CpmFitSource


def fun_2(inputs, cpm_source, t_E, f_s):
    """2-parameter function for optimisation; t_E and f_s - fixed"""
    t_0 = inputs[0]
    u_0 = inputs[1]
    if u_0 < 0. or t_E < 0. or f_s < 0.:
        return 1.e6

    model = cpm_source.pspl_model(t_0, u_0, t_E, f_s)
    cpm_source.run_cpm(model)
    
    return cpm_source.residuals_rms 

if __name__ == "__main__":
    # We want to extract the light curve of ob160795
    channel = 52
    campaign = 91
    ra = 271.001083
    dec = -28.155111

    half_size = 2
    n_select = 10
    l2 = 10**5.35
    t_0_a = 7512.6
    u_0_a = .12
    t_0_b = 7512.6055
    u_0_b = 0.066
    t_E = 4.38
    f_s = 94.6
    
    # prepare cpm_source:
    cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, 
            channel=channel)
    cpm_source.get_predictor_matrix()
    cpm_source.set_l2_l2_per_pixel(l2=l2)
    cpm_source.set_pixels_square(half_size)
    cpm_source.select_highest_prf_sum_pixels(n_select)

    # get RMS of residuals for first model:
    model_1 = utils.pspl_model(t_0_a, u_0_a, t_E, f_s, cpm_source.pixel_time)
    cpm_source.run_cpm(model_1)
    print(cpm_source.residuals_rms)

    # and for the other model:
    model_2 = utils.pspl_model(t_0_b, u_0_b, t_E, f_s, cpm_source.pixel_time)
    cpm_source.run_cpm(model_2)
    print(cpm_source.residuals_rms)

