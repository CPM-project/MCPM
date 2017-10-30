import numpy as np
from scipy.optimize import minimize
from os import path
import matplotlib.pyplot as plt
import sys

from MCPM import utils
from MCPM.cpmfitsource import CpmFitSource


def transform_model(t_0, amplitude_ratio, width_ratio, model_dt, model_flux, time):
    """Simple function for scaling and linear interpolation.
    First 3 parameters are floats, the rest are vectors.
    """
    model_time = t_0 + model_dt * width_ratio
    model = np.interp(time, model_time, model_flux) * amplitude_ratio
    return model


if __name__ == "__main__":
    # We want to extract the light curve of OGLE-BLG-ECL-234840
    channel = 31
    campaign = 91
    ra = 269.929125
    dec = -28.410833

    half_size = 2
    n_select = 10
    l2 = 10**8.5
    #start_1 = np.array([7516.])
    #start = np.array([7518., 0.5, 0.3])
    #tol = 0.0001
    ##method = 'Nelder-Mead' # only these 2 make sense
    #method = 'Powell'
    model_file = "example_1_model_averaged.dat"
    
    cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, 
            channel=channel)
    
    print("Mean target position: {:.2f} {:.2f}\n".format(cpm_source.mean_x, 
            cpm_source.mean_y))
    
    cpm_source.get_predictor_matrix()
    cpm_source.set_pixels_square(half_size)
    cpm_source.select_highest_prf_sum_pixels(n_select)
    
    (model_dt, model_flux) = np.loadtxt(model_file, unpack=True)
    model_flux[model_dt < -13.] = 0.
    model_flux[model_dt > 13.] = 0.
    
    model = transform_model(7520., 1., 1., model_dt, model_flux, cpm_source.pixel_time[0])
    
    cpm_source.run_cpm(l2, model)
    
    for i in range(cpm_source.n_pixels):
        mask = cpm_source._cpm_pixel[i].results_mask
        #plt.plot(cpm_source.pixel_time[i][mask], cpm_source._cpm_pixel[i].residue[mask], '.')
        plt.plot(cpm_source.pixel_time[i][mask], cpm_source._cpm_pixel[i].cpm_residue[mask], '.')
    plt.show()