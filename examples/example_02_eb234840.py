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

def fun(inputs, model_dt, model_flux, cpm_source):
    """3-parameter function for optimisation"""
    t_0 = inputs[0]
    amplitude_factor = inputs[1]
    width_ratio = inputs[2]

    model = transform_model(t_0, amplitude_factor, width_ratio, model_dt, 
                            model_flux, cpm_source.pixel_time)

    cpm_source.run_cpm(model)
    
    return cpm_source.residuals_rms
    

if __name__ == "__main__":
    # We want to extract the light curve of OGLE-BLG-ECL-234840
    channel = 31
    campaign = 91
    ra = 269.929125
    dec = -28.410833

    half_size = 2
    n_select = 10
    l2 = 10**8.5
    start = np.array([7518., 0.5, 0.3])
    tol = 0.0001
    ##method = 'Nelder-Mead' # only these 2 make sense
    method = 'Powell'
    model_file = "example_1_model_averaged.dat"
    
    cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, 
            channel=channel)
    
    print("Mean target position: {:.2f} {:.2f}\n".format(cpm_source.mean_x, 
            cpm_source.mean_y))
    
    cpm_source.get_predictor_matrix()
    cpm_source.set_l2_l2_per_pixel(l2=l2)
    cpm_source.set_pixels_square(half_size)
    cpm_source.select_highest_prf_sum_pixels(n_select)
    
    (model_dt, model_flux) = np.loadtxt(model_file, unpack=True)
    model_flux[model_dt < -13.] = 0.
    model_flux[model_dt > 13.] = 0.
    
    model = transform_model(7519., 1., 1., model_dt, model_flux, cpm_source.pixel_time)
    
    for i in [510, 860, 856, 1004, 968]:
        for j in range(n_select):
            cpm_source._pixel_mask[j][i] = False
    
    cpm_source.run_cpm(model)
    
    print("RMS: {:.4f}".format(cpm_source.residuals_rms))
    
    mask = cpm_source.residuals_mask
    plt.plot(cpm_source.pixel_time[mask], cpm_source.residuals[mask]+model[mask], '.')
    plt.show()

    # Optimize model parameters:"
    args = (model_dt, model_flux, cpm_source)
    out = minimize(fun, start, args=args, tol=tol, method=method)
    print()  
    print(out.success)
    print(out.nfev)
    print("{:.5f} {:.4f} {:.4f}  ==> {:.4f}".format(out.x[0], out.x[1], out.x[2], out.fun))
    
    model = transform_model(out.x[0], out.x[1], out.x[2], model_dt, model_flux, cpm_source.pixel_time)
    cpm_source.run_cpm(model)
    mask = cpm_source.residuals_mask
    plt.plot(cpm_source.pixel_time[mask], cpm_source.residuals[mask]+model[mask], '.')
    plt.show()
    
    
