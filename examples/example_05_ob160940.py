import numpy as np
from scipy.optimize import minimize
from os import path
import matplotlib.pyplot as plt
import sys

from MCPM import utils
from MCPM.cpmfitsource import CpmFitSource


def pspl_model(t_0, u_0, t_E, f_s, time=None, cpm_source=None):
    """Paczyncki model, provide either time vector or CpmFitSource instance cpm_source"""
    if (time is None) == (cpm_source is None):
        raise ValueError('provide time or cpm_source')
    if time is None:
        time = cpm_source.pixel_time

    tau = (cpm_source.pixel_time - t_0) / t_E
    u_2 = tau**2 + u_0**2
    model = (u_2 + 2.) / np.sqrt(u_2 * (u_2 + 4.))
    model *= f_s
    
    return model
    
def fun_3(inputs, cpm_source, l2, t_E):
    """3-parameter function for optimisation; t_E - fixed"""
    t_0 = inputs[0]
    u_0 = inputs[1]
    t_E = t_E
    f_s = inputs[2]
    
    if u_0 < 0. or t_E < 0. or f_s < 0.:
        return 1.e6

    model = pspl_model(t_0, u_0, t_E, f_s, cpm_source=cpm_source)

    cpm_source.run_cpm(l2, model)
    
    #print(t_0, u_0, t_E, f_s, cpm_source.residuals_rms)
    return cpm_source.residuals_rms
    
def fun_4(inputs, cpm_source, l2):
    """4-parameter function for optimisation"""
    t_0 = inputs[0]
    u_0 = inputs[1]
    t_E = inputs[2]
    f_s = inputs[3]
    
    if u_0 < 0. or t_E < 0. or f_s < 0.:
        return 1.e6

    model = pspl_model(t_0, u_0, t_E, f_s, cpm_source=cpm_source)

    cpm_source.run_cpm(l2, model)
    
    #print(t_0, u_0, t_E, f_s, cpm_source.residuals_rms)
    return cpm_source.residuals_rms
    

if __name__ == "__main__":
    # We want to extract the light curve of ob160980
    channel = 31
    campaign = 92
    ra = 269.564875
    dec = -27.963583

    half_size = 2
    n_select = 10
    l2 = 10**8.5
    l2 = 10**6
    start = np.array([7557., 0.4, 11., 300.])
    start_3 = np.array([7557., .4, 250.])
    t_E = 11.5
    tol = 0.001
    #method = 'Nelder-Mead' # only these 2 make sense
    method = 'Powell'
    n_remove = 10
    
    cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, 
            channel=channel)
    
    cpm_source.get_predictor_matrix()
    cpm_source.set_pixels_square(half_size)
    cpm_source.select_highest_prf_sum_pixels(n_select)

    # Optimize model parameters:
    args = (cpm_source, l2, t_E)
    out = minimize(fun_3, start_3, args=args, tol=tol, method=method)
    print(out)
    
    # plot the best model
    model = pspl_model(out.x[0], out.x[1], t_E, out.x[2], cpm_source=cpm_source)
    cpm_source.run_cpm(l2, model)
    print("RMS: {:.4f}  {:}".format(cpm_source.residuals_rms, np.sum(cpm_source.residuals_mask)))
    mask = cpm_source.residuals_mask
    plt.plot(cpm_source.pixel_time[mask], cpm_source.residuals[mask]+model[mask], '.')
    plt.plot(cpm_source.pixel_time[mask], cpm_source.residuals[mask], '.')
    plt.show()
    
    # you may want to plot residuals as a function of position:
    if False:
        plt.scatter(cpm_source.x_positions[mask], cpm_source.y_positions[mask], c=np.abs(cpm_source.residuals[mask]))
        plt.show()
    
    # Remove most outlying points:
    if True:
        limit = np.sort(np.abs(cpm_source.residuals[mask]))[-n_remove]
        cpm_source.mask_bad_epochs_residuals(limit)
        model = pspl_model(out.x[0], out.x[1], t_E, out.x[2], cpm_source=cpm_source)
        cpm_source.run_cpm(l2, model)
        print("RMS: {:.4f}  {:}".format(cpm_source.residuals_rms, np.sum(cpm_source.residuals_mask)))

    # Optimize model parameters once more:
    if True:
        out = minimize(fun_3, out.x, args=args, tol=tol, method=method)
        print(out)
        
        model = pspl_model(out.x[0], out.x[1], t_E, out.x[2], cpm_source=cpm_source)
        cpm_source.run_cpm(l2, model)
        print("RMS: {:.4f}  {:}".format(cpm_source.residuals_rms, np.sum(cpm_source.residuals_mask)))
        
        # plot it:
        if True:
            mask = cpm_source.residuals_mask
            plt.plot(cpm_source.pixel_time[mask], cpm_source.residuals[mask]+model[mask], '.')
            plt.plot(cpm_source.pixel_time[mask], cpm_source.residuals[mask], '.')
            plt.show()    
            