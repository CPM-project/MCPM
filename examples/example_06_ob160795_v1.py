import numpy as np
from scipy.optimize import minimize
from scipy.stats import sigmaclip
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

def fun_2(inputs, cpm_source, t_E, f_s):
    """2-parameter function for optimisation; t_E and f_S- fixed"""
    t_0 = inputs[0]
    u_0 = inputs[1]
    t_E = t_E
    f_s = f_s
    
    if u_0 < 0. or t_E < 0. or f_s < 0.:
        return 1.e6

    model = pspl_model(t_0, u_0, t_E, f_s, cpm_source=cpm_source)

    cpm_source.run_cpm(model)
    
    #print(t_0, u_0, t_E, f_s, cpm_source.residuals_rms)
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
    start_2 = np.array([7512.6, .12])
    t_E = 4.38
    f_s = 94.6
    tol = 0.001
    #method = 'Nelder-Mead' # only these 2 make sense
    method = 'Powell'
    
    cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, 
            channel=channel)
    
    cpm_source.get_predictor_matrix()
    cpm_source.set_l2_l2_per_pixel(l2=l2)
    cpm_source.set_pixels_square(half_size)
    cpm_source.select_highest_prf_sum_pixels(n_select)
    
    # Plot pixel curves:
    cpm_source.plot_pixel_curves()
    plt.savefig("ob160795_pixel_curves.png")
    plt.close()

    # Optimize model parameters:
    args = (cpm_source, t_E, f_s)
    out = minimize(fun_2, start_2, args=args, tol=tol, method=method)
    print(out)
    
    # plot the best model
    model = pspl_model(out.x[0], out.x[1], t_E, f_s, cpm_source=cpm_source)
    cpm_source.run_cpm(model)
    print("RMS: {:.4f}  {:}".format(cpm_source.residuals_rms, np.sum(cpm_source.residuals_mask))) 
    mask = cpm_source.residuals_mask
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(cpm_source.pixel_time, model, '-')
    ax.plot(cpm_source.pixel_time[mask], cpm_source.residuals[mask]+model[mask], '.')
    ax.plot(cpm_source.pixel_time[mask], cpm_source.residuals[mask], '.')
    plt.xlabel("HJD'")
    plt.ylabel("counts")
    ax.set_xbound(7505., 7520.)
    plt.title('MCPM for ob160795 (t_E = {:} d; f_s_Kp = {:})'.format(t_E, f_s))
    plt.savefig('ob160795_CPM_v1.png')
    plt.close()

    cpm_source.plot_pixel_residuals()
    plt.xlabel("HJD'")
    plt.ylabel("counts + const.")
    plt.title('MCPM residuals for each pixel separately')
    #plt.show()
    plt.savefig('ob160795_CPM_v1_res.png')

