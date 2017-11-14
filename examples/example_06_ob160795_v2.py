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
    
    #print(t_0, u_0, t_E, f_s, cpm_source.residuals_rms)
    return cpm_source.residuals_rms 
   

if __name__ == "__main__":
    # We want to extract the light curve of ob160795
    channel = 52
    campaign = 91
    ra = 271.001083
    dec = -28.155111

    half_size = 2
    n_select = 20
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
    print(cpm_source.pixels)

    # print meadians:
    #for (i, ind) in enumerate(cpm_source.pixels):
        #print(ind, np.median(cpm_source.pixel_flux[i]))

    pixel_mask = cpm_source.pixel_mask[0] * cpm_source.pixel_mask[1] * cpm_source.pixel_mask[2]
    signal = cpm_source.pixel_flux[0][pixel_mask] + cpm_source.pixel_flux[1][pixel_mask] + cpm_source.pixel_flux[2][pixel_mask]
    mask_2 = (cpm_source.pixel_time[pixel_mask] > 7510.) & (cpm_source.pixel_time[pixel_mask] < 7515.)
    plt.plot(cpm_source.pixel_time[pixel_mask][mask_2], signal[mask_2], 'o')
    plt.title('ob170795 - raw sum of pixels (522, 271), (522, 270), and (523, 271)')
    plt.xlabel("HJD'")
    plt.ylabel("counts")
    plt.savefig("ob160795_pixel_curves_v3.png")
    #plt.show()
    plt.close()

    signal = cpm_source.pixel_flux[6][pixel_mask] + cpm_source.pixel_flux[3][pixel_mask]
    print(cpm_source.pixels[6])
    print(cpm_source.pixels[3])
    #plt.plot(cpm_source.pixel_time[pixel_mask][mask_2],cpm_source.pixel_flux[6][pixel_mask][mask_2], 'ro')
    #plt.plot(cpm_source.pixel_time[pixel_mask][mask_2],cpm_source.pixel_flux[3][pixel_mask][mask_2], 'go')
#    plt.plot(cpm_source.pixel_time[pixel_mask][mask_2], signal[mask_2], 'o')
#    plt.show()
#    plt.close()

    # Plot pixel curves:
    cpm_source.plot_pixel_curves(y_lim=[1900., 4500.])
    plt.savefig("ob160795_pixel_curves_v2.png")
    plt.close()
    print("Pixel curves done")
    
    # Plot "Wei's model"
    model = cpm_source.pixel_time * 0. + f_s
    train_mask = np.ones_like(cpm_source.pixel_time, dtype=bool)
    mask = cpm_source.pixel_mask[0]
    train_mask[mask] = (cpm_source.pixel_time[mask] < 7510.) | (cpm_source.pixel_time[mask] > 7515.)
    cpm_source.set_train_mask(train_mask)
    cpm_source.run_cpm_and_plot_model(model, plot_residuals=True)
    print("RMS: {:.4f}".format(cpm_source.residuals_rms))
    #cpm_source._residuals_mask *= train_mask
    print("RMS: {:.4f}".format(cpm_source.residuals_rms))
    #plt.show()
    plt.close()
    #sys.exit()

    # plot centroids:
    train_mask_ = (~train_mask) & cpm_source.pixel_mask[0] & cpm_source._prf_values_mask

    print(cpm_source.x_positions[train_mask_])
    plt.plot(cpm_source.pixel_time[train_mask_], cpm_source.x_positions[train_mask_]-np.median(cpm_source.x_positions[train_mask_]), 'o')
    plt.plot(cpm_source.pixel_time[train_mask_], cpm_source.y_positions[train_mask_]-np.median(cpm_source.y_positions[train_mask_])+1, 'o')
    plt.show()
    plt.close()

    # Optimize model parameters:
    args = (cpm_source, t_E, f_s)
    out = minimize(fun_2, start_2, args=args, tol=tol, method=method)
    print(out)

    # plot single pixels:
    ids = ['A', 'B', 'C']
    for (i, id_) in enumerate(ids):
        cpm_source.plot_pixel_model_of_last_model(i, ~train_mask)
        plt.title('ob160795 - pixel {:}'.format(id_))
        plt.savefig('ob160795_pixel_{:}.png'.format(id_))
        plt.close()

    cpm_source.plot_pixel_model_of_last_model(8, ~train_mask)
    plt.title('ob160795 - pixel {:}'.format("H"))
    plt.savefig('ob160795_pixel_{:}.png'.format("H"))
    plt.close()

    cpm_source.plot_pixel_model_of_last_model(14, ~train_mask)
    plt.title('ob160795 - pixel {:}'.format("N"))
    plt.savefig('ob160795_pixel_{:}.png'.format("N"))
    plt.close()

    sys.exit()


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
   
