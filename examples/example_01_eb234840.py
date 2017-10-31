import numpy as np
from scipy.optimize import minimize
from os import path
import matplotlib.pyplot as plt

from MCPM.multipletpf import MultipleTpf
from MCPM.campaigngridradec2pix import CampaignGridRaDec2Pix
from MCPM import utils
from MCPM.prfdata import PrfData
from MCPM.prfforcampaign import PrfForCampaign
from MCPM.cpmfitpixel import CpmFitPixel


def transform_model(t_0, amplitude_ratio, width_ratio, model_dt, model_flux, time):
    """Simple function for scaling and linear interpolation.
    First 3 parameters are floats, the rest are vectors.
    """
    model_time = t_0 + model_dt * width_ratio
    model = np.interp(time, model_time, model_flux) * amplitude_ratio
    return model

def cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask,
        prfs, mask_prfs, model, l2):
    """runs CPM on a set of pixels and returns each result"""
    out_signal = []
    out_mask = []
    for i in range(len(tpf_flux)):
        cpm_pixel = CpmFitPixel(
            target_flux=tpf_flux[i], target_flux_err=None, target_mask=tpf_epoch_mask[i], 
            predictor_matrix=predictor_matrix, predictor_matrix_mask=predictor_mask,
            l2=l2, 
            model=model[i]*prfs[:,i], model_mask=mask_prfs,
            time=times[i]
        )
        out_signal.append(cpm_pixel.residue)
        out_mask.append(cpm_pixel.results_mask)
    return (out_signal, out_mask)

def mean_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask,
        prfs, mask_prfs, model, l2):
    """runs CPM on a set of pixels and returns mean residue"""
    (signal, mask) = cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
        predictor_mask, prfs, mask_prfs, model, l2)
    sum_out = 0.
    sum_n = 0
    for i in range(len(signal)):
        sum_out += np.sum(signal[i][mask[i]]**2) 
        sum_n += np.sum(mask[i])
    return (sum_out / sum_n)**0.5

def summed_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask,
        prfs, mask_prfs, model, l2, n_pix, times):
    """runs CPM and adds all the results"""
    (signal, mask) = cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
        predictor_mask, prfs, mask_prfs, model, l2)
    out = signal[0] * 0.
    for i in range(n_pix):
        out[mask[i]] += signal[i][mask[i]]
    return (out, mask[0])

def fun_1(inputs, model_dt, model_flux, time, tpf_flux, tpf_epoch_mask, 
        predictor_matrix, predictor_mask, prfs, mask_prfs, l2):
    """3-parameter function for optimisation"""
    t_0 = inputs[0]

    model = transform_model(t_0, 1., 1., model_dt, 
                            model_flux, time)

    out = mean_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
            predictor_mask, prfs, mask_prfs, model, l2)

    #print(t_0, out)
    return out
    
def fun(inputs, model_dt, model_flux, time, tpf_flux, tpf_epoch_mask, 
        predictor_matrix, predictor_mask, prfs, mask_prfs, l2):
    """3-parameter function for optimisation"""
    t_0 = inputs[0]
    amplitude_factor = inputs[1]
    width_ratio = inputs[2]

    model = transform_model(t_0, amplitude_factor, width_ratio, model_dt, 
                            model_flux, time)

    out = mean_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
            predictor_mask, prfs, mask_prfs, model, l2)

    #print(t_0, amplitude_factor, width_ratio, out)
    return out
    

if __name__ == "__main__":
    # We want to extract the light curve of OGLE-BLG-ECL-234840, which is
    # an eclipsing binary with a single eclipse in subcampaign 91. I know 
    # the star coordinates and that it is in channel 31, but you can find 
    # the channel this way:
    # >>> import K2fov
    # >>> K2fov.fields.getKeplerFov(9)pickAChannel(ra, dec)
    channel = 31
    campaign = 91
    ra = 269.929125
    dec = -28.410833

    half_size = 2
    n_select = 10
    l2 = 10**8.5
    start_1 = np.array([7516.])
    start = np.array([7518., 0.5, 0.3])
    tol = 0.0001
    #method = 'Nelder-Mead' # only these 2 make sense
    method = 'Powell'
    model_file = "example_1_model_averaged.dat"
    
    tpfs = MultipleTpf()
    tpfs.campaign = campaign
    tpfs.channel = channel
    
    out = tpfs.get_predictor_matrix(ra=ra, dec=dec, 
            min_distance=15,
            median_flux_ratio_limits=(0.2, 2.0))
    (predictor_matrix, predictor_matrix_mask) = out
    #n_predictor = 400
    #file_1_name = "eb234840_predictor_{:}.dat".format(n_predictor)
    #file_2_name = "eb234840_predictor_mask_{:}.dat".format(n_predictor)
    #if path.isfile(file_1_name) and path.isfile(file_2_name):
        #predictor_matrix = utils.load_matrix_xy(file_1_name)
        #predictor_matrix_mask = utils.read_true_false_file(file_2_name)
        
    grids = CampaignGridRaDec2Pix(campaign=campaign, channel=channel)
    (mean_x, mean_y, grids_mask) = grids.mean_position_clipped(ra, dec)
    print("Mean target position: {:.2f} {:.2f}\n".format(mean_x, mean_y))

    for i in [510, 860, 856, 1004, 968]:
    #for i in [510, 860, 856, 1004, 968, 475, 474]:
        grids_mask[i] = False
        
    pixels = utils.pixel_list_center(mean_x, mean_y, half_size)
    
    prf_template = PrfData(channel=channel)
    # Third, the highest level structure - something that combines grids and 
    # PRF data:
    prf_for_campaign = PrfForCampaign(campaign=campaign, grids=grids, 
                                    prf_data=prf_template)
    #pixels=[[264, 492], [264, 493], [263, 492], [265, 493], [263, 493],
            #[265, 492], [264, 494], [263, 494], [263, 495], [264, 495]]
                         
    (prfs, mask_prfs) = prf_for_campaign.apply_grids_and_prf(ra, dec, pixels)  
    
    mask_prfs *= grids_mask
    prf_sum = np.sum(prfs[mask_prfs], axis=0)
    sorted_indexes = np.argsort(prf_sum)[::-1][:n_select]
    pixels = pixels[sorted_indexes]
    prfs = prfs[:, sorted_indexes]
   
    (times, pixels_flux, pixel_masks) = tpfs.get_time_flux_mask_for_pixels(pixels)

    (model_dt, model_flux) = np.loadtxt(model_file, unpack=True)
    model_flux[model_dt < -13.] = 0.
    model_flux[model_dt > 13.] = 0.

    for i in [510, 860, 856, 1004, 968]:
    #for i in [510, 860, 856, 1004, 968, 475, 474]:
        predictor_matrix_mask[i] = False

    args = (model_dt, model_flux, times, pixels_flux, pixel_masks, 
            predictor_matrix, predictor_matrix_mask, prfs, mask_prfs, l2)

    # plot 1-D fit:
    if True:
        model = transform_model(7520., 1., 1., model_dt, model_flux, times)
        plt.figure()
        (result, result_mask) = summed_cpm_output(pixels_flux, pixel_masks, 
            predictor_matrix, predictor_matrix_mask, prfs, mask_prfs, model, l2, n_select, times)        
        plt.plot(times[0][result_mask], model[0][result_mask]+(result[result_mask]), '.')
        plt.plot(times[0][result_mask], (result[result_mask]), 'r.')
        plt.plot(times[0][result_mask], model[0][result_mask], '-')
        plt.show()

    # plot a set of function values:
    #if True:
    if False:
        inputs = np.linspace(7515, 7521, 200)
        outputs = 0. * inputs
        for (i, v) in enumerate(inputs):
            outputs[i] = fun_1([v], *args)
        plt.plot(inputs, outputs, 'o')
        plt.show()

    # 1-parameter fit
    out_1 = minimize(fun_1, start_1, args=args, tol=tol, method=method)
    print()
    print(out_1)   

    # 3-parameter fit
    out = minimize(fun, start, args=args, tol=tol, method=method)
    print()
    print(out)  
    