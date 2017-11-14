import numpy as np
import matplotlib.pyplot as plt

from MCPM.multipletpf import MultipleTpf
from MCPM.campaigngridradec2pix import CampaignGridRaDec2Pix
from MCPM import utils
from MCPM.prfdata import PrfData
from MCPM.prfforcampaign import PrfForCampaign
from MCPM.cpmfitpixel import CpmFitPixel


class CpmFitSource(object):
    """Class for performing CPM fit for a source that combines data 
    from a number of pixels"""
    
    def __init__(self, ra, dec, campaign, channel, 
            multiple_tpf=None, campaign_grids=None, prf_data=None):
        self.ra = ra
        self.dec = dec
        self.campaign = campaign
        self.channel = channel
        
        if multiple_tpf is None:
            multiple_tpf = MultipleTpf()
            multiple_tpf.campaign = campaign
            multiple_tpf.channel = channel
        self.multiple_tpf = multiple_tpf
        
        if campaign_grids is None:
            campaign_grids = CampaignGridRaDec2Pix(campaign=self.campaign, 
                                channel=self.channel)
        self.campaign_grids = campaign_grids
        
        if prf_data is None:
            prf_data = PrfData(channel=self.channel)
        self.prf_data = prf_data
        
        self.prf_for_campaign = PrfForCampaign(campaign=self.campaign, 
                grids=self.campaign_grids, 
                prf_data=self.prf_data)
        
        self._x_positions = None
        self._y_positions = None
        self._xy_positions_mask = None
        self._mean_x = None
        self._mean_y = None
        self._mean_xy_mask = None
        self._pixels = None
        self._predictor_matrix = None
        self._predictor_matrix_mask = None
        self._l2 = None
        self._l2_per_pixel = None
        self._prf_values = None
        self._prf_values_mask = None
        self._pixel_time = None
        self._pixel_flux = None
        self._pixel_mask = None  
        self._train_mask = None
        self._cpm_pixel = None
        self._pixel_residuals = None
        self._pixel_residuals_mask = None
        self._residuals = None
        self._residuals_mask = None

    @property
    def n_pixels(self):
        """number of pixels"""
        if self._pixels is None:
            return 0
        return len(self._pixels)        

    @property
    def pixels(self):
        """list of currently used pixels"""
        return self._pixels
    
    def _calculate_positions(self):
        """calcualte the pixel position of the source for all epochs"""
        out = self.campaign_grids.apply_grids(ra=self.ra, dec=self.dec)
        (self._x_positions, self._y_positions) = out
        self._xy_positions_mask = self.campaign_grids.mask
        
    @property
    def x_positions(self):
        """x-coordinate of pixel positions of the source for all epochs"""
        if self._x_positions is None:
            self._calculate_positions()
        return self._x_positions

    @property
    def y_positions(self):
        """y-coordinate of pixel positions of the source for all epochs"""
        if self._y_positions is None:
            self._calculate_positions()
        return self._y_positions

    @property
    def xy_positions_mask(self):
        """epoch mask for x_positions and y_positions"""
        if self._xy_positions_mask is None:
            self._calculate_positions()
        return self._xy_positions_mask

    def _calculate_mean_xy(self):
        """calculate mean position of the source in pixels"""
        out = self.campaign_grids.mean_position_clipped(ra=self.ra, dec=self.dec)
        (self._mean_x, self._mean_y, self._mean_xy_mask) = out
        
    @property
    def mean_x(self):
        """give mean x coordinate"""
        if self._mean_x is None:
            self._calculate_mean_xy()
        return self._mean_x
        
    @property
    def mean_y(self):
        """give mean y coordinate"""
        if self._mean_y is None:
            self._calculate_mean_xy()
        return self._mean_y
        
    @property
    def mean_xy_mask(self):
        """epoch mask for mean_x and mean_y"""
        if self._mean_xy_mask is None:
            self._calculate_mean_xy()
        return self._mean_xy_mask

    def set_pixels_square(self, half_size):
        """set pixels to be a square of size (2*half_size+1)^2 around 
        the mean position; e.g., half_size=2 gives 5x5 square"""
        self._pixels = utils.pixel_list_center(self.mean_x, self.mean_y, 
                half_size)
    
    def get_predictor_matrix(self, n_pixel=None, min_distance=None, 
            exclude=None, median_flux_ratio_limits=None, 
            median_flux_limits=None):
        """calculate predictor_matrix and its mask"""
        kwargs = {}
        if n_pixel is not None:
            kwargs['n_pixel'] = n_pixel
        if min_distance is not None:
            kwargs['min_distance'] = min_distance
        if exclude is not None:
            kwargs['exclude'] = exclude
        if median_flux_ratio_limits is not None:
            kwargs['median_flux_ratio_limits'] = median_flux_ratio_limits
        if median_flux_limits is not None:
            kwargs['median_flux_limits'] = median_flux_limits
            
        out = self.multiple_tpf.get_predictor_matrix(ra=self.ra, dec=self.dec, 
                **kwargs)
        
        self._predictor_matrix = out[0]
        self._predictor_matrix_mask = out[1]
        
    @property
    def predictor_matrix(self):
        """matrix of predictor fluxes"""
        if self._predictor_matrix is None:
            msg = 'run get_predictor_matrix() to get predicotor matrix'
            raise ValueError(msg)
        return self._predictor_matrix

    @property
    def predictor_matrix_mask(self):
        """epoch mask for matrix of predictor fluxes"""
        if self._predictor_matrix_mask  is None:
            msg = 'run get_predictor_matrix() to get predicotor matrix mask'
            raise ValueError(msg)
        return self._predictor_matrix_mask

    def set_l2_l2_per_pixel(self, l2=None, l2_per_pixel=None):
        """sets values of l2 and l2_per_pixel - provide ONE of them in input"""
        (self._l2, self._l2_per_pixel) = utils.get_l2_l2_per_pixel(
                                            self.predictor_matrix.shape[1],
                                            l2, l2_per_pixel)

    @property
    def l2(self):
        """strength of L2 regularization"""
        return self._l2

    @property
    def l2_per_pixel(self):
        """strength of L2 regularization divided by number of training
        pixels"""
        return self._l2_per_pixel

    def _get_prf_values(self):
        """calculates PRF values"""
        out = self.prf_for_campaign.apply_grids_and_prf(
                    ra=self.ra, dec=self.dec, 
                    pixels=self._pixels)  
        (self._prf_values, self._prf_values_mask) = out
        if self._mean_xy_mask is not None:
            self._prf_values_mask &= self._mean_xy_mask 
        
    @property
    def prf_values(self):
        """PRF values for every pixel and every epoch"""
        if self._prf_values is None:
            self._get_prf_values()
        return self._prf_values
        
    @property
    def prf_values_mask(self):
        """epoch mask for PRF values for every pixel and every epoch"""
        if self._prf_values_mask is None:
            self._get_prf_values()
        return self._prf_values_mask

    def select_highest_prf_sum_pixels(self, n_select):
        """calculate sum of PRF values for every pixel and select n_select 
        ones with the highest sum"""
        if n_select >= len(self._pixels):
            raise ValueError('selection of too many pixels requested')
        prf_sum = np.sum(self.prf_values[self.prf_values_mask], axis=0)
        sorted_indexes = np.argsort(prf_sum)[::-1][:n_select]
        
        self._pixels = self._pixels[sorted_indexes]
        self._prf_values = self._prf_values[:, sorted_indexes]

        self._pixel_time = None
        self._pixel_flux = None
        self._pixel_mask = None
        self._cpm_pixel = None

    def _get_time_flux_mask_for_pixels(self):
        """extract time vectors, flux vectors and epoch masks 
        for pixels from TPF files"""
        out = self.multiple_tpf.get_time_flux_mask_for_pixels(self._pixels)
        reference_time = out[0][0][out[2][0]]
        for i in range(1, self.n_pixels):
            masked_time = out[0][i][out[2][i]]
            if (reference_time != masked_time).any():
                msg = "we assumed time vactrors should be the same\n{:}\n{:}"
                raise ValueError(msg.format(out[0][0], out[0][i]))
        self._pixel_time = out[0][0]
        self._pixel_flux = out[1]
        self._pixel_mask = out[2]

    @property
    def pixel_time(self):
        """time vectors for all pixels"""
        if self._pixel_time is None:
            self._get_time_flux_mask_for_pixels()
        return self._pixel_time
        
    @property
    def pixel_flux(self):
        """fluxe vectors for all pixels"""
        if self._pixel_flux is None:
            self._get_time_flux_mask_for_pixels()
        return self._pixel_flux
        
    @property
    def pixel_mask(self):
        """epoch masks for pixel_flux and pixel_time"""
        if self._pixel_mask is None:
            self._get_time_flux_mask_for_pixels()
        return self._pixel_mask
    
    def mask_bad_epochs(self, epoch_indexes):
        """for every index in epoch_indexes each pixel_mask will be made False"""
        masks = self.pixel_mask
        for index in epoch_indexes:
            for i in range(self.n_pixels):
                masks[i][index] = False
                
    def mask_bad_epochs_residuals(self, limit=None):
        """mask epochs with residuals lrager than limit or smaller than -limit;
        if limit is not provided than 5*residuals_rms is assumed
        """
        if limit is None:
            limit = 5 * self.residuals_rms
        mask = self.residuals_mask
        mask_bad = (self.residuals[mask]**2 >= limit**2)
        indexes = np.arange(len(mask))[mask][mask_bad]
        self.mask_bad_epochs(indexes)                    
    
    def set_train_mask(self, train_mask):
        """sets the epoch mask used for training in CPM"""
        self._train_mask = train_mask

    def pspl_model(self, t_0, u_0, t_E, f_s):
        """Paczynski (or point-source/point-lens) microlensing model"""
        return utils.pspl_model(t_0, u_0, t_E, f_s, self.pixel_time)

    def run_cpm(self, model, model_mask=None):
        """Run CPM on all pixels. Model has to be provided for epochs in
        self.pixel_time. If the epoch mask model_mask is None, then it's 
        assumed it's True for each epoch. Mask of PRF is applied inside 
        this function."""
        self._cpm_pixel = [None] * self.n_pixels
        self._pixel_residuals = None
        self._pixel_residuals_mask = None
        self._residuals = None
        self._residuals_mask = None
        
        if model_mask is None:
            model_mask = np.ones_like(model, dtype=bool)
        model_mask *= self._prf_values_mask

        for i in range(self.n_pixels):
            model_i = model * self.prf_values[:,i]
            
            self._cpm_pixel[i] = CpmFitPixel(
                    target_flux=self.pixel_flux[i], 
                    target_flux_err=None, target_mask=self.pixel_mask[i], 
                    predictor_matrix=self.predictor_matrix, 
                    predictor_matrix_mask=self.predictor_matrix_mask, 
                    l2=self.l2, model=model_i, model_mask=model_mask, 
                    time=self.pixel_time, train_mask=self._train_mask)

    @property
    def pixel_residuals(self):
        """list of residuals for every pixel"""
        if self._pixel_residuals is None:
            self._pixel_residuals = [None] * self.n_pixels
            self._pixel_residuals_mask = [None] * self.n_pixels
            failed = []
            for i in range(self.n_pixels):
                residuals = self.pixel_time * 0.
                if self._cpm_pixel is None:
                    raise ValueError("CPM not yet run but you're trying to "
                            + "access its results")
                cpm = self._cpm_pixel[i]
                try:
                    residuals[cpm.results_mask] = cpm.residuals[cpm.results_mask]
                except np.linalg.linalg.LinAlgError as inst:
                    failed.append(inst)
                    txt = inst.args[0]
                    expected = "-th leading minor of the array is not positive definite"
                    number = txt.split("-")[0]
                    # Remove int(number)-1
                    #print(number)
                    #print(number+expected==txt)
                    continue
                self._pixel_residuals[i] = residuals
                self._pixel_residuals_mask[i] = cpm.results_mask
            if len(failed) > 0:
                fmt = "Failed: {:} of {:}" + len(failed) * "\n{:}"
                raise ValueError(fmt.format(len(failed), self.n_pixels, *failed))
        return self._pixel_residuals
            
    @property
    def pixel_residuals_mask(self):
        """epoch mask for pixel_residuals"""
        if self._pixel_residuals_mask is None:
            self.pixel_residuals
        return self._pixel_residuals_mask
    
    @property
    def residuals(self):
        """residuals summed over pixels"""
        if self._residuals is None:
            residuals = self.pixel_time * 0.
            residuals_mask = np.ones_like(residuals, dtype=bool)
            for i in range(self.n_pixels):
                mask = self.pixel_residuals_mask[i]
                residuals[mask] += self.pixel_residuals[i][mask]
                residuals_mask &= mask
            self._residuals = residuals
            self._residuals_mask = residuals_mask
        return self._residuals

    @property
    def residuals_mask(self):
        """epoch mask for residuals summed over pixels"""
        if self._residuals_mask is None:
            self.residuals
        return self._residuals_mask
    
    @property
    def pixel_residuals_rms(self):
        """calculate RMS of residuals using each pixel separately"""
        out = []
        for i in range(self.n_pixels):
            out.append(self.pixel_residuals[i][self.residuals_mask])
        rms = np.sqrt(np.mean(np.square(np.array(out))))
        return rms
        
    @property
    def residuals_rms(self):
        """calculate RMS of residuals combining all pixels"""
        rms = np.sqrt(np.mean(np.square(self.residuals[self.residuals_mask])))
        return rms

    def residuals_rms_for_mask(self, mask):
        """calculate RMS of residuals combining all pixels and applying 
        additional epoch mask"""
        mask_all = mask & self.residuals_mask
        rms = np.sqrt(np.mean(np.square(self.residuals[mask_all])))
        return rms
       
    def plot_pixel_residuals(self, shift=None):
        """Plot residuals for each pixel separately. Parameter
        shift (int or float) sets the shift in Y axis between the pixel,
        the default value is 2*RMS rounded up to nearest 10"""
        if shift is None:
            shift = round(2 * self.residuals_rms, -1) # -1 mean "next 10"

        mask = self.residuals_mask
        time = self.pixel_time[mask]
        for i in range(self.n_pixels):
            mask = self.residuals_mask
            y_values = self.pixel_residuals[i][mask] + i * shift
            plt.plot(time, time*0+i * shift, 'k--')
            plt.plot(time, y_values, '.', label="pixel {:}".format(i))

    def plot_pixel_curves(self, **kwargs):
        """Use matplotlib to plot raw data for a set of pixels. 
        For options look at MultipleTpf.plot_pixel_curves()"""
        self.multiple_tpf.plot_pixel_curves(self.mean_x, self.mean_y, **kwargs)

    def run_cpm_and_plot_model(self, model, model_mask=None, 
            plot_residuals=False, f_s=None):
        """Run CPM on given model and plot the model and model+residuals;
        also plot residuals if requested via plot_residuals=True.
        Magnification is used instead of counts if f_s is provided
        """
        lw = 5
        self.run_cpm(model=model, model_mask=model_mask)

        mask = self.residuals_mask
        if model_mask is None:
            model_mask = np.ones_like(self.pixel_time, dtype=bool)
        if self._train_mask is None:
            mask_2 = np.zeros_like(self.pixel_time, dtype=bool)
        else:
            mask_2 = mask & ~self._train_mask
            mask &= self._train_mask

        plt.xlabel("HJD'")
        if f_s is None:
            y_label = 'counts'
            residuals = self.residuals 
        else:
            y_label = 'magnification'
            model /= f_s
            residuals = self.residuals / f_s
            
        plt.ylabel(y_label)
        plt.plot(self.pixel_time[model_mask], model[model_mask], 'k-', lw=lw)
        plt.plot(self.pixel_time[mask], residuals[mask] + model[mask], 'b.')
        plt.plot(self.pixel_time[mask_2], residuals[mask_2] + model[mask_2], 
                'bo')
        
        if plot_residuals:
            if self._train_mask is None:
                mask_2 = None
            self._plot_residuals_of_last_model(mask, mask_2, f_s)

    def _plot_residuals_of_last_model(self, mask, mask_2=None, f_s=None):
        """inner function that makes the plotting; magnification is plotted instead of counts 
        if f_s is provided"""
        lw = 5
        plt.plot(self.pixel_time[mask], self.pixel_time[mask]*0., 'k--', lw=lw)

        residuals = self.residuals
        if f_s is not None:
            residuals /= f_s

        plt.plot(self.pixel_time[mask], residuals[mask], 'r.')
        if mask_2 is not None:
            plt.plot(self.pixel_time[mask_2], residuals[mask_2], 'ro')

    def plot_residuals_of_last_model(self):
        """plot residuals of last model run"""
        plt.xlabel("HJD'")
        plt.ylabel("residual counts")
        self._plot_residuals_of_last_model(self.residuals_mask)

    def plot_pixel_model_of_last_model(self, pixel_i, mask=None):
        """plot full model for pixel with given index pixel_i and the data
        themselves"""
        if mask is None:
            mask_ = self.residuals_mask
        else:
            mask_ = mask & self.residuals_mask
        signal = self.pixel_flux[pixel_i] - self._pixel_residuals[pixel_i]

        plt.xlabel("HJD'")
        plt.ylabel("counts")
        plt.plot(self.pixel_time[mask_], signal[mask_], 'k-')
        plt.plot(self.pixel_time[mask_], signal[mask_], 'ks')
        plt.plot(self.pixel_time[mask_], self.pixel_flux[pixel_i][mask_], 'ro')

