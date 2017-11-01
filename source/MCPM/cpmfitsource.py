import numpy as np

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
        self._prf_values = None
        self._prf_values_mask = None
        self._pixel_time = None
        self._pixel_flux = None
        self._pixel_mask = None  
        self._cpm_pixel = None
        self._pixel_residue = None
        self._pixel_residue_mask = None
        self._residue = None
        self._residue_mask = None
    
    @property
    def n_pixels(self):
        """number of pixels"""
        if self._pixels is None:
            return 0
        return len(self._pixels)        
        
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
                
    def mask_bad_epochs_residual(self, limit=None):
        """mask epochs with residuals lrager than limit or smaller than -limit;
        if limit is not provided than 5*residual_rms is assumed
        """
        if limit is None:
            limit = 5 * self.residual_rms
        mask = self.residue_mask
        mask_bad = (self.residue[mask]**2 >= limit**2)
        indexes = np.arange(len(mask))[mask][mask_bad]
        self.mask_bad_epochs(indexes)                    
    
    def run_cpm(self, l2, model):
        """run CPM on all pixels"""
        self._cpm_pixel = [None] * self.n_pixels
        self._pixel_residue = None
        self._pixel_residue_mask = None
        self._residue = None
        self._residue_mask = None
        
        for i in range(self.n_pixels):
            model_i = model * self.prf_values[:,i]
            model_mask=self._prf_values_mask
            
            self._cpm_pixel[i] = CpmFitPixel(
                    target_flux=self.pixel_flux[i], 
                    target_flux_err=None, target_mask=self.pixel_mask[i], 
                    predictor_matrix=self.predictor_matrix, 
                    predictor_matrix_mask=self.predictor_matrix_mask, 
                    l2=l2, model=model_i, model_mask=model_mask, 
                    time=self.pixel_time)

    @property
    def pixel_residue(self):
        """list of residuals for every pixel"""
        if self._pixel_residue is None:
            self._pixel_residue = [None] * self.n_pixels
            self._pixel_residue_mask = [None] * self.n_pixels
            for i in range(self.n_pixels):
                residuals = self.pixel_time * 0.
                cpm = self._cpm_pixel[i]
                residuals[cpm.results_mask] = cpm.residue[cpm.results_mask]
                self._pixel_residue[i] = residuals
                self._pixel_residue_mask[i] = cpm.results_mask
        return self._pixel_residue
            
    @property
    def pixel_residue_mask(self):
        """epoch mask for pixel_residue"""
        if self._pixel_residue_mask is None:
            self.pixel_residue
        return self._pixel_residue_mask
    
    @property
    def residue(self):
        """residuals summed over pixels"""
        if self._residue is None:
            residuals = self.pixel_time * 0.
            residuals_mask = np.ones_like(residuals, dtype=bool)
            for i in range(self.n_pixels):
                mask = self.pixel_residue_mask[i]
                residuals[mask] += self.pixel_residue[i][mask]
                residuals_mask &= mask
            self._residue = residuals
            self._residue_mask = residuals_mask
        return self._residue

    @property
    def residue_mask(self):
        """epoch mask for residuals summed over pixels"""
        if self._residue_mask is None:
            self.residue
        return self._residue_mask
    
    @property
    def residual_rms(self):
        out = []
        for i in range(self.n_pixels):
            out.append(self.pixel_residue[i][self.residue_mask])
        rms = np.sqrt(np.mean(np.square(np.array(out))))
        return rms
        