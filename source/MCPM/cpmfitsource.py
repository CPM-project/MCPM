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
    
    def get_predictor_matrix(self):
        """calculate predictor_matrix and its mask"""
        out = self.multiple_tpf.get_predictor_matrix(ra=self.ra, dec=self.dec)
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

    def _get_time_flux_mask_for_pixels(self):
        """extract time vectors, flux vectors and epoch masks 
        for pixels from TPF files"""
        out = self.multiple_tpf.get_time_flux_mask_for_pixels(self._pixels)
        self._pixel_time = out[0]
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
        if self._mask is None:
            self._get_time_flux_mask_for_pixels()
        return self._pixel_mask
    
#def cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask,
        #prfs, mask_prfs, model, l2):
    #"""runs CPM on a set of pixels and returns each result"""
    #out_signal = []
    #out_mask = []
    #for i in range(len(tpf_flux)):
        #cpm_pixel = CpmFitPixel(
            #target_flux=tpf_flux[i], target_flux_err=None, target_mask=tpf_epoch_mask[i], 
            #predictor_matrix=predictor_matrix, predictor_mask=predictor_mask,
            #l2=l2, 
            #model=model[i]*prfs[:,i], model_mask=mask_prfs,
            #time=times[i]
        #)
        #out_signal.append(cpm_pixel.residue)
        #out_mask.append(cpm_pixel.results_mask)
    #return (out_signal, out_mask)