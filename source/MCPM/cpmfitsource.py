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
        self._predictor_matrix = None
        self._predictor_matrix_mask = None
    
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
        