from os import path
import numpy as np

from MCPM.gridradec2pix import GridRaDec2Pix


class CampaignGridRaDec2Pix(object):
    """collects all grids for given (sub-)campaign"""

    def __init__(self, campaign=None, channel=None, file_name=None):
        if campaign is None:
            self.campaign = None
        else:
            self.campaign = int(campaign)

        self.grids = None
        
        if file_name is None:
            if campaign is None and channel is None:
                raise ValueError('Not enough information in ' + 
                                                    'CampaignGridRaDec2Pix')
            path_ = path.abspath(__file__)
            for i in range(3):
                path_ = path.dirname(path_)
            name = "grids_RADEC2pix_{:}_{:}.data".format(campaign, channel)
            if campaign in [91, 92]:
                k2c = "K2C9"
            elif campaign in [111, 112]:
                k2c = "K2C11"
            else:
                raise ValueError('Unrecognized campaign: {:}'.format(campaign))
            file_name = path.join(path_, "data", k2c, 'grids', name)

        self._read_from_file(file_name, campaign)

        if self.grids is None:
            raise ValueError('no input data provided')

        self.bjd_array = np.array(self.bjd)
        self.n_epochs = len(self.cadence)

    def _read_from_file(self, file_name, campaign):
        """read multiple grids from a single file"""
        if campaign == 91:
            cadence_begin = 125243
            cadence_end = 126532
        elif campaign == 92:
            cadence_begin = 126713
            cadence_end = 128734
        elif campaign == 111:
            cadence_begin = 132839
            cadence_end = 133979
        elif campaign == 112:
            cadence_begin = 134134
            cadence_end = 136469
        else:
            msg = 'no internal data for campaign {:}'.format(campaign)
            raise ValueError(msg)
        
        lengths = cadence_end + 1 - cadence_begin
        bjd = [None] * lengths
        stars_used = [None] * lengths
        sigma = [None] * lengths
        self.grids = [None] * lengths
        cadence = np.arange(cadence_begin, cadence_end+1)
        
        with open(file_name) as in_data:
            for line in in_data.readlines():
                if line[0] == "#":
                    continue
                columns = line.split()
                index = int(columns[0]) - cadence_begin
                self.grids[index] = GridRaDec2Pix(coeffs_x=columns[4:10], 
                                                  coeffs_y=columns[10:16])
                bjd[index] = float(columns[1])
                stars_used[index] = int(columns[2])
                sigma[index] = float(columns[3])
                
        self.bjd = np.array(bjd)
        self.stars_used = np.array(stars_used)
        self.sigma = np.array(sigma)
        self.cadence = cadence
        self.mask = np.ones_like(self.bjd, dtype=bool)
        for (i, value) in enumerate(self.bjd):
            if value is None:
                self.mask[i] = False

    def index_for_bjd(self, bjd):
        """find index that is closest to given BJD"""
        bjds_masked = self.bjd[self.mask]
        index = (np.abs(bjds_masked - bjd)).argmin()
        if np.abs(bjds_masked[index]-bjd) > 2.e-4:
            text = 'No close BJD found for {:}, the closest differs by {:}'
            message = text.format(bjd, np.abs(bjds_masked[index]-bjd))
            raise ValueError(message)
        return np.argwhere(self.bjd == bjds_masked[index])

    def apply_grids(self, ra, dec):
        """Calculate pixel coordinates for given (RA,Dec) for all epochs. 
        (RA,Dec) can be floats, lists, or numpy.arrays.
        
        For input of length n_stars the output is 2 arrays (first for X, 
        second for Y), each of shape (n_epochs, n_stars).
        If inputs are floats than output is 2 1D arrays.
        """
        out_1 = []
        out_2 = []
        for grid in self.grids:
            if grid is None:
                out_1.append(None)
                out_2.append(None)
                continue
            out = grid.apply_grid(ra=ra, dec=dec)
            out_1.append(out[0])
            out_2.append(out[1])
        if isinstance(ra, float):
            out_1 = np.array([v[0] if v is not None else None for v in out_1])
            out_2 = np.array([v[0] if v is not None else None for v in out_2])
        return (out_1, out_2)

    def mean_position(self, ra, dec):
        """for given (RA,Dec) calculate position for every epoch
        and report weighted average"""
        if not isinstance(ra, float) or not isinstance(dec, float):
            raise TypeError('mean_position() arguments have to be floats')
        (pixel_x, pixel_y) = self.apply_grids(ra=ra, dec=dec)
        weights = self.sigma[self.mask]**-2
        mean_x = (pixel_x[self.mask] * weights).sum() / weights.sum()
        mean_y = (pixel_y[self.mask] * weights).sum() / weights.sum()
        return (mean_x, mean_y)
        
    def mean_position_clipped(self, ra, dec, radius=2):
        """get mean position with input data narrowed down to radius"""
        (pixel_x, pixel_y) = self.apply_grids(ra, dec)
        weights = self.sigma[self.mask]**-2
        mean_x = (pixel_x[self.mask] * weights).sum() / weights.sum()
        mean_y = (pixel_y[self.mask] * weights).sum() / weights.sum()
        mask = ((pixel_x[self.mask]-mean_x)**2 + (pixel_y[self.mask]-mean_y)**2 < radius**2)
        weights = weights[mask]
        mean_x = (pixel_x[self.mask][mask] * weights).sum() / weights.sum()
        mean_y = (pixel_y[self.mask][mask] * weights).sum() / weights.sum()
        mask_out = np.copy(self.mask)
        mask_out[self.mask] = mask
        return (mean_x, mean_y, mask_out)
        
