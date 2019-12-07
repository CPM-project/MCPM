import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings

from MCPM.multipletpf import MultipleTpf
from MCPM.campaigngridradec2pix import CampaignGridRaDec2Pix
from MCPM import utils
from MCPM.prfdata import PrfData
from MCPM.prfforcampaign import PrfForCampaign
from MCPM.cpmfitpixel import CpmFitPixel


class CpmFitSource(object):
    """
    Class for performing CPM fit for a source that combines data 
    from a number of pixels
    """
    
    def __init__(self, ra, dec, campaign, channel, 
            multiple_tpf=None, campaign_grids=None, prf_data=None,
            use_uncertainties=True):
        self.ra = ra
        self.dec = dec
        self.campaign = campaign
        self.channel = channel
        self.use_uncertainties = use_uncertainties
        
        if multiple_tpf is None:
            multiple_tpf = MultipleTpf(campaign, channel)
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
        self._predictor_matrix_row = None
        self._predictor_matrix_column = None
        self._l2 = None
        self._l2_per_pixel = None
        self._prf_values = None
        self._prf_values_mask = None
        self._pixel_time = None
        self._pixel_flux = None
        self._pixel_flux_err = None
        self._all_pixels_flux_err = None
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
        """calculate the pixel position of the source for all epochs"""
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
        """
        set pixels to be a square of size (2*half_size+1)^2 around 
        the mean position; e.g., half_size=2 gives 5x5 square
        """
        self._pixels = utils.pixel_list_center(self.mean_x, self.mean_y, 
                half_size)
    
    def get_predictor_matrix(self, n_pixel=None, min_distance=None, 
            exclude=None, median_flux_ratio_limits=None, 
            median_flux_limits=None, n_pca_components=None,
            selected_pixels_file=None):
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
        kwargs_ = {**kwargs}
        if n_pca_components is not None:
            kwargs['n_pca_components'] = n_pca_components
            kwargs_['n_pca'] = n_pca_components
        else:
            kwargs_['n_pca'] = 0
            
        out = self.multiple_tpf.get_predictor_matrix(
                ra=self.ra, dec=self.dec, **kwargs)
        predictor_matrix = out[0]
        predictor_matrix_row = out[2]
        predictor_matrix_column = out[3]

        if selected_pixels_file is not None:
            with fits.open(selected_pixels_file) as hdus:
                np.testing.assert_almost_equal(hdus[0].header['RA'], self.ra)
                np.testing.assert_almost_equal(hdus[0].header['DEC'], self.dec)
                assert hdus[0].header['CHANNEL'] == self.channel
                rows = hdus[1].data.field('row')
                columns = hdus[1].data.field('column')

            indexes = self._get_indexes(
                predictor_matrix_row, predictor_matrix_column, rows, columns)

            predictor_matrix = predictor_matrix[:, indexes]
            predictor_matrix_row = predictor_matrix_row[indexes]
            predictor_matrix_column = predictor_matrix_column[indexes]

        self._predictor_matrix_kwargs = kwargs_
        self._predictor_matrix = predictor_matrix
        self._predictor_matrix_mask = out[1]
        self._predictor_matrix_row = predictor_matrix_row
        self._predictor_matrix_column = predictor_matrix_column

    def _get_indexes(self, predictor_matrix_row, predictor_matrix_column,
                     rows, columns):
        """
        find indexes of all (row, column) pixels in first 2 arrays
        """
        indexes = []
        for (row, column) in zip(rows, columns):
            mask_1 = (row == predictor_matrix_row)
            mask_2 = (column == predictor_matrix_column)
            mask = mask_1 * mask_2
            if sum(mask) != 1:
                print(predictor_matrix_row, "\n", predictor_matrix_column,
                      "\n", row, "\n", column, "\n", mask)
                raise ValueError('something went wrong with reading the ' +
                                 'file with selected pixels')
            indexes.append(np.arange(len(mask))[mask][0])
        return indexes

    def plot_predictor_pixels(self, **kwargs):
        """
        Plot pixels that are used for predictor matrix relative 
        to mean position of the target.
        Use plt.show() or plt.savefig(FILE_NAME) afterwards.
        **kwargs are passed to pyplot.plot. 
        """
        if self._predictor_matrix_row is None:
            raise ValueError('You cannot plot the predictor_matrix pixels' +
                "before running get_predictor_matrix()")
        difference_1 = self._predictor_matrix_column - int(round(self.mean_x))
        difference_2 = self._predictor_matrix_row - int(round(self.mean_y))

        kwargs['linestyle']='None'
        plt.plot(difference_1, difference_2, marker='s', **kwargs)

    @property
    def predictor_matrix(self):
        """matrix of predictor fluxes"""
        if self._predictor_matrix is None:
            msg = 'run get_predictor_matrix() to get predictor matrix'
            raise ValueError(msg)
        return self._predictor_matrix

    @property
    def predictor_matrix_mask(self):
        """epoch mask for matrix of predictor fluxes"""
        if self._predictor_matrix_mask  is None:
            msg = 'run get_predictor_matrix() to get predictor matrix mask'
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
        """
        calculate sum of PRF values for every pixel and select n_select 
        ones with the highest sum
        """
        if n_select > len(self._pixels):
            raise ValueError('selection of too many pixels requested')
        prf_sum = np.sum(self.prf_values[self.prf_values_mask], axis=0)
        sorted_indexes = np.argsort(prf_sum)[::-1][:n_select]
        
        self._pixels = self._pixels[sorted_indexes]
        self._prf_values = self.prf_values[:, sorted_indexes]

        self._pixel_time = None
        self._pixel_flux = None
        self._pixel_flux_err = None
        self._pixel_mask = None
        self._cpm_pixel = None

    def _get_time_flux_mask_for_pixels(self):
        """
        extract time vectors, flux vectors and epoch masks 
        for pixels from TPF files
        """
        out = self.multiple_tpf.get_time_flux_mask_for_pixels(self._pixels)
        reference_time = out[0][0][out[3][0]]
        for i in range(1, self.n_pixels):
            masked_time = out[0][i][out[3][i]]
            if (np.abs(reference_time-masked_time) > 1.e-5).any():
                msg = "the assumed time vectors should be the same\n{:}\n{:}"
                raise ValueError(msg.format(out[0][0], out[0][i]))
        self._pixel_time = out[0][0]
        self._pixel_flux = out[1]
        self._pixel_flux_err = out[2]
        self._pixel_mask = out[3]

        errors = np.array(self._pixel_flux_err)
        self._all_pixels_flux_err = np.sqrt(np.sum(errors**2, axis=0))

    @property
    def pixel_time(self):
        """
        *np.array*

        Time vectors for all pixels.
        These are in BJD TDB and correspond to the middle of the exposure.
        """
        if self._pixel_time is None:
            self._get_time_flux_mask_for_pixels()
        return self._pixel_time
        
    @property
    def pixel_flux(self):
        """flux vectors for all pixels"""
        if self._pixel_flux is None:
            self._get_time_flux_mask_for_pixels()
        return self._pixel_flux

    @property
    def pixel_flux_err(self):
        """flux error vectors for all pixels"""
        if self._pixel_flux_err is None:
            self._get_time_flux_mask_for_pixels()
        return self._pixel_flux_err

    @property
    def all_pixels_flux_err(self):
        """
        uncertainty of combined flux of the few pixels that are
        combined into a source
        """
        if self._all_pixels_flux_err is None:
            self._get_time_flux_mask_for_pixels()
        return self._all_pixels_flux_err

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
        """
        mask epochs with residuals larger than limit or smaller than -limit;
        if limit is not provided then 5*residuals_rms is assumed
        """
        if limit is None:
            limit = 5 * self.residuals_rms
        mask = self.residuals_mask
        mask_bad = (self.residuals[mask]**2 >= limit**2)
        indexes = np.where(mask)[0][mask_bad]
        self.mask_bad_epochs(indexes)                    
    
    def set_train_mask(self, train_mask):
        """sets the epoch mask used for training in CPM"""
        if self._cpm_pixel is None:
            raise ValueError('you must run run_cpm() before set_train_mask()')
        self._train_mask = train_mask
        for i in range(self.n_pixels):
            self._cpm_pixel[i].train_time_mask = self._train_mask

    def pspl_model(self, t_0, u_0, t_E, f_s):
        """Paczynski (or point-source/point-lens) microlensing model"""
        return utils.pspl_model(t_0, u_0, t_E, f_s, self.pixel_time)

    def run_cpm(self, model, model_mask=None):
        """
        Run CPM on all pixels. Model has to be provided for epochs in
        self.pixel_time. If the epoch mask model_mask is None, then it's 
        assumed it's True for each epoch. Mask of PRF is applied inside 
        this function.

        If you want to train the model on only a part of the light-curve,
        then don't change *model_mask* but use *set_train_mask()* instead.
        """
        if self._cpm_pixel is None:
            self._cpm_pixel = [None] * self.n_pixels
        self._pixel_residuals = None
        self._pixel_residuals_mask = None
        self._residuals = None
        self._residuals_mask = None
        self._model = np.copy(model)
       
        if model_mask is None:
            model_mask = np.ones_like(model, dtype=bool)
        model_mask *= self.prf_values_mask

        for i in range(self.n_pixels):
            model_i = model * self.prf_values[:,i]
            
            if self._cpm_pixel[i] is None:
                self._cpm_pixel[i] = CpmFitPixel(
                    target_flux=self.pixel_flux[i], 
                    target_flux_err=self.pixel_flux_err[i], 
                    target_mask=self.pixel_mask[i], 
                    predictor_matrix=self.predictor_matrix, 
                    predictor_matrix_mask=self.predictor_matrix_mask, 
                    l2=self.l2, model=model_i, model_mask=model_mask, 
                    time=self.pixel_time, train_mask=self._train_mask,
                    use_uncertainties=self.use_uncertainties)
            else:
                self._cpm_pixel[i].model = model_i
                self._cpm_pixel[i].model_mask = model_mask

    @property
    def pixel_residuals(self):
        """list of residuals for every pixel"""
        if self._pixel_residuals is None:
            self._pixel_residuals = [None] * self.n_pixels
            self._pixel_residuals_mask = [None] * self.n_pixels
            failed = []
            for i in range(self.n_pixels):
                residuals = np.zeros_like(self.pixel_time)
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

    def pixel_coeffs(self, n_pixel):
        """return CPM coeffs for pixel number n_pixel"""
        if self._cpm_pixel is None:
            raise ValueError("CPM not yet run but you're trying to "
                            + "access its results")
        return self._cpm_pixel[n_pixel].coeffs
        
    def set_pixel_coeffs(self, n_pixel, values):
        """set coeffs for given pixel"""
        if self._cpm_pixel is None:
            raise ValueError("CPM not yet run")        
        self._cpm_pixel[n_pixel].set_coeffs(values)
          
    def read_coeffs_from_fits(self, fits_name):
        """read and store coeffs from a fits file"""
        with fits.open(fits_name) as hdu:
            head = hdu[0].header
            np.testing.assert_almost_equal(head['RA'], self.ra, decimal=3)
            np.testing.assert_almost_equal(head['DEC'], self.dec, decimal=3)
            assert head['campaign'] == self.campaign
            assert head['channel'] == self.channel
            if 'n_pca' in head:
                assert head['n_pca'] == self._predictor_matrix_kwargs['n_pca']
            else:
                assert self._predictor_matrix_kwargs['n_pca'] == 0
            pixels = np.array([[a,b] for (a, b) in hdu[1].data])
            assert np.all([[a,b] for (a, b) in hdu[1].data] == self.pixels) 
            rows = [a[0] for a in hdu[2].data]
            assert np.all(rows == self._predictor_matrix_row)
            columns = [a[1] for a in hdu[2].data]
            assert np.all(columns == self._predictor_matrix_column)
            
            coeffs = np.array([list(v) for v in hdu[3].data]).T
            for (i, coeffs_) in enumerate(coeffs):
                self.set_pixel_coeffs(i, coeffs_)

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
            residuals = np.zeros_like(self.pixel_time)
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
        """
        calculate RMS of residuals combining all pixels and applying 
        additional epoch mask
        """
        mask_all = mask & self.residuals_mask
        rms = np.sqrt(np.mean(np.square(self.residuals[mask_all])))
        return rms

    #def residuals_rms_prf_photometry(self, model, model_mask=None):
        #"""XXX"""
        #(phot, phot_mask) = self.prf_photometry()
        #difference = phot - model
        #if model_mask is None:
            #model_mask = phot_mask
        #rms = np.sqrt(np.mean(np.square(difference[model_mask * phot_mask])))
        #return rms

    def residuals_prf(self):
        """
        residuals calculated with second application of PRF
        (first was to find the model flux for given pixel).
        """
        mask = self.residuals_mask

        prf_flux = np.zeros(np.sum(mask), dtype=float)
        prf_square = np.zeros(np.sum(mask), dtype=float)
        for i in range(self.n_pixels):
            prf = self.prf_values[:,i][mask]
            prf_flux += prf * self._cpm_pixel[i].cpm_residuals[mask]
            prf_square += prf**2
        out = np.zeros_like(mask, dtype=float)
        out[mask] = prf_flux / prf_square - self._model[mask]

        return out

    def prf_photometry(self):
        """
        Performs profile photometry using pixel value with subtracted 
        fitted_flux from CPM. Currently, does not include uncertainties 
        in calculations
        
        Returns flux vector and mask
        """
        mask = self.prf_values_mask
        for cpm_pixel in self._cpm_pixel:
            mask *= cpm_pixel.fitted_flux_mask

        prf_flux = np.zeros(np.sum(mask), dtype=float)
        prf_square = np.zeros(np.sum(mask), dtype=float)
        for i in range(self.n_pixels):
            prf = self.prf_values[:,i][mask]
            prf_flux += prf * self._cpm_pixel[i].cpm_residuals[mask]
            prf_square += prf**2
        out = np.zeros_like(mask, dtype=float)
        out[mask] = prf_flux / prf_square
        
        return (out, mask)

    def prf_photometry_no_CPM(self):
        """
        PRF photometry of the target that does not use CPM approach.
        It only performs single-parameter fit for each epoch:
        flux = SUM f_i * PRF_i / SUM PRF_i^2, where f_i is flux in given
        pixel i and PRF_i is PRF pixel for this pixel.

        You probably want to call subtract_flux_constant() before.

        Returns:
            flux: *np.ndarray*
                Calculated fluxes of the target star.

            mask: *np.ndarray*
                Mask to be applied to *flux* to remove epochs that
                lacked some data.
        """
        mask = self.prf_values_mask
        for m in self.pixel_mask:
            mask *= m

        prf_flux = np.zeros(np.sum(mask), dtype=float)
        prf_square = np.zeros(np.sum(mask), dtype=float)
        for i in range(self.n_pixels):
            prf = self.prf_values[:,i][mask]
            prf_flux += prf * self.pixel_flux[i][mask]
            prf_square += prf**2

        out = np.zeros_like(mask, dtype=float)
        out[mask] = prf_flux / prf_square

        return (out, mask)

    def _save_coeffs_to_fits(self, fits_name, coeffs):
        """save coeffs to a fits file"""
        column_1 = fits.Column(name='x', array=self.pixels[:,0], format='I')
        column_2 = fits.Column(name='y', array=self.pixels[:,1], format='I')
        hdu_1 = fits.BinTableHDU.from_columns([column_1, column_2], 
                name='event_pixels')
                
        column_1 = fits.Column(name='row', 
                array=self._predictor_matrix_row, format='I')
        column_2 = fits.Column(name='column', 
                array=self._predictor_matrix_column, format='I')
        hdu_2 = fits.BinTableHDU.from_columns([column_1, column_2],
                name='predictor_matrix_pixels')
        
        columns = []
        names = ["pix_{:}".format(i) for i in range(self.n_pixels)]
        for i in range(self.n_pixels):
            column = fits.Column(name=names[i], array=coeffs[i], format='E')
            columns.append(column)
        hdu_3 = fits.BinTableHDU.from_columns(columns, name='coeffs')

        sum_prfs = np.sum(self.prf_values[self.prf_values_mask], axis=0)
        column = fits.Column(name='sum_prf', array=sum_prfs, format='E')
        hdu_4 = fits.BinTableHDU.from_columns([column],
                                              name='sum_prf_over_epochs')

        header = fits.Header()
        header['RA'] = self.ra
        header['Dec'] = self.dec
        header['CAMPAIGN'] = self.campaign
        header['CHANNEL'] = self.channel
        header['L2'] = (self._l2, 'REGULARISATION')
        header['code'] = 'https://github.com/CPM-project/MCPM'
        header.update(self._predictor_matrix_kwargs)
        hdu_0 = fits.PrimaryHDU(header=header)
        hdus = fits.HDUList([hdu_0, hdu_1, hdu_2, hdu_3, hdu_4])
        hdus.writeto(fits_name)

    def save_coeffs_to_fits(self, fits_name):
        """save coeffs to a fits file; 
        you probably want run set_pixel_coeffs() before"""
        coeffs = [self.pixel_coeffs(i) for i in range(self.n_pixels)]
        self._save_coeffs_to_fits(fits_name, coeffs=coeffs)

    def plot_pixel_residuals(self, shift=None):
        """Plot residuals for each pixel separately. Parameter
        shift (int or float) sets the shift in Y axis between the pixel,
        the default value is 2*RMS rounded up to nearest 10"""
        if shift is None:
            shift = round(2 * self.residuals_rms + 10./2, -1) # -1 mean "next 10"

        mask = self.residuals_mask
        time = self.pixel_time[mask]
        zeros = np.zeros_like(time)
        for i in range(self.n_pixels):
            mask = self.residuals_mask
            y_values = self.pixel_residuals[i][mask] + i * shift
            plt.plot(time, zeros + i * shift, 'k--')
            plt.plot(time, y_values, '.', label="pixel {:}".format(i))
            
        plt.xlabel("HJD'")
        plt.ylabel("counts + const.")

    def plot_pixel_curves(self, **kwargs):
        """
        Use matplotlib to plot raw data for a set of pixels.
        For options look at MultipleTpf.plot_pixel_curves().
        **kwags can contain 'y_lim' keyword, which is passed to Axis.set_ylim().
        """
        # The code below removes 12 epochs in C91 and 0 in C92.
        flags = self.multiple_tpf.get_quality_flags_for_pixels(self._pixels)
        flags_mask = (flags[0] != 40992)
        for flags_ in flags:
            flags_mask *= (flags_ != 40992)
        pixel_flux = []
        for flux in self.pixel_flux:
            pixel_flux.append(flux[flags_mask])

        self.multiple_tpf.plot_pixel_curves(
                pixels=self._pixels, flux=pixel_flux,
                time_mask=flags_mask, **kwargs)

    def subtract_flux_from_star(self, star_ra, star_dec, flux):
        """
        Subtract the signal expected from star at given coords from all pixels.
        Nothing is returned, it just updates internal variables.

        Parameters :
            star_ra: *float*
                RA - decimal degrees

            star_dec: *float*
                Dec - decimal degrees

            flux: *float* or *np.ndarray*
                Fluxes to be subtracted. If *np.ndarray*, then its size is
                the same as total number of epochs.
        """
        (prf_values, mask) = self.prf_for_campaign.apply_grids_and_prf(
                    ra=star_ra, dec=star_dec, pixels=self._pixels)
        for i in range(self.n_pixels):
            self.pixel_flux[i] -= flux * prf_values[:,i]

    def subtract_flux_constant(self, flux):
        """
        From all target pixels subtract the same value of flux.
        """
        for i in range(self.n_pixels):
            self.pixel_flux[i] -= flux

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
        residuals = np.copy(self.residuals)
        if f_s is None:
            y_label = 'counts'
        else:
            y_label = 'magnification'
            model /= f_s
            residuals /= f_s

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
        """
        inner function that makes the plotting; magnification is plotted
        instead of counts if f_s is provided
        """
        lw = 5
        plt.plot(self.pixel_time[mask], np.zeros_like(pixel_time[mask]), 'k--', lw=lw)

        residuals = np.copy(self.residuals)
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

    def plot_image(self, time, data_type="cpm_residuals", **kwargs):
        """
        Plot small part of an image.

        Parameters :
            time: *float*
                Short-format (ie. no 245) BJD to be searched for.
                The closest epoch will be shown.
            data_type: 'cpm_residuals' or 'pixel_flux'
                Type of data to be plotted. The default 'cpm_residuals' means
                that instrumental trends are removed. The 'pixel_flux' means
                raw data.
            ``**kwargs``
                are passed to pyplot.imshow()
        """
        index = np.nanargmin(np.abs(self.pixel_time-time))
        if np.abs(self.pixel_time[index] - time) > 0.007:
            warnings.warn(
                'large difference between requested and closest epoch:' +
                '{:.5f} '.format(self.pixel_time[index] - time) +
                '{:.5f} {:.5f}'.format(self.pixel_time[index], time))
        x = self.pixels[:,0]
        y = self.pixels[:,1]
        min_x = min(x)
        min_y = min(y)
        image = np.zeros( (max(x)-min_x+1, max(y)-min_y+1) )
        x -= min_x
        y -= min_y

        for i in range(self.n_pixels):
            if data_type == 'cpm_residuals':
                image[x[i], y[i]] = self._cpm_pixel[i].cpm_residuals[index]
            elif data_type == 'pixel_flux':
                image[x[i], y[i]] = self.pixel_flux[i][index]
            else:
                raise ValueError('Unrecognized data_type: ' + data_type)

        plt.title("{:.5f}".format(self.pixel_time[index]))
        plt.imshow(image, **kwargs)
        plt.colorbar()
