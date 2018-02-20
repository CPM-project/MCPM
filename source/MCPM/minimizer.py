import os
import sys
import numpy as np
# for import matplotlib.pyplot -- see below

from MulensModel.utils import Utils


K2_MAG_ZEROPOINT = 25.

class Minimizer(object): 
    """
    An object to link an Event to the functions necessary to minimize chi2.
    
    NOTE that here we assume that theta has an additional parameter: satellite 
    source flux. It is also assumed that the last dataset is the satellite one. 

    To force periodic flush of file with all models set n_flush to 
    100 or 1000 etc.
    """

    def __init__(self, event, parameters_to_fit, cpm_source):
        self.event = event
        self.parameters_to_fit = parameters_to_fit
        self.n_parameters = len(self.parameters_to_fit)
        self.n_parameters += 1
        self.cpm_source = cpm_source
        self.reset_min_chi2()
        self._chi2_0 = None
        self._prior_min_values = None
        self._prior_max_values = None
        self._n_calls = 0
        self._color_constraint = None

        self._file_all_models_name = None
        self._file_all_models = None

        self._sat_mask = None
        self._sat_time = None
        self._sat_model = None
        self._sat_magnification = None
        self._sat_flux = None

        self._coefs_cache = None
        self.n_flush = None

    def close_file_all_models(self):
        """closes the file to which all models are saved"""
        self._file_all_models.close()
        self._file_all_models = None

    @property
    def file_all_models(self):
        """name of the file to save all the models"""
        return self._file_all_models_name

    @file_all_models.setter
    def file_all_models(self, file_name):
        if self._file_all_models_name is not None:
            self.close_file_all_models()
        self._file_all_models_name = file_name
        if self._file_all_models_name is not None:
            self._file_all_models = open(self._file_all_models_name, 'w')

    def reset_min_chi2(self):
        """reset minimum chi2 and corresponding parameters"""
        self._min_chi2 = None
        self._min_chi2_theta = None        

    def print_min_chi2(self):
        """Print minimum chi2 and corresponding values"""
        fmt = " ".join(["{:.4f}"] * self.n_parameters)
        parameteres = fmt.format(*list(self._min_chi2_theta))
        print("{:.3f}  {:}".format(self._min_chi2, parameteres))

    def set_parameters(self, theta):
        """for given event set attributes from parameters_to_fit (list of str) 
        to values from theta list"""
        for (key, val) in enumerate(self.parameters_to_fit):
            setattr(self.event.model.parameters, val, theta[key])

    def _run_cpm(self, theta):
        """set the satellite light curve and run CPM"""
        self.set_parameters(theta)
        self._sat_flux = theta[-1]
        if self._sat_mask is None:
            self._sat_mask = self.cpm_source.residuals_mask
            self._sat_time = self.cpm_source.pixel_time[self._sat_mask] + 2450000.
            self._sat_model = np.zeros(len(self.cpm_source.pixel_time))
        # Here we prepare the satellite lightcurve:
        self._sat_magnification = self.event.model.magnification(
                time = self._sat_time,
                satellite_skycoord = self.event.datasets[-1].satellite_skycoord)
        self._sat_model[self._sat_mask] = self._sat_magnification * self._sat_flux
        self.cpm_source.run_cpm(self._sat_model)

    def set_satellite_data(self, theta):
        """set satellite dataset magnitudes and fluxes"""
        self._run_cpm(theta)
        sat_residuals = self.cpm_source.residuals[self._sat_mask]
        flux = self._sat_model[self._sat_mask] + sat_residuals 
        self.event.datasets[-1].flux = flux
        mag_and_err = Utils.get_mag_and_err_from_flux(flux, 
                self.event.datasets[-1].err_flux, zeropoint=K2_MAG_ZEROPOINT)
        self.event.datasets[-1]._mag = mag_and_err[0]
        self.event.datasets[-1]._err_mag = mag_and_err[1]

    def add_color_constraint(self, ref_dataset, ref_zero_point, color, sigma_color):
        """
        Specify parameters that are used to constrain the source flux in 
        satellite band: 
            ref_dataset (int) - reference dataset
            ref_zero_point (float) - magnitude zeropoint of reference dataset
            color (float) - (satellite-ref_dataset) color (NOTE ORDER)
            sigma_color (float) - sigma of color        
        """
        self._color_constraint = [ref_dataset, ref_zero_point, color, sigma_color]
        
    def _chi2_for_color_constraint(self, satellite_flux):
        """calculate chi2 for flux constraint"""
        (ref_dataset, ref_zero_point, color, sigma_color) = self._color_constraint 
        
        before_ref = self.event.data_ref
        flux_ref = self.event.get_ref_fluxes(ref_dataset)[0][0]
        self.event.get_ref_fluxes(before_ref)
        
        mag_ref = ref_zero_point - 2.5 * np.log10(flux_ref)
        mag_sat = K2_MAG_ZEROPOINT - 2.5 * np.log10(satellite_flux)
        color_value = mag_sat - mag_ref
        out = ((color_value - color) / sigma_color)**2
        return out
        
    def chi2_fun(self, theta):
        """for a given set of parameters (theta), return the chi2"""
        self._run_cpm(theta)
        chi2 = (self.cpm_source.residuals_rms / np.mean(self.event.datasets[-1].err_flux))**2
        chi2 *= np.sum(self._sat_mask)
        n = len(self.event.datasets) - 1 # We subtract 1 because satellite 
        # chi2 is calculated above
        self.chi2 = [self.event.get_chi2_for_dataset(i) for i in range(n)]
        self.chi2.append(chi2)
        if self._color_constraint is not None:
            self.chi2.append(self._chi2_for_color_constraint(theta[-1]))
        chi2 = sum(self.chi2)
        if self._min_chi2 is None or chi2 < self._min_chi2:
            self._min_chi2 = chi2
            self._min_chi2_theta = theta
        self._n_calls += 1
        if self._file_all_models is not None:
            text = " ".join([repr(chi2)] + [repr(ll) for ll in theta]) + '\n'
            self._file_all_models.write(text)
            if self.n_flush is not None and self._n_calls % self.n_flush == 0:
                self._file_all_models.flush()
                os.fsync(self._file_all_models.fileno()) 
        if self._coefs_cache is not None:
            self._coefs_cache[tuple(theta.tolist())] = np.array([
                    self.cpm_source.pixel_coeffs(j).flatten() for j in range(self.cpm_source.n_pixels)])
        return chi2

    def set_chi2_0(self, chi2_0=None):
        """set reference value of chi2"""
        if chi2_0 is None:
            #chi2_0 = np.sum([d.n_epochs for d in self.event.datasets])
            chi2_0 = np.sum([np.sum(d.good) for d in self.event.datasets])
        self._chi2_0 = chi2_0
    
    def set_pixel_coeffs_from_dicts(self, coeffs, weights=None):
        """
        Take coeffs, average them, and set pixel coeffs to the averages.
       
        Arguments :
            coeffs: *dict*
                Dictionary of pixel coeffs. Each value specifies coeffs for all 
                pixels i.e., coeffs[key][j] is for j-th pixel. The keys can be 
                whatever, but most probably you want 
                tuple(list(model_parameters)) to be the keys.
            weights: *dict*, optional
                Dictionary of weights - uses the same keys as coeffs.
        """
        keys = list(coeffs.keys())
        if weights is None:
            weights_ = None
        else:
            weights_ = [weights[key] for key in keys]
        
        for j in range(self.cpm_source.n_pixels):
            data = [coeffs[key][j] for key in keys]
            average = np.average(np.array(data), 0, weights=weights_)
            self.cpm_source.set_pixel_coeffs(j, average.reshape((-1, 1)))

    def set_pixel_coeffs_from_models(self, models, weights=None):
        """run a set of models, remember the coeffs for every pixel,
        then average them (using weights) and remember
        
        NOTE: this version may be not very stable numerically. Try using 
        e.g., set_pixel_coeffs_from_dicts()
        """
        n_models = len(models)
        shape = (n_models, self.cpm_source.predictor_matrix.shape[1])
        coeffs = [np.zeros(shape) for i in range(self.cpm_source.n_pixels)]
        
        for i in range(n_models):
            self._run_cpm(models[i])
            for j in range(self.cpm_source.n_pixels):
                coeffs[j][i] = self.cpm_source.pixel_coeffs(j).reshape(shape[1])
                
        for j in range(self.cpm_source.n_pixels):
            average = np.average(coeffs[j], 0, weights=weights)
            self.cpm_source.set_pixel_coeffs(j, average.reshape((-1, 1)))
    
    def start_coeffs_cache(self):
        """
        Start internally remembering coeffs; also resets cache if cacheing was 
        working.
        """
        self._coefs_cache = dict()

    def get_cached_coeffs(self, theta):
        """
        Get pixel coeffs for model defined by theta; note that 
        theta = tuple(list(model_parameters))
        """
        if self._coefs_cache is None:
            raise ValueError("You want to get cached values and you haven't " + 
                "turned on caching (see start_coeffs_cache())? Strange...")
        if not isinstance(theta, tuple):
            raise TypeError('wrong type of get_cached_coeffs() input: \n' +
                    'got {:}, expected tuple'.format(type(theta)))
        return self._coefs_cache[theta]

    def stop_coeffs_cache(self):
        """turn off internally remembering coeffs"""
        self._coefs_cache = None

    def set_prior_boundaries(self, parameters_min_values, 
            parameters_max_values):
        """remebers 2 dictionaries that set minimum and maximum values of 
        parameters"""
        self._prior_min_values = parameters_min_values
        self._prior_max_values = parameters_max_values
        
    def ln_prior(self, theta):
        """return 0 in most cases, or -np.inf if beyond ranges provided"""
        inside = 0.
        outside = -np.inf
        
        if self._prior_min_values is not None:
            for (parameter, value) in self._prior_min_values.items():
                try:
                    index = self.parameters_to_fit.index(parameter)
                except ValueError:
                    index = -1 # no better idea right now for passing f_s_sat
                if theta[index] < value:
                    return outside

        if self._prior_max_values is not None:
            for (parameter, value) in self._prior_max_values.items():
                try:
                    index = self.parameters_to_fit.index(parameter)
                except ValueError:
                    index = -1 # no better idea right now for passing f_s_sat
                if theta[index] > value:
                    return outside
        
        return inside

    def ln_like(self, theta):
        """logarithm of likelihood"""
        return -0.5 * (self.chi2_fun(theta) - self._chi2_0) #/ 10.

    def ln_prob(self, theta):
        """combines prior and likelihood"""
        ln_prior = self.ln_prior(theta)
        if not np.isfinite(ln_prior):
            return -np.inf
        ln_like = self.ln_like(theta)
        if np.isnan(ln_like):
            return -np.inf

        return ln_prior + ln_like
        
    def set_MN_cube(self, min_values, max_values):
        """remembers how to transform unit cube to physical parameters for MN"""
        self._MN_cube = [(min_values[i], (max_values[i]-min_values[i])) 
                            for i in range(self.n_parameters)]
        
    def transform_MN_cube(self, cube):
        """transform unit cube to physical parameters"""
        out = []
        for i in range(len(cube)):
            (zero_point, range_) = self._MN_cube[i]
            out.append(zero_point + range_ * cube[i])
        return np.array(out)

    def satellite_maximum(self):
        """return time of maximum magnification, its value, and corresponding 
        flux for the satellite dataset; takes into account the epochs when 
        satellite data exist"""
        index = np.argmax(self._sat_magnification)
        return (self._sat_time[index], self._sat_magnification[index], self._sat_magnification[index])

    def plot_sat_magnitudes(self, **kwargs):
        """Plot satellite data in reference magnitude system"""
        import matplotlib.pyplot as plt
        data_ref = self.event.model.data_ref
        (fs, fb) = self.event.model.get_ref_fluxes()
        (fs_sat, fb_sat) = self.event.model.get_ref_fluxes(-1)
        self.event.model.data_ref = data_ref
        
        times = self._sat_time - 2450000.
        flux = (self._sat_magnification * self._sat_flux - fb_sat) * (fs[0] / fs_sat[0]) + fb
        
        plt.plot(times, Utils.get_mag_from_flux(flux), **kwargs)
