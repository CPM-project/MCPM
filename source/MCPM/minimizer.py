import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from MulensModel.utils import Utils


K2_MAG_ZEROPOINT = 25.

# Multiple cpm_source-s to be done:
#     def set_pixel_coeffs_from_models
#     def satellite_maximum

class Minimizer(object): 
    """
    An object to link an Event to the functions necessary to minimize chi2.
    
    NOTE that here we assume that theta has an additional parameter: satellite 
    source flux. It is also assumed that the last dataset is the satellite one. 

    To force periodic flush of file with all models set n_flush to 
    100 or 1000 etc.
    """

    def __init__(self, event, parameters_to_fit, cpm_sources):
        self.event = event
        self.n_datasets = len(self.event.datasets)
        self.parameters_to_fit = parameters_to_fit
        self.n_parameters = len(self.parameters_to_fit)
        self.n_parameters += 1
        if not isinstance(cpm_sources, list):
            cpm_sources = [cpm_sources]
        self.cpm_sources = cpm_sources
        self.n_sat = len(self.cpm_sources)
        self.reset_min_chi2()
        self._chi2_0 = None
        self._prior_min_values = None
        self._prior_max_values = None
        self._n_calls = 0
        self._color_constraint = None

        self._file_all_models_name = None
        self._file_all_models = None

        self._sat_masks = None
        self._sat_times = None
        self._sat_models = None
        self._sat_magnifications = None
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
        n_0 = self.n_datasets - self.n_sat

        if self._sat_masks is None:
            self._sat_masks = [cpm_source.residuals_mask for cpm_source in self.cpm_sources]
            self._sat_times = [self.cpm_sources[i].pixel_time[self._sat_masks[i]] + 2450000. for i in range(self.n_sat)]
            self._sat_models = [np.zeros(len(cpm_source.pixel_time)) for cpm_source in self.cpm_sources]
            self._sat_magnifications = [None] * self.n_sat

        for i in range(self.n_sat):
            # Here we prepare the satellite lightcurves:
            self._sat_magnifications[i] = self.event.model.magnification(
                    time=self._sat_times[i],
                    satellite_skycoord=self.event.datasets[n_0+i].satellite_skycoord)
            self._sat_models[i][self._sat_masks[i]] = self._sat_magnifications[i] * self._sat_flux
            self.cpm_sources[i].run_cpm(self._sat_models[i])

    def set_satellite_data(self, theta):
        """set satellite dataset magnitudes and fluxes"""
        self._run_cpm(theta)
        n_0 = self.n_datasets - self.n_sat
        for i in range(self.n_sat):
            ii = n_0 + i
            sat_residuals = self.cpm_sources[i].residuals[self._sat_masks[i]]
            flux = self._sat_models[i][self._sat_masks[i]] + sat_residuals
            self.event.datasets[ii].flux = flux
            mag_and_err = Utils.get_mag_and_err_from_flux(flux,
                self.event.datasets[ii].err_flux, zeropoint=K2_MAG_ZEROPOINT)
            self.event.datasets[ii]._mag = mag_and_err[0]
            self.event.datasets[ii]._err_mag = mag_and_err[1]

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
        n = self.n_datasets - self.n_sat
        chi2_sat = [np.sum(self._sat_masks[i])*(self.cpm_sources[i].residuals_rms/np.mean(self.event.datasets[n+i].err_flux))**2 for i in range(self.n_sat)]
        self.chi2 = [self.event.get_chi2_for_dataset(i) for i in range(n)]
        self.chi2 += chi2_sat
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
            coeffs = []
            for i in range(self.n_sat):
                n_pixels = self.cpm_sources[i].n_pixels
                c = [self.cpm_sources[i].pixel_coeffs(j).flatten() for j in range(n_pixels)]
                coeffs.append(np.array(c))
            self._coefs_cache[tuple(theta.tolist())] = coeffs
            #np.array([self.cpm_source.pixel_coeffs(j).flatten() for j in range(self.cpm_source.n_pixels)])
        return chi2

    def set_chi2_0(self, chi2_0=None):
        """set reference value of chi2"""
        if chi2_0 is None:
            #chi2_0 = np.sum([d.n_epochs for d in self.event.datasets])
            chi2_0 = np.sum([np.sum(d.good) for d in self.event.datasets])
        self._chi2_0 = chi2_0
    
    def set_pixel_coeffs_from_samples(self, samples):
        """
        Provide a matrix samples[n_models, n_params] and for each 
        model there get the cached coeffs (caching MUST be turned ON)
        and set pixel coeffs to the mean of these cached coeffs. 
        You may want to run stop_coeffs_cache() afterwards.
        """
        weights = dict()
        coeffs = dict()
        for sample in samples:
            key = tuple(sample.tolist())
            if key in weights:
                weights[key] += 1
            else:
                weights[key] = 1
                coeffs[key] = self.get_cached_coeffs(key)
        self.set_pixel_coeffs_from_dicts(coeffs, weights)

    def set_pixel_coeffs_from_dicts(self, coeffs, weights=None):
        """
        Take coeffs, average them, and set pixel coeffs to the averages.
       
        Arguments :
            coeffs: *dict*
                Dictionary of pixel coeffs. Each value specifies a list 
                (length same as number of satellite datasets) of coeffs for all 
                pixels i.e., coeffs[key][i][j] is for j-th pixel in i-th cpm_source. 
                The keys can be whatever, but most probably you want 
                tuple(list(model_parameters)) to be the keys.
            weights: *dict*, optional
                Dictionary of weights - uses the same keys as coeffs.
        """
        keys = list(coeffs.keys())
        if weights is None:
            weights_ = None
        else:
            weights_ = [weights[key] for key in keys]
       
        for i in range(self.n_sat):
            for j in range(self.cpm_sources[i].n_pixels):
                data = [coeffs[key][i][j] for key in keys]
                average = np.average(np.array(data), 0, weights=weights_)
                self.cpm_sources[i].set_pixel_coeffs(j, average.reshape((-1, 1)))

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

    def save_coeffs_to_fits(self, files):
        """saves coeffs to fits files"""
        for (file_, cpm_source) in zip(files, self.cpm_sources):
            cpm_source.save_coeffs_to_fits(file_)

    def read_coeffs_from_fits(self, files):
        """read coeffs from fits files"""
        for (file_, cpm_source) in zip(files, self.cpm_sources):
            cpm_source.read_coeffs_from_fits(file_)

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
        data_ref = self.event.model.data_ref
        (fs, fb) = self.event.model.get_ref_fluxes()
        n = self.n_datasets - self.n_sat

        for i in range(self.n_sat):
            times = self._sat_times[i] - 2450000.
            (fs_sat, fb_sat) = self.event.model.get_ref_fluxes(n+i)
            mags = self._sat_magnifications[i]
            flux = (mags * self._sat_flux - fb_sat) * (fs[0] / fs_sat[0]) + fb
            plt.plot(times, Utils.get_mag_from_flux(flux), **kwargs)
        self.event.model.data_ref = data_ref

    def standard_plot(self, t_start, t_stop, ylim, title=None):
        """Make plot of the event and residuals. """
        grid_spec = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        plt.figure()
        plt.subplot(grid_spec[0])
        if title is not None:
            plt.title(title)
        alphas = [0.35] * self.n_datasets
        for i in range(self.n_sat):
            alphas[-(i+1)] = 1.
            
        self.event.plot_model(
            color='black', subtract_2450000=True, 
            t_start=t_start+2450000., t_stop=t_stop+2450000.)
        self.plot_sat_magnitudes(color='yellow')
        
        self.event.plot_data(alpha_list=alphas, 
            zorder_list=np.arange(self.n_datasets, 0, -1), 
            marker='o', markersize=5, subtract_2450000=True)
        plt.ylim(ylim[0], ylim[1])
        plt.xlim(t_start, t_stop)
        
        plt.subplot(grid_spec[1])
        self.event.plot_residuals(subtract_2450000=True)
        plt.xlim(t_start, t_stop)
