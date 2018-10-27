import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines

from MulensModel.utils import Utils, MAG_ZEROPOINT
from MulensModel import Trajectory


K2_MAG_ZEROPOINT = 25.

# Multiple cpm_source-s to be done:
#     def set_pixel_coeffs_from_models
#     def satellite_maximum

# Issues with flux uncertsinteis read from TPF files:
#  - not yet properly used in chi2_fun()
#  - plot functions here and in cpmfitsource.py
#  - also CpmFitPixel.target_masked needs _err equivalent

class Minimizer(object): 
    """
    An object to link an Event to the functions necessary to minimize chi2.
    
    Arguments :
        event: *MulensModel.Event*
            ...
            It is assumed that the last datasets are the satellite ones.
            
        parameters_to_fit: *list* of *str*
            Parameters that will be fitted. Except 
            the *MulensModel.ModelParameters* parameters one can use satellite 
            source fluxes: 'f_s_sat' and  'f_b_sat'.
            
        cpm_sources: *CpmFitSource* or *list* of them
    
    To force periodic flush of file with all models set n_flush to 
    100 or 1000 etc.
    
    """
    def __init__(self, event, parameters_to_fit, cpm_sources):
        self.event = event
        self.n_datasets = len(self.event.datasets)
        self.parameters_to_fit = parameters_to_fit
        self.n_parameters = len(self.parameters_to_fit)
        #self.n_parameters += 1
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
        self.model_masks = [None] * self.n_sat

        self._file_all_models_name = None
        self._file_all_models = None
        self.save_fluxes = True

        self._sat_masks = None
        self._sat_times = None
        self._sat_models = None
        self._sat_magnifications = None
        self._sat_source_flux = None
        self._sat_blending_flux = 0.

        self._coeffs_cache = None
        self.n_flush = None

        self.other_constraints = dict()

        self.sigma_scale = 1.

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
        parameters = fmt.format(*list(self._min_chi2_theta))
        print("{:.3f}  {:}".format(self._min_chi2, parameters))

    def set_parameters(self, theta):
        """
        for given event set attributes from parameters_to_fit (list of str) 
        to values from theta list
        """
        if len(self.parameters_to_fit) != len(theta):
            raise ValueError('wrong number of parameters {:} vs {:}'.format(
                    len(self.parameters_to_fit), len(theta)))
        for (i, param) in enumerate(self.parameters_to_fit):
            if param == 'f_s_sat':
                self._sat_source_flux = theta[i]
            elif param == 'f_b_sat':
                self._sat_blending_flux = theta[i]
            else:
                setattr(self.event.model.parameters, param, theta[i])

    def _run_cpm(self, theta):
        """set the satellite light curve and run CPM"""
        self.set_parameters(theta)
        #self._sat_source_flux = theta[-1]
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
            self._sat_models[i][self._sat_masks[i]] = (self._sat_blending_flux +
                    # f_s*(A-1) version:
                    (self._sat_magnifications[i] - 1.) * self._sat_source_flux)
                    # f_s*A version:
                    #self._sat_magnifications[i] * self._sat_source_flux)
            self.cpm_sources[i].run_cpm(self._sat_models[i], 
                    model_mask=self.model_masks[i])

    def set_satellite_data(self, theta):
        """set satellite dataset magnitudes and fluxes"""
        self._run_cpm(theta)
        n_0 = self.n_datasets - self.n_sat
        for i in range(self.n_sat):
            ii = n_0 + i
            # OLD:
            sat_residuals = self.cpm_sources[i].residuals[self._sat_masks[i]]
            flux = self._sat_models[i][self._sat_masks[i]] + sat_residuals
            # NEW - IF ACCEPTED!!! :
            #sat_residuals = self.cpm_sources[i].residuals[self.cpm_sources[i].residuals_mask]
            #flux = self._sat_models[i][self.cpm_sources[i].residuals_mask] + sat_residuals
            self.event.datasets[ii].flux = flux
            mag_and_err = Utils.get_mag_and_err_from_flux(flux,
                self.event.datasets[ii].err_flux, zeropoint=K2_MAG_ZEROPOINT)
            self.event.datasets[ii]._mag = mag_and_err[0]
            self.event.datasets[ii]._err_mag = mag_and_err[1]

    def add_full_color_constraint(self,
            ref_dataset_0, ref_dataset_1, ref_dataset_2, 
            polynomial_2, sigma, ref_zero_point_0=MAG_ZEROPOINT, 
            ref_zero_point_1=MAG_ZEROPOINT, ref_zero_point_2=MAG_ZEROPOINT):
        """
        Specify parameters that are used to constrain the source flux in 
        satellite band:
                Kp-m0 = polynomial_value(m1-m2)
        In common case (e.g., Zhu+17 method: Kp-I = f(V-I)) m0 can be equal to 
        m1 or m2.
            ref_dataset_0 (int) - dataset to calculate satellite color
            ref_dataset_1 (int) - first dataset for ground-based color
            ref_dataset_2 (int) - second dataset for ground-based color
            polynomial (np.array of floats) - color polynomial coefficients: 
                    a0, a1, a2, that will be translated to 
                    a0 + a1*(m1-m2) + a2*(m1-m2)**2
            sigma (float) - scatter or color for the constraint
            ref_zero_point_0 (float) - defines magnitude scale for 0-th dataset
            ref_zero_point_1 (float) - defines magnitude scale for 1-st dataset
            ref_zero_point_2 (float) - defines magnitude scale for 2-nd dataset
        """
        self._color_constraint = [ref_dataset_0, ref_dataset_1, ref_dataset_2, 
            ref_zero_point_0, ref_zero_point_1, ref_zero_point_2, polynomial_2, 
            sigma]

    def add_color_constraint(self, ref_dataset, ref_zero_point, color, sigma_color):
        """
        Specify parameters that are used to constrain the source flux in 
        satellite band: 
            ref_dataset (int) - reference dataset
            ref_zero_point (float) - magnitude zeropoint of reference dataset
                                    probably MulensModel.utils.MAG_ZEROPOINT
            color (float) - (satellite-ref_dataset) color (NOTE ORDER)
            sigma_color (float) - sigma of color        
        """
        self._color_constraint = [ref_dataset, ref_zero_point, color, sigma_color]
        
    def _chi2_for_color_constraint(self, satellite_flux):
        """calculate chi2 for flux constraint"""
        before_ref = self.event.data_ref
        if len(self._color_constraint) == 4:
            (ref_dataset, ref_zero_point, color, sigma_color) = self._color_constraint 
        elif len(self._color_constraint) == 8:
            (ref_dataset, ref_dataset_1, ref_dataset_2) = self._color_constraint[:3]
            (ref_zero_point, ref_zero_point_1, ref_zero_point_2) = self._color_constraint[3:6]
            (polynomial, sigma_color) = self._color_constraint[6:]
            flux_1 = self.event.get_ref_fluxes(ref_dataset_1)[0][0]
            flux_2 = self.event.get_ref_fluxes(ref_dataset_2)[0][0]
            mag_1 = ref_zero_point_1 - 2.5 * np.log10(flux_1)
            mag_2 = ref_zero_point_2 - 2.5 * np.log10(flux_2)
            c = mag_1 - mag_2
            x = 1.
            color = 0.
            for p in polynomial:
                color += x * p
                x *= c
        else:
            raise ValueError('wrong size of internal variable')
        
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
        chi2_sat = []
        for source in self.cpm_sources:
            residuals = source.residuals_prf()[source.residuals_mask]
            # OLD:
            #residuals = source.residuals[source.residuals_mask]
            sigma = source.all_pixels_flux_err[source.residuals_mask]
            sigma *= self.sigma_scale
            chi2_sat.append(np.sum((residuals/sigma)**2))
        # Correct the line below.
        #chi2_sat = [np.sum(self._sat_masks[i])*(self.cpm_sources[i].residuals_rms/np.mean(self.event.datasets[n+i].err_flux))**2 for i in range(self.n_sat)]
        # We also tried:
        #chi2_sat = 0.
        #for i in range(self.n_sat):
            #ii = n + i
            #rms = self.cpm_sources[i].residuals_rms_prf_photometry(self._sat_models[i])
            #rms /= np.mean(self.event.datasets[n+i].err_flux)
            #chi2_sat += np.sum(self._sat_masks[i]) * rms**2        
        # fit_blending=False :
        #self.chi2 = [self.event.get_chi2_for_dataset(i, fit_blending=False) for i in range(n)]
        self.chi2 = [self.event.get_chi2_for_dataset(i) for i in range(n)]
        self.chi2 += chi2_sat
        if self._color_constraint is not None:
            self.chi2.append(self._chi2_for_color_constraint(self._sat_source_flux))
        chi2 = sum(self.chi2)
        if self._min_chi2 is None or chi2 < self._min_chi2:
            self._min_chi2 = chi2
            self._min_chi2_theta = theta
        self._n_calls += 1
        if self.save_fluxes:
            self.event.get_chi2_per_point()
            self.fluxes = 2 * n * [0.]
            self.fluxes[::2] = [self.event.fit.flux_of_sources(self.event.datasets[i])[0] for i in range(n)]
            self.fluxes[1::2] = [self.event.fit.blending_flux(self.event.datasets[i]) for i in range(n)]
        if self._file_all_models is not None:
            text = " ".join([repr(chi2)] + [repr(ll) for ll in theta])
            if self.save_fluxes:
                text += " " + " ".join(["{:.5f}".format(f) for f in self.fluxes])
            self._file_all_models.write(text + '\n')
            if self.n_flush is not None and self._n_calls % self.n_flush == 0:
                self._file_all_models.flush()
                os.fsync(self._file_all_models.fileno()) 
        if self._coeffs_cache is not None:
            coeffs = []
            for i in range(self.n_sat):
                n_pixels = self.cpm_sources[i].n_pixels
                c = [self.cpm_sources[i].pixel_coeffs(j).flatten() for j in range(n_pixels)]
                coeffs.append(np.array(c))
            self._coeffs_cache[tuple(theta.tolist())] = coeffs
        
        return chi2

    def set_chi2_0(self, chi2_0=None):
        """set reference value of chi2"""
        if chi2_0 is None:
            chi2_0 = np.sum([np.sum(d.good) for d in self.event.datasets])
        self._chi2_0 = chi2_0
    
    def set_pixel_coeffs_from_samples(self, samples):
        """
        Provide a matrix samples[n_models, n_params] and for each 
        model there get the cached coeffs (caching MUST be turned ON)
        and set pixel coeffs to the mean of these cached coeffs. 
        You may want to run stop_coeffs_cache() afterward.
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
        if self.n_sat > 1:
            raise ValueError("set_pixel_coeffs_from_models() doesn't allow " +
                "multiple cpm_sources")
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
        Start internally remembering coeffs; also resets cache if caching was 
        working.
        """
        self._coeffs_cache = dict()

    def get_cached_coeffs(self, theta):
        """
        Get pixel coeffs for model defined by theta; note that 
        theta = tuple(list(model_parameters))
        """
        if self._coeffs_cache is None:
            raise ValueError("You want to get cached values and you haven't " + 
                "turned on caching (see start_coeffs_cache())? Strange...")
        if not isinstance(theta, tuple):
            raise TypeError('wrong type of get_cached_coeffs() input: \n' +
                    'got {:}, expected tuple'.format(type(theta)))
        return self._coeffs_cache[theta]

    def stop_coeffs_cache(self):
        """turn off internally remembering coeffs"""
        self._coeffs_cache = None

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
        """
        remembers 2 dictionaries that set minimum and maximum values of 
        parameters
        """
        self._prior_min_values = parameters_min_values
        self._prior_max_values = parameters_max_values
        
    def ln_prior(self, theta):
        """return 0 in most cases, or -np.inf if beyond ranges provided"""
        inside = 0.
        outside = -np.inf
        
        if self._prior_min_values is not None:
            for (parameter, value) in self._prior_min_values.items():
                index = self.parameters_to_fit.index(parameter)
                if theta[index] < value:
                    return outside

        if self._prior_max_values is not None:
            for (parameter, value) in self._prior_max_values.items():
                index = self.parameters_to_fit.index(parameter)
                if theta[index] > value:
                    return outside

        for (key, value) in self.other_constraints.items():
            if key == 't_0':
                t_0_1 = theta[self.parameters_to_fit.index('t_0_1')]
                t_0_2 = theta[self.parameters_to_fit.index('t_0_2')]
                if value == 't_0_1 < t_0_2':
                    if t_0_1 >= t_0_2:
                        return outside
                elif value == 't_0_1 > t_0_2':
                    if t_0_2 >= t_0_1:
                        return outside
                else:
                    raise ValueError('urecognized value: {:}'.format(value))
            elif key == 'min_blending_flux':
                (data, limit) = value
                index = self.event.datasets.index(data)
                self.event.get_chi2_for_dataset(index)
                if self.event.fit.blending_flux(data) <= limit:
                    return outside
            else:
                raise KeyError('unkown constraint: {:}'.format(key))

        return inside

    def ln_like(self, theta):
        """logarithm of likelihood"""
        chi2 = self.chi2_fun(theta)

        ln_likelihood = -0.5 * (chi2 - self._chi2_0)

        return ln_likelihood

    def ln_prob(self, theta):
        """combines prior and likelihood"""
        ln_prior = self.ln_prior(theta)
        if not np.isfinite(ln_prior):
            if self.save_fluxes:
                return (-np.inf, self.fluxes)
            else:
                return -np.inf
        
        ln_like = self.ln_like(theta)
        if np.isnan(ln_like):
            if self.save_fluxes:
                return (-np.inf, self.fluxes)
            else:
                return -np.inf

        ln_probability = ln_prior + ln_like
        if self.save_fluxes:
            return (ln_probability, self.fluxes)
        else:
            return ln_probability
        
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
        """
        return time of maximum magnification, its value, and corresponding 
        flux for the satellite dataset; takes into account the epochs when 
        satellite data exist

        NOTE: This function is not yet fully tested.
        """
        if self.n_sat > 1:
            raise ValueError("satellite_maximum() doesn't allow " +
                "multiple cpm_sources")
        index = np.argmax(self._sat_magnifications[0])
        magnification = self._sat_magnifications[0][index]
        u_0 = (2*magnification*(magnification**2-1.)**-.5 - 2.)**.5
        trajectory = Trajectory(
            self._sat_times[-1], parameters=self.event.model.parameters,
            parallax=self.event.model._parallax, coords=self.event.coords,
            satellite_skycoord=self.event.datasets[-1].satellite_skycoord)
        if trajectory.y[index] < 0.:
            u_0 = -u_0
        return (self._sat_times[0][index], magnification, u_0)

    def plot_sat_magnitudes(self, **kwargs):
        """Plot satellite model in reference magnitude system"""
        data_ref = self.event.model.data_ref
        (fs, fb) = self.event.model.get_ref_fluxes()
        n = self.n_datasets - self.n_sat

        for i in range(self.n_sat):
            times = self._sat_times[i] - 2450000.
            #(fs_sat, fb_sat) = self.event.model.get_ref_fluxes(n+i)
            #mags = self._sat_magnifications[i]
            #flux = (mags * self._sat_source_flux - fb_sat) * (fs[0] / fs_sat[0]) + fb
            flux = self._sat_magnifications[i] * fs[0] + fb
            plt.plot(times, Utils.get_mag_from_flux(flux), 
                zorder=np.inf, # We want the satellite models to be at the very top. 
                **kwargs)
        self.event.model.data_ref = data_ref

    def standard_plot(self, t_start, t_stop, ylim, title=None,
                      label_list=None, color_list=None, line_width=1.5,
                      legend_order=None, separate_residuals=False):
        """Make plot of the event and residuals. """
        if (label_list is None) != (color_list is None):
            raise ValueError('wrong input in standard_plot')
        if not separate_residuals:
            grid_spec = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.12)
        else:
            grid_spec = gridspec.GridSpec(3, 1, height_ratios=[5, 1, 1], hspace=0.13)
        plt.figure()
        plt.subplot(grid_spec[0])
        if title is not None:
            plt.title(title)
        alphas = [0.35] * self.n_datasets
        for i in range(self.n_sat):
            alphas[-(i+1)] = 1.

        self.event.plot_model(
            color='black', subtract_2450000=True,
            t_start=t_start+2450000., t_stop=t_stop+2450000., label="ground-based model", lw=4)
        self.plot_sat_magnitudes(color='orange', lw=2, label="K2 model") #alpha=0.75,

        if color_list is None:
            color_list_ = ['black'] * (self.n_datasets-self.n_sat) + ['red']*self.n_sat
        else:
            color_list_ = color_list
        zorder_list = np.arange(self.n_datasets, 0, -1)
        zorder_list[1] = self.n_datasets + 1

        self.event.plot_data(#alpha_list=alphas,
            zorder_list=zorder_list, mfc='none', lw=line_width, mew=line_width,
            marker='o', markersize=6, subtract_2450000=True,
            color_list=color_list_, label_list=label_list)
        plt.ylim(ylim[0], ylim[1])
        plt.xlim(t_start, t_stop)

        # ax2 = plt.gca().twiny()
        # ax2.set_ylabel('K2 counts', color='red')
        # ax2.set_yticks([16., 15., 14.])
        # ax2.set_yticklabels(["100", "200", "300"])

        if legend_order is not None:
            if isinstance(legend_order, tuple):
                (handles, labels) = plt.gca().get_legend_handles_labels()
                for (i, l_o) in enumerate(legend_order):
                    handles_ = [handles[idx] for idx in l_o]
                    labels_ = [labels[idx] for idx in l_o]
                    if i == 0:
                        first_legend = plt.legend(handles_, labels_, loc='upper left')
                        plt.gca().add_artist(first_legend)
                    else:
                        plt.legend(handles_, labels_, loc='upper right')
            else:
                (handles, labels) = plt.gca().get_legend_handles_labels()
                handles_ = [handles[idx] for idx in legend_order]
                labels_ = [labels[idx] for idx in legend_order]
                plt.legend(handles_, labels_)
        elif color_list is not None and label_list is not None:
            plt.legend(loc='best')
        else:  # Prepare legend "manually":
            black_line = mlines.Line2D([], [], color='black', marker='o', lw=0,
                          markersize=5, label='ground-based', alpha=alphas[0])
            red_line = mlines.Line2D([], [], color='red', marker='o', lw=0,
                          markersize=5, label='K2C9 data')
            blue_line = mlines.Line2D([], [], color='orange', lw=2, #alpha=0.75,
                          markersize=5, label='K2C9 model')
            plt.legend(handles=[red_line, blue_line, black_line], loc='best')

        plt.subplot(grid_spec[1])
        kwargs_ = dict(mfc='none', lw=line_width, mew=line_width)
        if not separate_residuals:
            self.event.plot_residuals(subtract_2450000=True, **kwargs_)
            plt.xlim(t_start, t_stop)
        else:
            plt.plot([0., 3000000.], [0., 0.], color='black')
            self.event.datasets[-1].plot(
                phot_fmt='mag', show_errorbars=True, subtract_2450000=True,
                model=self.event.model, plot_residuals=True, **kwargs_)
            plt.ylim(0.29, -0.29) # XXX
            plt.ylabel('K2 residuals')
            plt.xlim(t_start, t_stop)

            plt.subplot(grid_spec[2])
            plt.plot([0., 3000000.], [0., 0.], color='black')
            for data in self.event.datasets[:-1]:
                data.plot(
                    phot_fmt='mag', show_errorbars=True, subtract_2450000=True,
                    model=self.event.model, plot_residuals=True, **kwargs_)
            plt.ylabel('Residuals')
            plt.xlim(t_start, t_stop)

    def very_standard_plot(self, t_start, t_stop, ylim, title=None):
        """Make plot of the event and residuals. """
        grid_spec = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.1)
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
        self.plot_sat_magnitudes(color='blue', lw=3.5, alpha=0.75)
        
        self.event.plot_data(alpha_list=alphas, 
            zorder_list=np.arange(self.n_datasets, 0, -1), 
            marker='o', markersize=5, subtract_2450000=True,
            color_list=['black'] * (self.n_datasets-self.n_sat) + ['red']*self.n_sat)
        plt.ylim(ylim[0], ylim[1])
        plt.xlim(t_start, t_stop)
        
        # Prepare legend "manually":
        black_line = mlines.Line2D([], [], color='black', marker='o', lw=0,
                          markersize=5, label='ground-based', alpha=alphas[0])
        red_line = mlines.Line2D([], [], color='red', marker='o', lw=0, 
                          markersize=5, label='K2C9 data')
        blue_line = mlines.Line2D([], [], color='blue', lw=3.5, alpha=0.75,
                          markersize=5, label='K2C9 model')
        plt.legend(handles=[red_line, blue_line, black_line], loc='best')
        
        plt.subplot(grid_spec[1])
        self.event.plot_residuals(subtract_2450000=True)
        plt.xlim(t_start, t_stop)
