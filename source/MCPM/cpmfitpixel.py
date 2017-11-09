import numpy as np

from MCPM import utils
from MCPM import leastSquareSolver as solver


class CpmFitPixel(object):
    """Class for performing CPM fit for a single pixel"""
    
    def __init__(self, target_flux, target_flux_err, target_mask, 
            predictor_matrix, predictor_matrix_mask,
            l2=None, l2_per_pixel=None, 
            model=None, model_mask=None, 
            time=None, train_lim=None, train_mask=None):
                
        self.target_flux = target_flux
        self.target_flux_err = target_flux_err
        self.target_mask = target_mask
        
        self.predictor_matrix = predictor_matrix
        self.predictor_matrix_mask = predictor_matrix_mask
        self.n_epochs = self.predictor_matrix.shape[0]
        self.n_train_pixels = self.predictor_matrix.shape[1]

        self._model = model
        if model_mask is None:
            self.model_mask = np.ones(self.n_epochs, dtype=bool)
        else:    
            self.model_mask = model_mask
        
        self.time = time
        self.train_lim = train_lim
        
        if self.train_lim is not None:
            if self.time is None:
                raise ValueError('time must be set if train_lim is provided')
            self.train_time_mask = (self.time < self.train_lim[0]) \
                & (self.time > self.train_lim[1])
            if sum(self.train_time_mask) == 0:
                msg = 'something went wrong with train_lim: {:} {:} {:}'
                raise ValueError(msg.format(self.train_lim[0], 
                    self.train_lim[1], self.time))
        else:
            self.train_time_mask = np.ones(self.n_epochs, dtype=bool)
        if train_mask is not None:
            self.train_time_mask *= train_mask

        (self.l2, self.l2_per_pixel) = utils.get_l2_l2_per_pixel(
                                        self.n_train_pixels, l2, l2_per_pixel)

        self.reset_results()
    
    def reset_results(self):
        """sets all the results to None so that they're re-calculated 
        when you access them next time"""
        self._target_masked = None
        self._coefs = None
        self._fitted_flux = None
        
    @property
    def model(self):
        """the astrophysical model to be subtracted before the CPM is run"""
        return self._model
    
    @model.setter
    def model(self, value):
        self.reset_results()
        self._model = value
        
    @property
    def results_mask(self):
        """the mask to be applied to all the results"""
        return (self.target_mask * self.predictor_matrix_mask 
                * self.model_mask)
    
    @property
    def train_mask(self):
        """Full mask to be applied for training part of CPM.
        Note that it differs from train_time_mask.
        """
        return self.results_mask * self.train_time_mask
        
    @property
    def target_masked(self):
        """target flux after subtracting model and applying mask"""
        if self._target_masked is None:
            mask = self.train_mask
            self._target_masked = self.target_flux[mask]
            if self.model is not None:
                self._target_masked -= self.model[mask]
        return self._target_masked
        
    @property
    def coefs(self):
        """coefficients inside the CPM - they're multipled by 
        predictor_matrix_masked to get the prediction"""
        if self._coefs is None:
            predictor = self.predictor_matrix[self.train_mask]
            target = self.target_masked
            yvar = None
            
            self._coefs = solver.linear_least_squares(predictor, target, yvar, self.l2)
            
        return self._coefs
        
    @property
    def fitted_flux(self):
        """predicted flux values"""
        if self._fitted_flux is None:
            predictor = self.predictor_matrix[self.results_mask]
            fit = np.dot(predictor, self.coefs)[:,0]
            self._fitted_flux = np.zeros(self.n_epochs, dtype=float)
            self._fitted_flux[self.results_mask] = fit
        return self._fitted_flux
        
    @property
    def residuals(self):
        """residuals of the fit itself i.e., if there was model then 
        it's not added here"""
        out = np.zeros(self.n_epochs, dtype=float)
        mask = self.results_mask
        out[mask] = self.target_flux[mask] - self.fitted_flux[mask]

        if self.model is not None:
            out[mask] -= self.model[mask]
        return out
        
    @property
    def cpm_residuals(self):
        """residuals of the fit with added model"""
        out = np.zeros(self.n_epochs, dtype=float)
        mask = self.results_mask
        out[mask] = self.target_flux[mask] - self.fitted_flux[mask]
        return out
        
