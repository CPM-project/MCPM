import numpy as np

from MCPM import leastSquareSolver as solver


class CpmFit(object):
    """Class for performing CPM fit"""
    
    def __init__(self, target_flux, target_flux_err, target_mask, 
            predictor_matrix, predictor_mask,
            l2=None, l2_per_pixel=None, 
            model=None, model_mask=None, 
            time=None, train_lim=None):
                
        self.target_flux = target_flux
        self.target_flux_err = target_flux_err
        self.target_mask = target_mask
        
        self.predictor_matrix = predictor_matrix
        self.predictor_mask = predictor_mask
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

        if (l2 is None) == (l2_per_pixel is None):
            raise ValueError('you must set either l2 or l2_per_pixel')
        if l2_per_pixel is not None:
            if not isinstance(l2_per_pixel, (float, np.floating)):
                raise TypeError('l2_per_pixel must be of float type')
            l2 = l2_per_pixel * self.n_train_pixels
        else:
            if not isinstance(l2, (float, np.floating)):
                raise TypeError('l2 must be of float type')
        self.l2 = l2
        self.l2_per_pixel = self.l2 / self.n_train_pixels

        self.reset_results()
    
    def reset_results(self):
        """sets all the results to None so that they're re-calculated 
        when you access them next time"""
        self._coefs = None
        self._fit_flux = None
        self._residue = None
        
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
        return (self.target_mask * self.predictor_mask * self.model_mask)
    
    @property
    def train_mask(self):
        """Full mask to be applied for training part of CPM.
        Note that it differs from train_time_mask.
        """
        return self.results_mask * self.train_time_mask
        
    @property
    def coefs(self):
        """coefficients inside the CPM - they're multipled by 
        predictor_matrix_masked to get the prediction"""
        if self._coefs is None:
            mask = self.train_mask
            predictor = self.predictor_matrix[mask]
            if self.model is not None:
                target = self.target_flux[mask] - self.model[mask]
            else:
                target = self.target_flux[mask]
            yvar = None
            
            self._coefs = solver(predictor, target, yvar, self.l2)
            
        return self._coefs
        
# TO BE DONE:        
#       fit_flux
#       residue
#       cpm_residue - this is residue + model

# In original code, it was:
#     fit_flux = np.dot(predictor_matrix, result)
#     dif = target_flux - fit_flux[:,0]