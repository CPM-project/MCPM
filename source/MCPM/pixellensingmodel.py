import numpy as np

from MulensModel.coordinates import Coordinates

from MCPM.pixellensingmodelparameters import PixelLensingModelParameters


class PixelLensingModel(object):
    """
    Class that emulates MulensModel.Model for pixel lensing case
    """
    def __init__(self, parameters=None, coords=None):
        self._parameters = PixelLensingModelParameters(parameters)
        self._coords = None
        if coords is not None:
            self._coords = Coordinates(coords)

        self._datasets = []

    @property
    def n_sources(self):
        """
        *int*

        number of luminous sources
        """
        return self._parameters.n_sources

    def flux_difference(self, times):
        """
        Flux excess over baseline
        (Gould 1996, ApJ 470, 201) eq 2.4 and 2.5
        """
        t_0 = self._parameters.parameters['t_0']
        t_E_beta = self._parameters.parameters['t_E_beta']
        f_s_sat_over_beta = self._parameters.parameters['f_s_sat_over_beta']
        
        g = 1./np.sqrt(((times - t_0) / t_E_beta)**2 + 1)
        return f_s_sat_over_beta * g

    def set_datasets(self, datasets, data_ref=0):
        """
        Set :obj:`datasets` property
        """
        if isinstance(datasets, list):
            self._datasets = datasets
        else:
            self._datasets = [datasets]
        self.data_ref = data_ref

    @property
    def parameters(self):
        """
        PixelLensingModelParameters instance
        """
        return self._parameters

