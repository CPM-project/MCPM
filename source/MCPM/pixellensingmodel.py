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

# {'color': 'black', 'subtract_2450000': True, 't_start': 2457500.3, 't_stop': 2457528.0, 'flux_ratio_constraint': None}
    def plot_lc(
            self, times=None, t_range=None, t_start=None, t_stop=None,
            dt=None, n_epochs=None, data_ref=None, f_source=None, f_blend=None,
            subtract_2450000=False, subtract_2460000=False,
            flux_ratio_constraint=None):
        """
        Plot the model lightcurve in magnitudes (???)
        """
        not_implemented = [times, t_range, dt, n_epochs, data_ref,
                           f_source, f_blend, subtract_2460000,
                           flux_ratio_constraint]
        for t in not_implemented:
            if not isinstance(t, None):
                raise NotImplementedError(
                    'not implemented: {:}\n'.format(t))

        subtract = 0.
        if subtract_2450000:
            subtract = 2450000.
# Input to be implemented:
# {'color': 'black', 'subtract_2450000': True, 't_start': 2457500.3, 't_stop': 2457528.0}
        raise NotImplementedError('not finished')
