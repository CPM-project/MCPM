class PixelLensingModelParameters(object):
    """
    Class that emulates MulensModel.ModelParameters for pixel lensing case
    """
    def __init__(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError(
                'input should be dict, not {:}'.format(type(parameters)))
        self.n_sources = 1
        self._check_valid_combination_1_source(parameters.keys())
        self._set_parameters(parameters)

    def _check_valid_combination_1_source(self, keys):
        """
        Check that the user hasn't over-defined the ModelParameters.
        """
        allowed_keys = set(['t_0', 't_E_beta', 'f_s_sat_over_beta'])
        if allowed_keys != keys:
            raise KeyError('something went wrong: {:}'.format(keys))

    def _set_parameters(self, parameters):
        """
        check if parameter values make sense and remember the copy of the dict
        """
        self._check_valid_parameter_values(parameters)
        self.parameters = dict(parameters)

    def _check_valid_parameter_values(self, parameters):
        """
        Prevent user from setting unphysical values
        """
        names = ['t_E_beta', 'f_s_sat_over_beta']
        for name in names:
            if name in parameters.keys():
                if parameters[name] < 0.:
                    raise ValueError("{:} cannot be negative: {:}".format(
                        name, parameters[name]))

    @property
    def t_0(self):
        """
        *float*
        """
        return self.parameters['t_0']

    @t_0.setter
    def t_0(self, new_t_0):
        self.parameters['t_0'] = new_t_0

    @property
    def t_E_beta(self):
        """
        *float*
        """
        return self.parameters['t_E_beta']

    @t_E_beta.setter
    def t_E_beta(self, new_t_E_beta):
        self.parameters['t_E_beta'] = new_t_E_beta

    @property
    def f_s_sat_over_beta(self):
        """
        *float*
        """
        return self.parameters['f_s_sat_over_beta']

    @f_s_sat_over_beta.setter
    def f_s_sat_over_beta(self, new_f_s_sat_over_beta):
        self.parameters['f_s_sat_over_beta'] = new_f_s_sat_over_beta
