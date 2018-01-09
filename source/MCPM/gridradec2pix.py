import numpy as np

from MCPM import utils


class GridRaDec2Pix(object):
    """a single transformation from (RA, Dec) to (x,y)"""

    def __init__(self, coeffs_x, coeffs_y):
        """coeffs_x and coeffs_y has to be of the same type: np.array, list, string"""
        if type(coeffs_x) != type(coeffs_y):
            raise TypeError('different types of input data')

        if isinstance(coeffs_x, np.ndarray):
            coeffs_x_in = coeffs_x
            coeffs_y_in = coeffs_y
        elif isinstance(coeffs_x, list):
            if isinstance(coeffs_x[0], str):
                coeffs_x_in = np.array([float(value) for value in coeffs_x])
                coeffs_y_in = np.array([float(value) for value in coeffs_y])
            else:
                raise TypeError('unrecognized type in input list: {:}'.format(type(coeffs_x[0])))
        else:
            raise TypeError('unrecognized input type: {:}'.format(type(coeffs_x)))

        self.coeffs_x = coeffs_x_in
        self.coeffs_y = coeffs_y_in

    def apply_grid(self, ra, dec):
        """calculate pixel coordinates for given (RA,Dec) which can be floats, lists, or numpy.arrays"""
        x_out = utils.eval_poly_2d(ra, dec, self.coeffs_x)
        y_out = utils.eval_poly_2d(ra, dec, self.coeffs_y)
        return (x_out, y_out)
    
    def apply_grid_single(self, ra, dec):
        """calculate pixel coordinates for a single sky position (RA,Dec)"""
        if (not isinstance(ra, (float, np.floating)) 
                    or not isinstance(dec, (float, np.floating))):
            raise TypeError('2 floats expected, got {:} and {:}'.format(
                                                type(ra), type(dec)))
        out = self.apply_grid(ra=ra, dec=dec)
        return (out[0][0], out[1][0])
        