import numpy as np

import poly2d


class GridRaDec2Pix(object):
    """a single transformation from (RA, Dec) to (x,y)"""

    def __init__(self, coefs_x, coefs_y):
        """coefs_x and coefs_y has to be of the same type: np.array, list, string"""
        if type(coefs_x) != type(coefs_y):
            raise TypeError('different types of input data')

        if isinstance(coefs_x, np.ndarray):
            coefs_x_in = coefs_x
            coefs_y_in = coefs_y
        elif isinstance(coefs_x, list):
            if isinstance(coefs_x[0], str):
                coefs_x_in = np.array([float(value) for value in coefs_x])
                coefs_y_in = np.array([float(value) for value in coefs_y])
            else:
                raise TypeError('unrecognized type in input list: {:}'.format(type(coefs_x[0])))
        else:
            raise TypeError('unrecognized input type: {:}'.format(type(coefs_x)))

        self.coefs_x = coefs_x_in
        self.coefs_y = coefs_y_in

    def apply_grid(self, ra, dec):
        """calculate pixel coordinates for given (RA,Dec) which can be floats, lists, or numpy.arrays"""
        x_out = poly2d.eval_poly_2d(ra, dec, self.coefs_x)
        y_out = poly2d.eval_poly_2d(ra, dec, self.coefs_y)
        return (x_out, y_out)
    
    def apply_grid_single(self, ra, dec):
        """calculate pixel coordinates for a single sky position (RA,Dec)"""
        if not isinstance(ra, (float, np.floating)) 
                    or not isinstance(dec, (float, np.floating)):
            raise TypeError('2 floats expected, got {:} and {:}'.format(
                                                type(ra), type(dec)))
        out = self.apply_grid(ra=ra, dec=dec)
        return (out[0][0], out[1][0])
        