from os import path
import glob
import numpy as np
from scipy.interpolate import RectBivariateSpline
from math import fabs

from astropy.io import fits

import MCPM
from MCPM.utils import module_output_for_channel


class PrfData(object):
    """
    K2 PRF data 
    """

    data_directory = path.join(MCPM.MODULE_PATH, 'data', 'Kepler_PRF') 

    def __init__(self, channel=None, module=None, output=None):
        """
        provide channel or both module and output
        data_directory has to be set
        """
        if (module is None) != (output is None):
            raise ValueError('You must set both module and output options')
        if (channel is None) == (module is None):
            raise ValueError('provide channel or both module and output')

        if channel is not None:
            (module, output) = module_output_for_channel[channel]
        text = "kplr{:02d}.{:}_*_prf.fits".format(int(module), output)
        names = path.join(self.data_directory, text)

        try:
            file_name = glob.glob(names)[-1]
        except:
            www = 'http://archive.stsci.edu/missions/kepler/fpc/prf/'
            raise FileNotFoundError(('PRF files {:} not found. The file ' +
                                'should be downloaded from {:} to {:}'
                                ).format(names, www, self.data_directory))
        
        keys = ['CRPIX1P', 'CRVAL1P', 'CDELT1P', 
                'CRPIX2P', 'CRVAL2P', 'CDELT2P']
        with fits.open(file_name) as prf_hdus:
            self._data = []
            self._keywords = []
            for hdu in prf_hdus[1:]:
                self._data.append(hdu.data)
                keywords = dict()
                for key in keys:
                    keywords[key] = hdu.header[key]
                self._keywords.append(keywords)

        # make sure last hdu is for central area
        center_x = np.array([value['CRVAL1P'] for value in self._keywords])
        center_y = np.array([value['CRVAL2P'] for value in self._keywords])
        dx = center_x - np.mean(center_x)
        dy = center_y - np.mean(center_y)
        if np.argmin(np.sqrt(dx**2+dy**2)) != len(center_x)-1:
            raise ValueError('The last hdu in PRF file is not the one in ' + 
                            'the center - contarary to what we assumed here!')
        
        # make a list of pairs but exclude the central point
        n = len(center_x)
        self._corners_pairs = [(i, i+1) if i>=0 else (i+n-1, i+1) for i in 
                                range(-1, n-2)]

        # make sure that the first four corners are in clockwise, or 
        # anti-clockwise order:
        for (i, j) in self._corners_pairs:
            # We want one coordinate to be equal and other to be different.
            if (fabs(center_x[i] - center_x[j]) < .001 != 
                    fabs(center_y[i] - center_y[j]) < .001): 
                msg = 'something wrong with order of centers of hdus'
                raise ValueError(msg)

        # prepare equations to be used for barycentric interpolation
        self._equations = dict()
        for (i, j) in self._corners_pairs:
            xs = [center_x[i], center_x[j], center_x[-1]]
            ys = [center_y[i], center_y[j], center_y[-1]]
            self._equations[(i, j)] = np.array([xs, ys, [1., 1., 1.]])

        # grid on which prf is defined:
        x_lim = self._keywords[0]['CRPIX1P'] - .5
        y_lim = self._keywords[0]['CRPIX2P'] - .5
        self._prf_grid_x = np.linspace(-x_lim, x_lim, num=int(2*x_lim+1+.5))
        self._prf_grid_y = np.linspace(-y_lim, y_lim, num=int(2*y_lim+1+.5))
        self._prf_grid_x *= self._keywords[0]['CDELT1P']
        self._prf_grid_y *= self._keywords[0]['CDELT2P']
        #self._prf_grid_x = (np.arange(nx) - nx / 2. + .5) * self._keywords[0]['CDELT1P']
        #self._prf_grid_y = (np.arange(ny) - ny / 2. + .5) * self._keywords[0]['CDELT2P']
        
        self.center_x = center_x
        self.center_y = center_y

        # For interpolation lazy loading:
        self._spline_function = None
        self._fast_x = None
        self._fast_y = None

    def _get_barycentric_interpolation_weights(self, x, y):
        """find in which triangle given point is located and 
        calculate weights for barycentric interpolation"""
        for (i, j) in self._corners_pairs:
            equation = self._equations[(i, j)]
            weights = np.linalg.solve(equation, np.array([x, y, 1.]))
            if np.all(weights >= 0.): # i.e. we found triangle in which 
                return (np.array([i, j, -1]), weights) # the point is located
        raise ValueError("Point ({:}, {:}) doesn't lie in any of the triangles".format(x, y))

    def _interpolate_prf(self, x, y):
        """barycentric interpolation on a traiangle grid"""
        (indexes, weights) = self._get_barycentric_interpolation_weights(x=x, 
                                                                         y=y)
        prf = (self._data[indexes[0]] * weights[0] 
                    + self._data[indexes[1]] * weights[1] 
                    + self._data[indexes[2]] * weights[2])
        return prf

    def get_interpolated_prf(self, star_x, star_y, pixels_list, fast=True):
        """
        For star centered at given position calculate PRF for list of pixels.
        Example:    star_x=100.5, 
                    star_y=200.5, 
                    pixels_list=[[100., 200.], [101., 200.], [102., 200.]]
        The fast option controls if we're doing full interpolation (False), 
        or use results from some previous run. The full interpolation is done 
        if the current pixel is further than 3 pix from the remembered run.
        """
        max_distance = 3.
        
        if self._fast_x is None:
            distance2 = 2. * max_distance**2
        else:
            distance2 = (self._fast_x-star_x)**2+(self._fast_y-star_y)**2
        
        if (self._spline_function is None 
                                or not fast or distance2 > max_distance**2):
            prf = self._interpolate_prf(star_x, star_y)
            self._spline_function = RectBivariateSpline(x=self._prf_grid_x,
                                                    y=self._prf_grid_y, z=prf)
            self._fast_x = star_x
            self._fast_y = star_y
                    
        out = np.array([self._spline_function(y-star_y, x-star_x)[0][0] 
                                                for (x, y) in pixels_list])
        # Yes, here we revert the order of x,y because of K2 PRF data format.

        out[(out < 0.)] = 0.
        return out
