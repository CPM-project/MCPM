import sys
import numpy as np

from astropy.io import fits
from astropy.table import Table


def select_predictor_pixels(file_in, n_select, file_out):
    """
    Read file with posterior coeffs, select n_select pixels with most
    information and save to fits file. The information is sum of 
    mean_PRF * |coeff|.
    """
    with fits.open(file_in) as hdul:
        sum_prf_coeffs_abs = 0.
        for i in range(len(hdul[3].data[0])):
            sum_prf_coeffs_abs += hdul[4].data[i][0] * np.abs(hdul[3].data.field(i))
        indexes = np.argsort(sum_prf_coeffs_abs)[::-1][:n_select]
        sums = sum_prf_coeffs_abs[indexes]
        pixels = hdul[2].data[indexes]

        keys_check = ['RA', 'Dec', 'CAMPAIGN', 'CHANNEL', 'code', 'n_pixel',
                      'min_distance', 'exclude', 'median_flux_ratio_limits',
                      'median_flux_limits']
        header = fits.Header()
        for key in keys_check:
            if key in hdul[0].header:
                header[key] = hdul[0].header[key]

    header['input'] = file_in
    header['comment'] = 'Indexes refer to input data.'
    header['comment'] = 'Pixels are selected based on sum of PRF values'
    header['comment'] = 'multiplied by absolute value of coeffs.'
    hdu_0 = fits.PrimaryHDU(header=header)

    column_1 = fits.Column(name='row', array=[r for (r, c) in pixels], format='I')
    column_2 = fits.Column(name='column', array=[c for (r, c) in pixels], format='I')
    column_3 = fits.Column(name='sum_prf_abs_coeff', array=sums, format='E')
    column_4 = fits.Column(name='index', array=indexes, format='I')
    columns = [column_1, column_2, column_3, column_4]
    hdu_1 = fits.BinTableHDU.from_columns(columns, name='selected_pixels')

    hdus = fits.HDUList([hdu_0, hdu_1])
    hdus.writeto(file_out)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError('3 parameters required')
    n_select = int(sys.argv[2])

    select_predictor_pixels(sys.argv[1], n_select, sys.argv[3])
