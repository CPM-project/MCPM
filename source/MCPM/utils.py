import sys
import numpy as np
from math import fsum
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# File with short functions used in different parts of the code.
# Contains:
# - pspl_model()
# - scale_model()
# - limit_time()
# - apply_limit_time()
# - mask_nearest_epochs()
# - pixel_list_center()
# - load_matrix_xy()
# - save_matrix_xy()
# - read_true_false_file()
# - degree_to_n_coeffs()
# - n_coeffs_to_degree()
# - eval_poly_2d_no_coeffs()
# - eval_poly_2d_coeffs()
# - eval_poly_2d()
# - fit_two_poly_2d()
# - plot_matrix_subplots()
# - construct_matrix_from_list()
# - module_output_for_channel

def pspl_model(t_0, u_0, t_E, f_s, times):
    """Paczynski model for given times"""
    tau = (times - t_0) / t_E
    u_2 = tau**2 + u_0**2
    model = (u_2 + 2.) / np.sqrt(u_2 * (u_2 + 4.))
    model *= f_s
    return model

def scale_model(t_0, width_ratio, flux, times, model_time, model_value):
    """
    scale and interpolate (model_time, model_value); good for a single eclipse
    """
    time = model_time * width_ratio + t_0
    value = (model_value - 1.) #* depth_ratio 
    #value += 1. # This is commented because we want baseline subtracted.
    value *= flux
    out = np.ones_like(times, dtype=np.float)
    mask_min = (times < np.min(time))
    mask_max = (times > np.max(time))
    mask = (~np.isnan(times) & ~mask_min & ~mask_max)
    out[mask] = interp1d(time, value, kind='cubic')(times[mask])
    out[mask_min] = value[0]
    out[mask_max] = value[-1]
    return out

def limit_time(time, epoch_begin=None, epoch_end=None):
    """
    Make the mask for a time vector.  Limit to epochs later than epoch_begin. 
    Limit it also to epochs earlier than epoch_end. These 2 limits are 
    combined using AND or OR in the case epoch_begin<epoch_end and 
    epoch_begin>epoch_end, respectively, i.e., it's inteligent and do what you
    expect.
    NAN epochs are masked out.
    """
    if epoch_begin is not None and epoch_begin == epoch_end:
        raise ValueError('illegal input in limit_time()')

    mask_nan = np.isnan(time)
    mask_not_nan = np.logical_not(mask_nan)
    
    if epoch_begin is None:
        mask_begin = np.ones_like(time, dtype=bool)
        mask_begin[mask_nan] = False
    else:
        mask_begin = np.zeros_like(time, dtype=bool)
        mask_begin[mask_not_nan] = (time[mask_not_nan] > epoch_begin)

    if epoch_end is None:
        mask_end = np.ones_like(time, dtype=bool)
        mask_end[mask_nan] = False
    else:
        mask_end = np.zeros_like(time, dtype=bool)
        mask_end[mask_not_nan] = (time[mask_not_nan] < epoch_end)

    if epoch_begin is None or epoch_end is None:
        out = mask_begin & mask_end
    elif epoch_begin < epoch_end:
        out = mask_begin & mask_end
    else:
        out = mask_begin | mask_end
       
    if np.sum(out) == 0:
        raise ValueError('time limits resulted in 0 epochs accepted\n' +
            '{:}\n{:}\n{:}\n'.format(time, epoch_begin, epoch_end))

    return out

def apply_limit_time(cpm_source, options):
    """
    For given CpmFitSource and dictionary options check if there are 
    keys 'train_mask_begin' or 'train_mask_end' and if so, 
    then run limit_time() and set_train_mask().
    """
    if 'train_mask_begin' not in options:
        train_mask_begin = None
    else:
        train_mask_begin = options['train_mask_begin']
    if 'train_mask_end' not in options:
        train_mask_end = None
    else:
        train_mask_end = options['train_mask_end']

    if train_mask_begin is not None or train_mask_end is not None:
        mask = limit_time(cpm_source.pixel_time + 2450000., 
            train_mask_begin, train_mask_end)
        cpm_source.set_train_mask(mask)

def mask_nearest_epochs(time, epochs, max_dt=0.005):
    """
    Take elements of epochs (list or array) and for each of them find closest 
    element in vector time. Then mask out the element, if it within max_dt
    of the searched value (do nothing otherwise). Mask out also nan epochs. 
    Return value is a mask.
    """
    time_nan = np.isnan(time)
    time_not_nan = np.logical_not(time_nan)
    mask_cumsum_1 = np.cumsum(time_not_nan) - 1
    masked_time = time[time_not_nan]
    #indexes = []
    mask = np.ones_like(time, dtype=bool)
    for epoch in epochs:
        delta_time = np.abs(masked_time-epoch)
        index = delta_time.argmin()
        if delta_time[index] < max_dt:
            mask[mask_cumsum_1 == index] = False
            #indexes.append(index)
    #for index in indexes
    mask[time_nan] = False
    
    return mask

def pixel_list_center(center_x, center_y, half_size):
    """
    Return list of pixels centered on (center_x,center_y) 
    [rounded to nearest integer] and covering 
    n = 2*half_size+1 pixels on the side. The output shape is (n^2, 2)
    """
    int_x = int(center_x + 0.5)
    int_y = int(center_y + 0.5)
    return np.mgrid[(int_x-half_size):(int_x+half_size+1), 
                    (int_y-half_size):(int_y+half_size+1)].reshape(2, -1).T

def load_matrix_xy(file_name, data_type='float'):
    """reads file with matrix in format like:
    0 0 123.454
    0 1 432.424
    ...
    into numpy array"""
    parser = {'TRUE': True, 'FALSE': False}
    table_as_list = []
    with open(file_name) as infile:
        for line in infile.readlines():
            if line[0] == '#':
                continue
            data = line.split()
            if len(data) != 3:
                raise ValueError("incorrect line read from file {:} : {:}".format(file_name, line[:-1]))
            x = int(data[0])
            y = int(data[1])
            
            if data_type == 'float':
                value = float(data[2])
            elif data_type == 'boolean':
                value = parser[data[2].upper()]
            else:
                raise ValueError('Unknown data_type in load_matrix_xy()')

            if len(table_as_list) < x + 1:
                for i in range(x+1-len(table_as_list)):
                    table_as_list.append([])
            if len(table_as_list[x]) < y + 1:
                for i in range(y+1-len(table_as_list[x])):
                    table_as_list[x].append(None)
            
            table_as_list[x][y] = value

    return np.array(table_as_list)
    
def save_matrix_xy(matrix, file_name, data_type='float'):
    """saves numpy array (matrix) in format like:
    0 0 123.454
    0 1 432.424
    ...
    """
    with open(file_name, 'w') as out_file:
        if data_type == 'float':
            for (index, value) in np.ndenumerate(matrix):
                out_file.write("{:} {:} {:.8f}\n".format(index[0], index[1], value))
        elif data_type == 'boolean':
            parser = {1: "True", 0: "False"}
            for (index, value) in np.ndenumerate(matrix):
                out_file.write("{:} {:} {:}\n".format(index[0], index[1], parser[value]))
        else:
            raise ValueError('save_matrix_xy() - unrecognized format')
        
def read_true_false_file(file_name):
    """Reads file with values True or False into a boolean numpy array.
    To save such a file, just use:
    np.savetxt(FILE_NAME, BOOL_ARRAY, fmt='%r')
    """
    parser = {'TRUE': True, 'FALSE': False}
    out = []
    with open(file_name) as in_file:
        for line in in_file.readlines():
            out.append(parser[line[:-1].upper()])
    return np.array(out)

def degree_to_n_coeffs(degree):
    """how many coefficients has a 2d polynomial of given degree"""
    return int((degree+1)*(degree+2)/2.+0.5)

def n_coeffs_to_degree(n_coeffs):
    """what is degree if 2d polynomial has n_coeffs coefficients"""
    delta_sqrt = int((8 * n_coeffs + 1.)**.5 + 0.5)
    if delta_sqrt**2 != (8*n_coeffs+1.):
        raise ValueError('Wrong input in n_coeffs_to_degree(): {:}'.format(n_coeffs))
    return int((delta_sqrt - 3.) / 2. + 0.5)

def eval_poly_2d_no_coeffs(x, y, deg):
    """evaluate powers of given values and return as a table: [1, x, y, x^2, xy, y^2] for deg = 2"""
    pow_x = np.polynomial.polynomial.polyvander(x, deg)
    pow_y = np.polynomial.polynomial.polyvander(y, deg)
    results = []
    for i in range(deg+1):
        for j in range(i+1):
            results.append(pow_x[:,i-j]*pow_y[:,j])
    return np.array(results)

def eval_poly_2d_coeffs(x, y, coeffs):
    """evaluate 2d polynomial without summing up"""
    c = np.copy(coeffs).reshape(coeffs.size, 1)
    deg = n_coeffs_to_degree(len(c))
    return c * eval_poly_2d_no_coeffs(x=x, y=y, deg=deg)

def eval_poly_2d(x, y, coeffs):
    """evaluate 2d polynomial"""
    monomials = eval_poly_2d_coeffs(x=x, y=y, coeffs=coeffs)
    results = []
    for i in range(monomials.shape[1]):
        results.append(fsum(monomials[:,i]))
    return np.array(results)

def fit_two_poly_2d(x_in, y_in, x_out, y_out, degree):
    """fits 2 polynomials: x_out = f(x_in, y_in, coeffs) and same for y_out"""
    basis = eval_poly_2d_no_coeffs(x_in, y_in, degree).T
    (coeffs_x, residuals_x, rank_x, singular_x) = np.linalg.lstsq(basis, x_out)
    (coeffs_y, residuals_y, rank_y, singular_y) = np.linalg.lstsq(basis, y_out)
    return (coeffs_x, coeffs_y)

def plot_matrix_subplots(figure, time, matrix, same_y_axis=True, 
                        data_mask=None, y_lim=None, **kwargs):
    """
    Plot given 3D matrix in subpanels. Note that 3rd component of matrix.shape 
    must be the same as time.size i.e., matrix.shape[2]==time.size
    
    **kwargs are passed Axis.plot()
    
    example usage:
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.set_size_inches(50,30)
    matrix = np.random.random(size=540).reshape((3, 3, -1))
    plot_matrix(fig, time, matrix)
    plt.savefig("file_name.png")
    plt.close()
    """
    x_lim_expand = 0.06
    y_lim_expand = 0.08

    if y_lim is None:
        if data_mask is not None:
            y_lim = [np.nanmin(matrix[:,:,data_mask]), np.nanmax(matrix[:,:,data_mask])]
        else:    
            y_lim = [np.nanmin(matrix), np.nanmax(matrix)]
        d_y_lim = y_lim[1] - y_lim[0]  
        y_lim[0] -= d_y_lim * y_lim_expand / 2.
        y_lim[1] += d_y_lim * y_lim_expand / 2.

    (i_max, j_max, _) = matrix.shape
    panels = np.flipud(np.arange(i_max*j_max).reshape(i_max, j_max)) + 1
    if data_mask is not None:
        time = time[data_mask]
    d_time = (max(time) - min(time)) * x_lim_expand / 2.
    x_lim = [min(time)-d_time, max(time)+d_time]    

    for i in range(i_max):
        for j in range(j_max):
            ax = plt.subplot(i_max, j_max, panels[i, j])
            y_axis = matrix[i][j]
            if data_mask is not None:
                y_axis = y_axis[data_mask]
                
            ax.plot(time, y_axis, '.k', **kwargs)
            
            if i != 0:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)
            if same_y_axis:
                ax.set_ylim(y_lim)
            ax.set_xlim(x_lim)
    figure.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)

def construct_matrix_from_list(pixel_list, time_series_list):
    """
    take a list of pixels (pixel_list) and a list of corresponding 
    flux values (time_series_list) and make matrix that can be given to 
    plot_matrix_subplots
    """
    (x_0, y_0) = np.min(pixel_list, axis=0)
    (x_range, y_range) = np.max(pixel_list, axis=0) - np.array([x_0, y_0]) + 1

    value_length = len(time_series_list[0])
    for values in time_series_list:
        if len(values) != value_length:
            raise ValueError('construct_matrix_from_list() - all ' +
                    'time series vectors have to be of the same ' +
                    'length')

    matrix = np.zeros((x_range, y_range, value_length))

    for (i, (x, y)) in enumerate(pixel_list):
        matrix[x-x_0, y-y_0, :] = time_series_list[i]

    return matrix

def get_l2_l2_per_pixel(n_pixel, l2=None, l2_per_pixel=None):
    """function used in different places that parses the l2 and l2_per_pixel
    parameters - exactly one of them has to be set"""
    if (l2 is None) == (l2_per_pixel is None):
        raise ValueError('you must set either l2 or l2_per_pixel')

    if l2_per_pixel is not None:
        if not isinstance(l2_per_pixel, (float, np.floating)):
            if isinstance(l2_per_pixel, (int, np.integer)):
                l2_per_pixel = float(l2_per_pixel)
            else:
                raise TypeError('l2_per_pixel must be of float or int type')
        l2 = l2_per_pixel * n_pixel
    else:
        if not isinstance(l2, (float, np.floating)):
            if isinstance(l2, (int, np.integer)):
                l2 = float(l2)
            else:
                raise TypeError('l2 must be of float type')
    return (l2, l2 / float(n_pixel))

# For K2 channel number give corresponding module and output numbers.
module_output_for_channel = {
1: (2, 1), 2: (2, 2), 3: (2, 3), 4: (2, 4), 5: (3, 1), 
6: (3, 2), 7: (3, 3), 8: (3, 4), 9: (4, 1), 10: (4, 2), 
11: (4, 3), 12: (4, 4), 13: (6, 1), 14: (6, 2), 15: (6, 3), 
16: (6, 4), 17: (7, 1), 18: (7, 2), 19: (7, 3), 20: (7, 4), 
21: (8, 1), 22: (8, 2), 23: (8, 3), 24: (8, 4), 25: (9, 1), 
26: (9, 2), 27: (9, 3), 28: (9, 4), 29: (10, 1), 30: (10, 2), 
31: (10, 3), 32: (10, 4), 33: (11, 1), 34: (11, 2), 35: (11, 3), 
36: (11, 4), 37: (12, 1), 38: (12, 2), 39: (12, 3), 40: (12, 4), 
41: (13, 1), 42: (13, 2), 43: (13, 3), 44: (13, 4), 45: (14, 1), 
46: (14, 2), 47: (14, 3), 48: (14, 4), 49: (15, 1), 50: (15, 2), 
51: (15, 3), 52: (15, 4), 53: (16, 1), 54: (16, 2), 55: (16, 3), 
56: (16, 4), 57: (17, 1), 58: (17, 2), 59: (17, 3), 60: (17, 4), 
61: (18, 1), 62: (18, 2), 63: (18, 3), 64: (18, 4), 65: (19, 1), 
66: (19, 2), 67: (19, 3), 68: (19, 4), 69: (20, 1), 70: (20, 2), 
71: (20, 3), 72: (20, 4), 73: (22, 1), 74: (22, 2), 75: (22, 3), 
76: (22, 4), 77: (23, 1), 78: (23, 2), 79: (23, 3), 80: (23, 4), 
81: (24, 1), 82: (24, 2), 83: (24, 3), 84: (24, 4)
}
