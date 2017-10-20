import sys
import numpy as np
from math import fsum
import matplotlib.pyplot as plt



# File with short functions used in different parts of the code.
# Contains:
# - pixel_list_center()
# - load_matrix_xy()
# - save_matrix_xy()
# - read_true_false_file()
# - degree_to_n_coefs()
# - n_coefs_to_degree()
# - eval_poly_2d_no_coefs()
# - eval_poly_2d_coefs()
# - eval_poly_2d()
# - fit_two_poly_2d()
# - plot_matrix_subplots()
# - construct_matrix_from_list()

def pixel_list_center(center_x, center_y, half_size):
    """Return list of pixels centered on (center_x,center_y) 
    [rounded to nearest integer] and covering 
    n = 2*half_size+1 pixels on the side. The output shape is (n^2, 2)"""
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

def degree_to_n_coefs(degree):
    """how many coefficients has a 2d polynomial of given degree"""
    return int((degree+1)*(degree+2)/2.+0.5)

def n_coefs_to_degree(n_coefs):
    """what is degree if 2d polynomial has n_coefs coeficients"""
    delta_sqrt = int((8 * n_coefs + 1.)**.5 + 0.5)
    if delta_sqrt**2 != (8*n_coefs+1.):
        raise ValueError('Wrong input in n_coefs_to_degree(): {:}'.format(n_coefs))
    return int((delta_sqrt - 3.) / 2. + 0.5)

def eval_poly_2d_no_coefs(x, y, deg):
    """evaluate powers of given values and return as a table: [1, x, y, x^2, xy, y^2] for deg = 2"""
    pow_x = np.polynomial.polynomial.polyvander(x, deg)
    pow_y = np.polynomial.polynomial.polyvander(y, deg)
    results = []
    for i in range(deg+1):
        for j in range(i+1):
            results.append(pow_x[:,i-j]*pow_y[:,j])
    return np.array(results)

def eval_poly_2d_coefs(x, y, coefs):
    """evaluate 2d polynomial without summing up"""
    c = np.copy(coefs).reshape(coefs.size, 1)
    deg = n_coefs_to_degree(len(c))
    return c * eval_poly_2d_no_coefs(x=x, y=y, deg=deg)

def eval_poly_2d(x, y, coefs):
    """evaluate 2d polynomial"""
    monomials = eval_poly_2d_coefs(x=x, y=y, coefs=coefs)
    results = []
    for i in range(monomials.shape[1]):
        results.append(fsum(monomials[:,i]))
    return np.array(results)

def fit_two_poly_2d(x_in, y_in, x_out, y_out, degree):
    """fits 2 polynomials: x_out = f(x_in, y_in, coefs) and same for y_out"""
    basis = eval_poly_2d_no_coefs(x_in, y_in, degree).T
    (coeffs_x, residuals_x, rank_x, singular_x) = np.linalg.lstsq(basis, x_out)
    (coeffs_y, residuals_y, rank_y, singular_y) = np.linalg.lstsq(basis, y_out)
    return (coeffs_x, coeffs_y)

def plot_matrix_subplots(figure, time, matrix, same_y_axis=True, 
                        data_mask=None, **kwargs):
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
                ax.get_xaxis().set_visible(False)
            if j != 0:
                ax.get_yaxis().set_visible(False)
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
