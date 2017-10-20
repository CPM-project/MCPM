import numpy as np

import MCPM.utils as utils


def test_pixel_list_center():
    expected_1 = np.array([[ 99, 199], [ 99, 200], [ 99, 201], [100, 199], 
        [100, 200], [100, 201], [101, 199], [101, 200], [101, 201]])
    expected_2 = np.array([[399, 299], [399, 300], [399, 301], [399, 302], 
        [399, 303], [400, 299], [400, 300], [400, 301], [400, 302], 
        [400, 303], [401, 299], [401, 300], [401, 301], [401, 302], 
        [401, 303], [402, 299], [402, 300], [402, 301], [402, 302], 
        [402, 303], [403, 299], [403, 300], [403, 301], [403, 302], 
        [403, 303]])
        
    out_1 = utils.pixel_list_center(100.1, 200.1, 1)
    out_2 = utils.pixel_list_center(400.7, 300.9, 2)
    
    assert (expected_1 == out_1).all()
    assert (expected_2 == out_2).all()
    
def test_eval_poly_2d():
    x = np.array([1, 2, 0, 1, 1])
    y = np.array([3, 4, 1, 0, 10])
    out = utils.eval_poly_2d(x, y, np.array([5, 6, 7]))
    np.testing.assert_almost_equal(out, [ 32., 45., 12., 11., 81.])
