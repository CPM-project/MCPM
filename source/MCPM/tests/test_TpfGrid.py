import numpy as np

from MCPM.tpfgrid import TpfGrid


def test_tpf_grid_1():
    g = TpfGrid(92, 31)
    
    (x, y) = g.apply_grid_single(269.013923, -28.227162)
    np.testing.assert_almost_equal(x, 937, decimal=0)
    np.testing.assert_almost_equal(y, 821, decimal=0)
    
    g = TpfGrid(92, 52)
    (x, y) = g.apply_grid_single(271.401045, -27.67326)
    np.testing.assert_almost_equal(x, 687, decimal=0)
    np.testing.assert_almost_equal(y, 871, decimal=0)
    