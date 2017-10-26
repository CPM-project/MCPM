import numpy as np

from MCPM.tpfrectangles import TpfRectangles
    
    
def test_closest_epics_1():
    channel=52 
    campaign=92
    
    rectangle = TpfRectangles(campaign, channel)
    (out_1, out_2) = rectangle.closest_epics(670, 883)
    
    expect_1 = ['200071074', '200071071', '200071075', '200071072', '200071117', 
                '200071070', '248368847', '200071060', '200071061', '200071116']
    expect_2 = [0.0, 8.0, 20.0, 21.541, 45.0, 45.706, 56.851, 58.0, 58.549, 61.0]
    
    assert (out_1[:10] == expect_1).all()
    np.testing.assert_almost_equal(out_2[:10], expect_2, decimal=3)
    assert len(out_1) == 380
    assert len(rectangle.epic) == 381
    