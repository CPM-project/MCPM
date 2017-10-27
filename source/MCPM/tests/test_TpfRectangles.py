import numpy as np

from MCPM.tpfrectangles import TpfRectangles
    
    
def test_closest_epics_1():
    channel=52 
    campaign=92
    
    rectangle = TpfRectangles(campaign, channel)
    (out_1, out_2) = rectangle.closest_epics(670, 883)
    
    expect_1 = ['200071074', '200071071', '200071075', '200071072', '200071117', 
                '200071070', '248368847', '200071060', '200071061', '200071116']
    expect_2 = [0.0, 9., 19., 21.024, 46.0, 46.872, 55.444, 59.0, 59.414, 62.0]
    
    assert (out_1[:10] == expect_1).all()
    np.testing.assert_almost_equal(out_2[:10], expect_2, decimal=3)
    assert len(out_1) == 380
    assert len(rectangle.epic) == 381
  
def test_get_epic_id_for_pixel_1():
    channel=52 
    campaign=92  
    
    rectangle = TpfRectangles(campaign, channel)

    assert rectangle.get_epic_id_for_pixel(670, 883) == '200071074'
    
def test_get_epic_id_for_pixel_2():    
    channel = 31
    campaign = 91 
    pix_x = 262
    pix_y = 468
    epic = '200069761'
    
    rectangle = TpfRectangles(campaign, channel)

    assert rectangle.get_epic_id_for_pixel(pix_x, pix_y) == epic
    assert rectangle.get_epic_id_for_pixel(pix_x-1, pix_y) != epic
    assert rectangle.get_epic_id_for_pixel(pix_x, pix_y-1) != epic
    