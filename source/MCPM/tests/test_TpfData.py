import numpy as np
from os import path

import MCPM
from MCPM import tpfdata


def test_1():
    """.reference_pixel and .rows"""
    campaign = 92
    channel = 31
    pix_y = 670
    pix_x = 883
    epic_id = 200071074

    tpf_data = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)

    assert (tpf_data.reference_pixel == np.array([662, 838])).all()
    assert set(tpf_data.rows) == set(np.arange(838, 902))
    assert tpf_data.rows.shape == (3200,)

def test_check_pixel_in_tpf():
    """testing check_pixel_in_tpf() method of TpfData"""
    campaign = 92
    channel = 31
    pix_y = 670
    pix_x = 883
    epic_id = 200071074

    tpf_data = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)

    assert tpf_data.check_pixel_in_tpf(column=pix_y, row=pix_x) is True
    assert tpf_data.check_pixel_in_tpf(column=pix_x, row=pix_y) is False
