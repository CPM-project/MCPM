"""
Produces files with (X, Y) coords for 3 events. 
Failed epochs are marked by double "0.0000".
"""
import numpy as np

from MCPM.campaigngridradec2pix import CampaignGridRaDec2Pix


ids = ['ob160795', 'ob160940', 'ob160980']
ras = [271.0010833, 269.5648750, 271.3542917]
decs = [-28.1551111, -27.9635833, -28.0055833]
channels = [52, 31, 52]
campaigns = [91, 92, 92]

for (name, ra, dec, channel, campaign) in zip(ids, ras, decs, channels, campaigns):
    grid = CampaignGridRaDec2Pix(campaign=campaign, channel=channel)
    (x, y) = grid.apply_grids(ra, dec)
    x[~grid.mask] = 0.
    y[~grid.mask] = 0.
    np.savetxt("grid_"+name+".txt", np.array([x, y]).T, fmt="%.4f %.4f")

