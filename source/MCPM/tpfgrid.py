import numpy as np

from MCPM.gridradec2pix import GridRaDec2Pix


poly_x = {}
poly_y = {}
# The coefs below come from fitting 2D polynomial of second order to WCS info 
# in TPF files. 
poly_x[(91, 30)] = np.array([277642.8375, -1113.86318265, 1787.9985768, 
    0.236717733144, -7.42780150635, -0.157826637598])
poly_y[(91, 30)] = np.array([150914.256422, -1517.34529435, -1290.30903425, 
    3.22424558711, 1.41382265331, -0.506503527612])
poly_x[(91, 31)] = np.array([236887.541626, -809.564888175, 1805.0769435, 
    -0.330080664114, -7.4689686502, -0.0525417429192])
poly_y[(91, 31)] = np.array([-162471.016825, 1608.81421231, 1190.4679193, 
    -3.38498268453, -1.24142655525, -0.450635425221])
poly_x[(91, 32)] = np.array([-392151.859325, 2002.36033911, -1604.72136458, 
    -1.93098370565, 6.66219790546, -0.237106688973])
poly_y[(91, 32)] = np.array([-154350.220411, 1560.44865712, 1303.44412622, 
    -3.31959577579, -1.70235068541, -0.642428571824])
poly_x[(91, 49)] = np.array([-347066.168294, 1634.61641873, -1690.71819605, 
    -1.22141419562, 7.00376709891, -0.132643513526])
poly_y[(91, 49)] = np.array([129483.075187, -1377.44168486, -1480.94041078, 
    3.00241976273, 2.16813184999, -0.263362558986])
poly_x[(91, 52)] = np.array([-342773.611853, 1608.20265637, -1637.89491469, 
    -1.18392614459, 6.78021149258, -0.266563795126])
poly_y[(91, 52)] = np.array([-143844.789196, 1488.85218907, 1379.99645939, 
    -3.19769595122, -1.97663739663, -0.626481128611])
poly_x[(92, 30)] = np.array([277666.963879, -1114.00444471, 1788.16800812, 
    0.236920423264, -7.42822379334, -0.157274870698])
poly_y[(92, 30)] = np.array([150899.819819, -1517.23803198, -1290.47477526, 
    3.22405123849, 1.41473053931, -0.504657747521])
poly_x[(92, 31)] = np.array([236905.52749, -809.674253736, 1805.10894086, 
    -0.329922255065, -7.46908585919, -0.0529920002992])
poly_y[(92, 31)] = np.array([-162475.103485, 1608.84474212, 1190.63834514, 
    -3.38502685517, -1.2420022645, -0.450759148992])
poly_x[(92, 32)] = np.array([-392187.404797, 2002.59892781, -1604.78361765, 
    -1.93137968288, 6.66243235361, -0.236625114525])
poly_y[(92, 32)] = np.array([-154356.001448, 1560.49292014, 1303.62373494, 
    -3.31966748488, -1.70295738328, -0.642531447823])
poly_x[(92, 49)] = np.array([-345832.368553, 1624.92484725, -1696.34110409, 
    -1.20243860447, 7.02476265745, -0.131017042671])
poly_y[(92, 49)] = np.array([129512.5018, -1377.60062895, -1480.53016387, 
    3.00253900876, 2.16549681005, -0.268546003191])
poly_x[(92, 52)] = np.array([-341630.932157, 1599.61752259, -1639.1993025, 
    -1.16781235344, 6.7848989855, -0.266691743941])
poly_y[(92, 52)] = np.array([-143707.607872, 1487.78477496, 1379.62794905, 
    -3.1956071218, -1.97512363005, -0.626106755156])

class TpfGrid(object):
    """Grid transformation that is fitted to WCS information in the TPF files"""
    def __init__(self, campaign, channel):
        args = (campaign, channel)
        self._grid = GridRaDec2Pix(poly_x[args], poly_y[args])

    def apply_grid(self, ra, dec):
        """calculate pixel coordinates for given (RA,Dec) which can be floats, lists, or numpy.arrays"""
        return self._grid.apply_grid(ra=ra, dec=dec)
    
    def apply_grid_single(self, ra, dec):
        """calculate pixel coordinates for a single sky position (RA,Dec)"""
        return self._grid.apply_grid_single(ra=ra, dec=dec)
    