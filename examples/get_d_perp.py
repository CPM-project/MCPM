import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

import MulensModel as MM


def get_d_perp(ra_deg, dec_deg, t_0_par, ephemeris_file):
    """
    extract D_perp for satellite for given epoch
    """
    parameters = {'t_0': t_0_par, 'u_0': 0, 't_E': 100.,
                  'pi_E_N': 2.**-0.5, 'pi_E_E': 2.**-0.5, 't_0_par': t_0_par}
    params = MM.ModelParameters(parameters)
    coords = MM.Coordinates(SkyCoord(ra_deg, dec_deg, unit=u.deg))
    satellite = MM.SatelliteSkyCoord(ephemerides_file=ephemeris_file)
    ephemeris = satellite.get_satellite_coords([t_0_par])
    trajectory = MM.Trajectory([t_0_par], params, coords=coords,
                               parallax={'satellite': True},
                               satellite_skycoord=ephemeris)
    return np.sqrt(trajectory.x**2+trajectory.y**2)[0]


if __name__ == '__main__':
    ra = 271.390792
    dec = -28.542811
    epoch = 2457563.
    ephemeris_file = 'K2_ephemeris_01.dat'

    print(get_d_perp(ra, dec, epoch, ephemeris_file))
