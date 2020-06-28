"""
Calculate proper motion statistics for all input *.ephem_interp_CCR files.
"""
import sys
import numpy as np


pixel_scale = 3.96  # pixel scale of Kepler camera

def calculate_proper_motion_stats(file_name):
    """
    Calculate proper motion based on time and (x,y) pixels.
    Return median, min, and max [arcsec/h], plus number of epochs.
    """
    (time, channel, x, y) = np.loadtxt(file_name, unpack=True,
                                       usecols=(0, 3, 4, 5))

    motion = []
    for c in set(channel):
        if c == 0.:
            continue
        mask = (channel == c)
        time_ = time[mask]
        x_ = x[mask]
        y_ = y[mask]

        dx = x_[1:] - x_[:-1]
        dy = y_[1:] - y_[:-1]
        dt = time_[1:] - time_[:-1]

        motion_ = np.sqrt(dx**2 + dy**2) / dt
        motion_ *= pixel_scale / 24.
        motion += motion_.tolist()

    return (np.median(motion), np.min(motion), np.max(motion), len(motion))


if __name__ == '__main__':
    len_max = max([len(f) for f in sys.argv[1:]])

    fmt = "{:" + str(len_max) + "} {:5.1f} {:5.1f} {:5.1f} {:4}"
    for file_name in sys.argv[1:]:
        print(fmt.format(file_name, *calculate_proper_motion_stats(file_name)))

