"""
find Channel, Column & Row for given list of
RA and Dec (2nd and 3rd column in input file)
"""
import sys
import numpy as np

import K2fov


def get_CCR(ra, deg, campaign=9):
    """
    Get Channel, Column & Row for given ra, deg
    (both in deg and both np.arrays).

    Returns:
        on_silicon: np.array of *bool*
        channel: np.array of *int*
        column: np.array of *float*
        row: np.array of *float*
    """
    field_info = K2fov.fields.getFieldInfo(campaign)
    fovRoll_deg = K2fov.fov.getFovAngleFromSpacecraftRoll(field_info["roll"])
    field = K2fov.fov.KeplerFov(
        field_info["ra"], field_info["dec"], fovRoll_deg)

    ch = (field.pickAChannelList(ra, dec)+0.5).astype(int)

    col = np.zeros(len(ch))
    row = np.zeros(len(ch))

    for channel in set(ch):
        mask = np.array(ch == channel)
        (col[mask], row[mask]) = field.getColRowWithinChannelList(
            ra[mask], dec[mask], channel, wantZeroOffset=False,
            allowIllegalReturnValues=True)

    on_silicon = field.colRowIsOnSciencePixelList(col, row)

    return (on_silicon, ch, col, row)


if __name__ == '__main__':
    with open(sys.argv[1]) as data_in:
        fdata = []
        for line in data_in.readlines():
            fdata.append(line.split())

    ra = np.array([float(t[1]) for t in fdata])
    dec = np.array([float(t[2]) for t in fdata])

    (on_silicon, ch, col, row) = get_CCR(ra, dec)

    for i in range(len(on_silicon)):
        text = "F  0    0.0    0.0"
        if on_silicon[i]:
            text = "T {:} {:6.1f} {:6.1f}".format(ch[i], col[i], row[i])
        print("{:} {:} {:}  {:}".format(*fdata[i][:3], text))
