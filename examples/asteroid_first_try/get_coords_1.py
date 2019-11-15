"""
Get K2C9-based positions for a list of Solar System targets.
"""
import sys
import os

from astroquery.jplhorizons import Horizons


input_file = sys.argv[1]
location = '@-227'
begins = ["2016-04-22 12:00:00", "2016-05-22 17:00:00"]
ends = ["2016-05-18 23:59:00", "2016-07-02 23:59:00"]
step = "15m"
directories = ["ephem_C9a_v1", "ephem_C9b_v1"]
columns = ['datetime_jd', 'RA', 'DEC', 'V']

### Settings end here.

def print_output(ephem, columns, directory, file_name_root):
    """save selected columns from ephemeris to a file"""
    file_name = os.path.join(directory, file_name_root + '.ephem')
    fmt = "{:.5f} {:.6f} {:.6f} {:.3f}\n"

    if not os.path.isdir(directory):
        os.mkdir(directory)

    with open(file_name, 'w') as out:
        columns_ = [ephem[t].data for t in columns]
        for i in range(len(ephem)):
            out.write(fmt.format(*[c[i] for c in columns_]))


epochs_1 = {'start': begins[0], 'stop': ends[0], 'step': step}
epochs_2 = {'start': begins[1], 'stop': ends[1], 'step': step}
kwargs_1 = {'location': location, 'epochs': epochs_1}
kwargs_2 = {'location': location, 'epochs': epochs_2}

with open(sys.argv[1]) as in_data:
    for line in in_data.readlines():
        id_name = line.split()[0]
        id_name_orig = id_name
        try:
            horizons = Horizons(id=id_name, **kwargs_1)
            ephem = horizons.ephemerides()
        except:
            id_name = id_name.replace("_", " ")
            try:
                horizons = Horizons(id=id_name, **kwargs_1)
                ephem = horizons.ephemerides()
            except:
                raise
        print_output(ephem, columns, directories[0], id_name_orig)
        horizons = Horizons(id=id_name, **kwargs_2)
        ephem = horizons.ephemerides()
        print_output(ephem, columns, directories[1], id_name_orig)

