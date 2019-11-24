Here I show how I reduced photometry for a single asteroid. It's very inefficient, but it works.

First, get the ephemeris:
```
python get_coords_1.py in_1.txt
```
This produces `ephem_C9a_v1/30617.ephem` and `ephem_C9b_v1/30617.ephem`.

Get epochs for specific target:
```
python get_K2_epochs.py in_2.yaml
```
It produces `ephem_C9b_v1/30617_time_tpf.dat` and uses coords which are currently found in non-automatic way (that will be corrected).

Interpolate ephemeris (_requires corrections at edges_):
```
python interpolate_ephemeris.py > ephem_C9b_v1/30617.ephem_interp
```

Then remove the epochs which are not inside K2C9 superstamp:
```
python clean_1.py ephem_C9b_v1/30617.ephem_interp > ephem_C9b_v1/30617.ephem_interp_v2
```

Next step is to get Channel, Column, Row (CCR) for each epoch:
```
python get_CCR.py ephem_C9b_v1/30617.ephem_interp_v2 > ephem_C9b_v1/30617.ephem_interp_CCR
```

After that, we prepare configuration files - one per epoch:
```
python make_cfg_1.py > settings_30617.txt
python prepare_1.py settings_30617.txt template_1.cfg
```

...

