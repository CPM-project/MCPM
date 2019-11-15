Here I show how I reduced photometry for a single asteroid. It's very inefficient, but it works.

First, get the ephemeris:
```
python get_coords_1.py in_1.txt
```
This produces `ephem_C9a_v1/30617.ephem` and `ephem_C9b_v1/30617.ephem`.

Then remove the epochs which are not inside K2C9 superstamp:
```
python clean_1.py ephem_C9b_v1/30617.ephem > ephem_C9b_v1/30617.ephem_v2
python clean_2.py ephem_C9b_v1/30617.ephem_v2 > ephem_C9b_v1/30617.ephem_v3
```

Next step is to get Channel, Column, Row (CCR) for each epoch:

...

