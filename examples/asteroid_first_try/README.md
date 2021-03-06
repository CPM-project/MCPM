Here I show how I reduced photometry for a single asteroid. It's very inefficient, but it works.

First, get the ephemeris:
```
python get_coords_1.py in_1.txt
```
This produces `ephem_C9a_v1/30617.ephem` and `ephem_C9b_v1/30617.ephem`.

Interpolate ephemeris, remove the epochs which are not inside K2C9 superstamp, and get epochs (here we use C9b  only):
```
python format_ephemeris.py ephem_C9b_v1/30617.ephem > ephem_C9b_v1/30617.ephem_interp_v2
```

Next step is to get Channel, Column, Row (CCR) for each epoch:
```
python get_CCR.py ephem_C9b_v1/30617.ephem_interp_v2 > ephem_C9b_v1/30617.ephem_interp_CCR
```

After that, we prepare configuration files - one per epoch:
```
python make_cfg_1.py ephem_C9b_v1/30617.ephem_interp_CCR 30617 > settings_30617_92.txt
python prepare_1.py settings_30617_92.txt template_1.cfg
```

We need file with Kepler ephemeris:
```
cp ../K2_ephemeris_01.dat .
```

Now we have to run all the cfg files. I do it this way (this takes hours!):

```
awk 'BEGIN{print "#! /bin/tcsh"}{split($1, a, "."); split (a[1], b, "/");  printf "(time python3 ../evaluate_MM_MCPM_v1.py %s > out_files/%s.out ) >& out_files/%s.err\n", $1, b[2], b[2]}' settings_30617_92.txt > run_30617_92.sh
chmod u+x run_30617_92.sh
(time ./run_30617_92.sh > run_30617_92.out ) >& run_30617_92.err &
```

Extract specific epoch from each file:
```
awk '{printf "%s %.5f\n", $8, ($6+$7)/2}' settings_30617_92.txt | python extract_epoch.py /dev/stdin > lc_30617_92_v1.dat
```
and from the second type of photometry:
```
awk '{printf "%s %.5f\n", $9, ($6+$7)/2}' settings_30617_92.txt | python extract_epoch.py /dev/stdin > lc_30617_92_v2.dat
```

and plot:
```
python plot_30617.py
```

For C9a one has to change directory and file names and the parameters above:
 - C9a -> C9b
 - 91 -> 92


### Difference image

We want to plot postage stamp images for selected epochs. And we the images will be after the CPM is applied, i.e., instrumental trends are removed. Selected epochs for two bright objects (first one moves very slow, second - very fast):

```
python ../evaluate_MM_MCPM_v1.py 30617_plot.cfg
python ../evaluate_MM_MCPM_v1.py 14352_plot.cfg
```

If you want to plot raw data, then change `type = cpm_residuals` to `type = pixel_flux` and re-run the scripts. Note the changes in color scale.

