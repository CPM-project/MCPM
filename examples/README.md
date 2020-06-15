### MCPM example usage.

First, let's try to reproduce Fig. 4 from [MCPM paper](https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..54P/abstract). We will use a script that uses MulensModel, but with flux = 0, the MM input is effectively ignored. There are some explanations in `eb_234840.cfg`. Run:

```
python evaluate_MM_MCPM_v1.py eb_234840.cfg
```

When you plot file `eb_234840_2.dat`, then it should look like Fig. 4 from MCPM paper.


