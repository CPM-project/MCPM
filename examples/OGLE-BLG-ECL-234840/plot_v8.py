import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


in_data = "run_6/run_6_e2_phot_prf_limit.dat"
in_model = "run_6/run_6_e2_phot.res"
out_file = "run_6/plot_eb234840_v8.png"

kwargs = {'color': 'red', 'marker': '.', 'ls': 'none'}
x_lim = [7500., 7528.]
y_lim = [-4000., 500.]

kwargs_1 = {'color': 'blue', 'ls': ':', 'lw': 2, 'zorder': 10}

xlabel = 'BJD - 2450000'
ylabel = 'delta flux'

band = np.arange(7500, 7508.0001)
kwargs_band = {'color': 'blue', 'lw': 2, 'zorder': 10}

################
# End of settings

(times, values, errors) = np.loadtxt(in_data, unpack=True)
(times_model, _, _, values_model) = np.loadtxt(in_model, unpack=True)

plt.errorbar(times, values, yerr=errors, **kwargs)
mask = (times_model > band[-1])
plt.plot(times_model[mask], values_model[mask], **kwargs_1)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xlim(x_lim)
plt.ylim(y_lim)

plt.plot(band, band*0., **kwargs_band)

plt.savefig(out_file)

