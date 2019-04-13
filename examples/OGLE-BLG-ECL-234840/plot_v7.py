import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


in_data = "run_6/run_6_e2_phot.res"
out_file = "run_6/plot_eb234840_v7.png"

height_ratio = 5
kwargs_1 = {'color': 'blue', 'lw': 2, 'zorder': 10}
kwargs_2 = {'color': 'red', 'marker': '.', 'ls': 'none'}
kwargs_2_0 = {'color': 'black', 'ls': 'dashed', 'lw': 1.5, 'zorder': 10}
x_lim = [7500., 7528.]
y_lim_2 = [-220, 220]
yticks_2 = [200, 100, 0, -100, -200]

xlabel = 'BJD - 2450000'
ylabel = 'delta flux'
ylabel_2 = 'residuals'

################
# End of settings

(times, values, errors, model) = np.loadtxt(in_data, unpack=True)

gs = gridspec.GridSpec(2, 1, height_ratios=[height_ratio, 1])

plt.subplot(gs[0])
plt.errorbar(times, values, yerr=errors, **kwargs_2)
plt.plot(times, model, **kwargs_1)
plt.ylabel(ylabel)
plt.xlim(x_lim)

plt.subplot(gs[1])
plt.errorbar(times, values-model, yerr=errors, **kwargs_2)
plt.plot(x_lim, 0.*np.array(x_lim), **kwargs_2_0)
plt.xlabel(xlabel)
plt.ylabel(ylabel_2)
plt.xlim(x_lim)
plt.ylim(y_lim_2)
plt.yticks(yticks_2)

plt.savefig(out_file)

