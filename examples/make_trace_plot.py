"""
Make trace plot for calculations.
All settings are read from cfg file:

python make_trace_plot.py event_K2.cfg

python make_trace_plot.py event_K2.cfg trace_plot.png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import configparser
import os

import read_config


def plot_trace_plot(parameters, file_in, file_out):
    """
    Read .npy file file_in, plot the trace plot, and save to file_out
    (or display on screen if file_out is *None*).
    """
    data = np.load(file_in)
    if len(data.shape) == 4:
        data = data[0, ...]
    n_parameters = len(parameters)
    alpha = 0.5

    grid = gridspec.GridSpec(n_parameters, 1, hspace=0)

    plt.figure(figsize=(7.5, 10.5))  # A4 is 8.27 11.69.
    plt.subplots_adjust(left=0.13, right=0.97, top=0.99, bottom=0.05)

    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.4

    for i in range(n_parameters):
        if i == 0:
            plt.subplot(grid[i])
            ax0 = plt.gca()
        else:
            plt.gcf().add_subplot(grid[i], sharex=ax0)
        if parameters[i] == 't_0':
            shift = int(np.average(data[:, :, i])+0.5)
            parameters[i] = r'$\Delta$ t_0'
        else:
            shift = 0.
        plt.ylabel(parameters[i])
        for j in range(data.shape[0]):
            vector = data[j, :, i] - shift
            plt.plot(np.arange(len(vector)), vector, alpha=alpha)
        plt.xlim(0, len(vector))
        plt.gca().tick_params(axis='both', which='both', direction='in',
                              top=True, right=True)
        if i != n_parameters - 1:
            plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.gca().set_prop_cycle(None)

    plt.xlabel('step count')

    if file_out is None:
        plt.show()
    else:
        plt.savefig(file_out)


if __name__ == '__main__':

    config_file = sys.argv[1]
    if len(sys.argv) == 3:
        file_out = sys.argv[2]
    else:
        file_out = None

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)
    settings = read_config.read_EMCEE_options(config, check_files=False)
    parameters = settings[1]
    file_in = settings[4]['file_posterior']
    if file_in == "":
        file_in = os.path.splitext(config_file)[0] + ".posterior.npy"

    plot_trace_plot(parameters, file_in, file_out)

