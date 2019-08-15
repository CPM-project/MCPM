"""
A script for plotting a corner plot from a single file.
"""
import sys
import numpy as np
import configparser
import matplotlib.pyplot as plt

import corner


if len(sys.argv) != 2:
    raise ValueError('one parameter needed - config file')

config = configparser.ConfigParser()
config.optionxform = str
config.read(sys.argv[1])

section = 'DEFAULT'

input_file = config.get(section, 'input_file')
try:
    in_data = np.load(input_file)
except Exception:
    in_data = np.loadtxt(input_file)
output_file = None
if 'output_file' in config[section]:
    output_file = config.get(section, 'output_file')

kwargs = {}
key = 'quantiles'
if key in config[section]:
    kwargs[key] = [float(t) for t in config.get(section, key).split()]
if 'bins' in config[section]:
    kwargs['bins'] = config.getint(section, 'bins')
if 'weights_column' in config[section]:
    column = config.getint(section, 'weights_column')-1
    kwargs['weights'] = in_data[:, column]
kwargs['show_titles'] = True

data = []
ranges = []
labels = []
for section in config.sections():
    column = config.getint(section, 'column') - 1
    selected = in_data[:, column]
    if 'subtract' in config[section]:
        selected -= config.getfloat(section, 'subtract')
    data.append(selected)
    if 'label' in config[section]:
        labels.append(config.get(section, 'label'))
    if 'range' in config[section]:
        text = config.get(section, 'range')
        if len(text.split()) == 1:
            range_ = float(text)
        elif len(text.split()) == 2:
            range_ = [float(t) for t in text.split()]
        else:
            raise ValueError('wrong length of range')
        ranges.append(range_)

if len(labels) > 0:
    if len(labels) != len(data):
        raise ValueError('label problem')
    kwargs['labels'] = labels
if len(ranges) > 0:
    if len(ranges) != len(data):
        raise ValueError('range problem')
    kwargs['range'] = ranges

# main function:
figure = corner.corner(np.array(data).T, **kwargs)

if output_file is None:
    plt.show()
else:
    plt.savefig(output_file)

