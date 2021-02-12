"""
Script for plotting CDF of periods from our results and LCDB
"""
import numpy as np
import matplotlib.pyplot as plt


def get_periods_from_LCDB(file_name="LC_DAT_PUB.TXT"):
    """
    extract distribution of periods from LCDB

    Returns:
        periods: *dict*
    """
    skip_lines = 5

    with open(file_name) as in_data:
        lines = in_data.readlines()

    periods = dict()
    for (i, line) in enumerate(lines):
        if i < skip_lines or len(line) == 1:
            continue

        if len(line[:8].strip()) > 0:
            period = line[117:130].strip()
            flag = line[115]
            quality_code = line[159:161].strip()
            if len(period) == 0 or flag in ['U', '<', '>'] or len(quality_code) == 0:
                # U - uncertain
                continue
            if quality_code in ['1', '1-']:
                continue
            id_ = line[:8].strip()
            if id_ == "0":
                continue

            periods[id_] = [float(period), quality_code]

    return periods


def get_orbital_elements(file_name="MPCORB.DAT"):
    """
    read the file with orbital elements

    Returns :
        elements: *dict* of *dicts*
    """
    skip_lines = 43

    with open(file_name) as in_data:
        lines = in_data.readlines()

    elements = dict()
    for (i, line) in enumerate(lines):
        if i < skip_lines:
            continue

        try:
            i_1 = line.index("(")
            i_2 = line.index(")")
        except Exception:
            continue

        id_ = line[i_1+1:i_2]
        words = line.split()
        elements[id_] = {
            'a': float(words[10])}

    return elements


def combine_periods_and_elements(periods, elements):
    """
    combine information on periods and orbital elements
    """
    missing = set(periods.keys()) - set(elements.keys())
    if len(missing) > 0:
        raise KeyError('missing orbital elements for: {:}'.format(missing))

    all_data = dict()
    for (id_, (period, flag)) in periods.items():
        all_data[id_] = {
            'period': period,
            'period_flag': flag,
            **elements[id_]}

    return all_data


def plot_period_vs_element(all_data, element, out_file, xlim=None,
                           ylim=None, log_y=False):
    """
    make a plot of rotation period vs. selected orbital element

    Parameters:
        all_data: *dict*
        element: *str*
            e.g. 'a' for semi-major axis
        out_file: *str* or *None*
        xlim: *list*
        log_y: *bool*
            do you want log scale on Y axis?
    """
    x = []
    y = []
    for (_, data) in all_data.items():
        x.append(data[element])
        y.append(data['period'])
    if element == 'a':
        xlabel = r'$a$ [AU]'
    else:
        xlabel = r'${:}$'.format(element)

    plt.plot(x, y, 'k,')

    font = 14

    plt.rcParams['font.size'] = font
    plt.rcParams['axes.linewidth'] = 2.0
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.11)
    if log_y:
        plt.gca().set_yscale('log')
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel(xlabel, fontsize=font)
    plt.ylabel(r'rotation period [h]', fontsize=font)
    plt.gca().tick_params(axis='both', which='both', direction='in',
                          top=True, right=True, labelsize=font)
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)


if __name__ == '__main__':

    periods = get_periods_from_LCDB()

    elements = get_orbital_elements()

    all_data = combine_periods_and_elements(periods, elements)

    plot_period_vs_element(all_data, 'a', "plot_a_P_LCDB.png", xlim=[1.7, 4.1],
                           ylim=[1.9, 200], log_y=True)

