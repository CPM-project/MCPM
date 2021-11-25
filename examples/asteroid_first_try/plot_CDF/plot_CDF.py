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
    plt.rcParams['axes.linewidth'] = 1.4
    
    plt.plot(x, y, 'k,')

    font = 15

    plt.rcParams['font.size'] = font
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
    plt.close()


def get_our_results(file_1="results_1.tex", file_2="results_2.tex",
                    file_3="results_3.tex", noise=True):
    """
    Read our results.

    The file_3 results will be random values.
    """
    max_p = 1000.  # This value is assumed as maximum for file_3
    min_p = 90. # We assume file_3 has periods longer than that

    out_1 = []
    for file_ in [file_1, file_2]:
        with open(file_) as in_data:
            for line in in_data.readlines():
                word = line.split()[-2][1:]
                t = "\\"
                if "^" in word:
                    t = "^"
                out_1.append(float(word[:word.index(t)]))

    with open(file_3) as in_data:
        n = len(in_data.readlines())

    out_2 = np.random.rand(n) * (max_p - min_p) + min_p
    out_2 = out_2.tolist()

    return (out_1, out_2)


def get_Pal_et_al_2020_results(file_in="Pal_2020/release.merge"):
    """
    read data from Pal et al. 2020
    https://ui.adsabs.harvard.edu/abs/2020ApJS..247...26P/abstract
    https://archive.konkoly.hu/pub/tssys/dr1/
    """
    out = dict()
    with open(file_in) as in_data:
        for line in in_data.readlines():
            words = line.split()
            if words[13] == "-":  # We skip objects with no semi-major axis info
                continue
            out[words[0]] = {
                'period': float(words[3]),
                'a': float(words[13])}
#    (id_, p, a) = np.loadtxt(file_in, unpack=True, usecols=(0, 3, 13))
#    print(id_)
#    print(p)
#    print(a)
    return out


def plot_CDF(all_data, our_data, a_min=1.14, a_max=5.29,
             flags_accepted=['3', '3-', '2+'], 
             Pal20_data=None,
             out_file=None):
    """
    Plot CDF.
    """
    data_1 = []
    for (_, data) in all_data.items():
        if a_min < data['a'] < a_max and data['period_flag'] in flags_accepted:
            data_1.append(data['period'])

    plt.rcParams['axes.linewidth'] = 1.4
    font = 15
    plt.rcParams['font.size'] = font

    data_1 = sorted(data_1)
    n_1 = len(data_1)
    plt.plot(data_1, np.arange(n_1)/n_1, label='LCDB', ls='dotted', lw='3')

    data_2 = sorted(our_data)
    n_2 = len(data_2)
    plt.plot(data_2, np.arange(n_2)/n_2, label='this work')

    if Pal20_data is not None:
        data_3 = []
        for (_, data) in Pal20_data.items():
            if a_min < data['a'] < a_max:
                data_3.append(data['period'])
        data_3 = sorted(data_3)
        n_3 = len(data_3)
        plt.plot(data_3, np.arange(n_3)/n_3, label='Pal et al. (2020)')

    plt.legend()
    plt.gca().set_xscale('log')
    plt.xlim(0.9, 1050)
    plt.xlabel('rotation period [h]', fontsize=font)
    plt.ylabel('CDF', fontsize=font)
    plt.gca().tick_params(axis='both', which='both', direction='in',
                          top=True, right=True, labelsize=font)
    plt.gca().tick_params(width=1.4, which='both')
    plt.subplots_adjust(left=0.11, right=0.97, top=0.99, bottom=0.12)

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

    (our_1, our_2) = get_our_results()
    our_data = our_1 + our_2

    data_Pal20 = get_Pal_et_al_2020_results()

#    plot_CDF(all_data, our_data)
    plot_CDF(all_data, our_1, out_file="plot_CDF.png")
#    plot_CDF(all_data, our_1, Pal20_data=data_Pal20)

