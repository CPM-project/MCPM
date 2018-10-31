import matplotlib.pyplot as plt
#from matplotlib import rc
from matplotlib import rcParams

from MCPM.cpmfitsource import CpmFitSource


def plot_tpf_data(ra, dec, channel, campaign, file_out, half_size=2,
                  stars_subtract=[], adjust=None, xlabel=None, ylabel=None):
    """
    Plot TPF data for given settings.
    """
    cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, channel=channel)
    cpm_source.set_pixels_square(half_size)
    for (ra, dec, flux) in stars_subtract:
        cpm_source.subtract_flux_from_star(ra, dec, flux)
    cpm_source.plot_pixel_curves()
    if adjust is not None:
        plt.subplots_adjust(**adjust)
    if xlabel is not None:
        plt.figtext(0.51, 0.004, xlabel)
    if ylabel is not None:
        plt.figtext(0.002, 0.5, ylabel, rotation=90)

    plt.savefig(file_out)
    plt.close()

if __name__ == "__main__":
    
    #stars_0241 = [[270.63370, -27.52653, 30.e3]]
    stars_0241 = [[270.63370, -27.52653, 16996.5]]
    plot_tpf_data(270.6323333, -27.5296111, 49, 91, "ob160241_c91_pixel_curves.png",
        half_size=3, stars_subtract=stars_0241)
    plot_tpf_data(270.6323333, -27.5296111, 49, 92, "ob160241_c92_pixel_curves.png",
        half_size=3, stars_subtract=stars_0241)

    plot_tpf_data(269.5648750, -27.9635833, 31, 92, "ob160940_pixel_curves.png")

    default = rcParams['font.size']
    rcParams['font.size'] = 18

    plot_tpf_data(
        271.2375417, -28.6278056, 52, 92, "ob160975_pixel_curves.png",
        adjust={"left": 0.07, "bottom":0.06, "right":.995, "top":.995},
        xlabel="BJD-2450000", ylabel='counts')
    plot_tpf_data(
        271.001083, -28.155111, 52, 91, "ob160795_pixel_curves.png",
        adjust={"left": 0.07, "bottom":0.06, "right":.995, "top":.995},
        xlabel="BJD-2450000", ylabel='counts')

    rcParams['font.size'] = default

    plot_tpf_data(271.354292, -28.005583, 52, 92, "ob160980_pixel_curves.png")
    
    plot_tpf_data(269.9291250, -28.4108333, 31, 91, "eb234840_pixel_curves.png")

# Isolated stars:
# blg224.1 217850
# 217850 18:02:55.72 -27:52:28.9 1748.12 3011.50 14.511  0.944 13.567   6   0 0.011 1029   3 0.012
# blg224.1 222357
# 222357 18:02:58.64 -27:51:57.5 1867.60 3160.89 14.236  0.711 13.524   6   0 0.011 1031   1 0.010
# Kp magnitudes predicted using Wei's R_I=1, A_I=1 ("random" part of his plot)
# 14.327 & 14.130
# Hence fluxes:
# 18585.4 22278.8
    plot_tpf_data(270.7321739, -27.8746756, 52, 91, "isolated_1_pixel_curves_91.png")
    plot_tpf_data(270.7321739, -27.8746756, 52, 92, "isolated_1_pixel_curves_92.png")
#    plot_tpf_data(270.7443333, -27.8659933, 49, 91, "isolated_2_pixel_curves_91.png")
#    plot_tpf_data(270.7443333, -27.8659933, 49, 92, "isolated_2_pixel_curves_92.png")

