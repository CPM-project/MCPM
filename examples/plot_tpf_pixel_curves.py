import matplotlib.pyplot as plt

from MCPM.cpmfitsource import CpmFitSource


def plot_tpf_data(ra, dec, channel, campaign, file_out, half_size=2):
    """
    Plot TPF data for given settings.
    """
    cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, channel=channel)
    cpm_source.set_pixels_square(half_size)
    cpm_source.plot_pixel_curves()
    plt.savefig(file_out)
    plt.close()

if __name__ == "__main__":
    plot_tpf_data(271.2375417, -28.6278056, 52, 92, "ob160975_pixel_curves.png")
    plot_tpf_data(271.001083, -28.155111, 52, 91, "ob160795_pixel_curves.png")
    plot_tpf_data(271.354292, -28.005583, 52, 92, "ob160980_pixel_curves.png")
    plot_tpf_data(269.5648750, -27.9635833, 31, 92, "ob160940_pixel_curves.png")
