import matplotlib.pyplot as plt

from MCPM.cpmfitsource import CpmFitSource


def plot_tpf_data(ra, dec, channel, campaign, file_out, half_size=2, stars_subtract=[]):
    """
    Plot TPF data for given settings.
    """
    cpm_source = CpmFitSource(ra=ra, dec=dec, campaign=campaign, channel=channel)
    cpm_source.set_pixels_square(half_size)
    for (ra, dec, flux) in stars_subtract:
        cpm_source.subtract_flux_from_star(ra, dec, flux)
    cpm_source.plot_pixel_curves()
    plt.savefig(file_out)
    plt.close()

if __name__ == "__main__":
    
    #stars_0241 = [[270.63370, -27.52653, 30.e3]]
    stars_0241 = [[270.63370, -27.52653, 16996.5]]
    plot_tpf_data(270.6323333, -27.5296111, 49, 91, "ob160241_c91_pixel_curves.png", 
        half_size=3, stars_subtract=stars_0241)
    plot_tpf_data(270.6323333, -27.5296111, 49, 92, "ob160241_c92_pixel_curves.png", 
        half_size=3, stars_subtract=stars_0241)

    plot_tpf_data(271.001083, -28.155111, 52, 91, "ob160795_pixel_curves.png")
    plot_tpf_data(269.5648750, -27.9635833, 31, 92, "ob160940_pixel_curves.png")
    plot_tpf_data(271.2375417, -28.6278056, 52, 92, "ob160975_pixel_curves.png")
    plot_tpf_data(271.354292, -28.005583, 52, 92, "ob160980_pixel_curves.png")
    
    plot_tpf_data(269.9291250, -28.4108333, 31, 91, "eb234840_pixel_curves.png")
