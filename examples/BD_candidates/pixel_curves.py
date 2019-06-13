import sys

sys.path.append("..")
import plot_tpf_pixel_curves


settings_1 = {
    'ra': 271.390792, 'dec': -28.542811, 'channel': 52, 'campaign': 92,
    'file_out': 'kb160133_pixel_curves.png'}
settings_2 = {
    'ra': 271.001083, 'dec': -28.155111, 'channel': 52, 'campaign': 91,
    'file_out': 'ob160795_pixel_curves.png'}
settings_3 = {
    'ra': 269.886542, 'dec': -28.407417, 'channel': 31, 'campaign': 91,
    'file_out': 'ob160813_pixel_curves.png'}
settings_4 = {
    'ra': 269.202417, 'dec': -28.655250, 'channel': 31, 'campaign': 92,
    'file_out': 'ob161231_pixel_curves.png'}

settings = [settings_1, settings_2, settings_3, settings_4]

for kwargs in settings:
    plot_tpf_pixel_curves.plot_tpf_data(**kwargs)
    print(kwargs['file_out'])

