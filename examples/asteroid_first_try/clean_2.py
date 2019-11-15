"""
Removes points which are not on the K2C9 superstamp.
"""
import sys

from K2fov.c9 import inMicrolensRegion


with open(sys.argv[1]) as data:
    lines = data.readlines()

for line in lines:
    ra = float(line.split()[1])
    dec = float(line.split()[2])
    if inMicrolensRegion(ra, dec):
        print(line[:-1])

