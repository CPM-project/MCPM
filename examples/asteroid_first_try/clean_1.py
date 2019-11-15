"""
Removes points which are for sure not on the K2C9 superstamp.
Further scripts are significantly slower, so we want to remove
obvious epochs ASAP.
"""
import numpy as np
import sys

with open(sys.argv[1]) as data:
    lines = data.readlines()

ra = np.array([float(line.split()[1]) for line in lines])
dec = np.array([float(line.split()[2]) for line in lines])

mask = np.ones(len(ra), dtype=bool)

mask &= (ra > 267.8) & (ra < 271.6)
mask &= (dec > -29.4) & (dec < -26.7)

a1 = 0.199534997
x1 = 267.992500
y1 = -29.362592

mask &= (dec - (a1 * (ra - x1) + y1) > 0.)

a2 = 0.338481274
x2 = 270.828908
y2 = -26.906844

mask &= (dec - (a2 * (ra - x2) + y2) < 0.)

a3 = -3.69091035
x3 = 269.969197
y3 = -27.286145

mask_3 = (dec - (a3 * (ra - x3) + y3) < 0.)

a4 = -3.69599634
x4 = 270.444758
y4 = -27.373938

mask_4 = (dec - (a4 * (ra - x4) + y4) > 0.)

mask &= (mask_3 | mask_4)

for (line, mask_) in zip(lines, mask):
    if mask_:
        print(line[:-1])

