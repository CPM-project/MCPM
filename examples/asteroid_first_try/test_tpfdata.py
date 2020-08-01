import sys
import astropy

from MCPM.tpfdata import TpfData


if len(sys.argv) != 3:
    raise ValueError("2 arguments required: EPIC id and campaign id, " +
                     "e.g., 200070197 92")

print("astropy", astropy.__version__)

epic = sys.argv[1]
campaign = int(sys.argv[2])

tpf = TpfData(epic, campaign)

