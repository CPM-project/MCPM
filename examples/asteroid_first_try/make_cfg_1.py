"""
Prepare settings for hundreds of cfg files for a single asteroid
"""
import os


dir_out_cfg = "cfg_files"
dir_out = "out_files"
object_id = "30617"
input_file = "ephem_C9b_v1/30617.ephem_interp_CCR"
dt = 2.  # For each epoch we will remove from training
# the epochs within +-dt [days]

# directories:
if not os.path.isdir(dir_out_cfg):
    os.mkdir(dir_out_cfg)
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

# read data:
with open(input_file) as data_in:
    in_data = []
    for line in data_in.readlines():
        in_data.append(line.split())

# print output:
i1 = 0
for (i_, line) in enumerate(in_data):
    if line[3] == '0':
        continue
    i1 += 1
    file_cfg = "{:}/{:}_{:04}.cfg".format(dir_out_cfg, object_id, i1)
    t_1 = float(line[0]) - dt
    t_2 = float(line[0]) + dt
    file_out_1 = "{:}/{:}_{:04}_v1.dat".format(dir_out, object_id, i1)
    file_out_2 = "{:}/{:}_{:04}_v2.dat".format(dir_out, object_id, i1)
    print(file_cfg, line[1], line[2], line[3],
          t_1, t_2, file_out_1, file_out_2)
