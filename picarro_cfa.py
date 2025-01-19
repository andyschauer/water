#!/usr/bin/env python3
"""
picarro_cfa.py

20220908 - aschauer@uw.edu

This script is meant to be run after picarro_h5.py if the data set exclusively contains continuous flow analysis (CFA) data.

Change Log:
    20220908 - saved from picarro_inj.py


Notes:
    - use this to time:
        start_time = t.time()
        print("\n--- %s seconds ---" % (t.time() - start_time))

ToDo:

"""

__author__ = "Andy Schauer"
__email__ = "aschauer@uw.edu"
__last_modified__ = "2024-10-31"
__version__ = "0.2"
__copyright__ = "Copyright 2025, Andy Schauer"
__license__ = "Apache 2.0"


# -------------------- imports --------------------
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN  # , INLINE
import datetime as dt
import h5py
import json
import matplotlib.pyplot as pplt
from natsort import natsorted
import numpy as np
import os
from picarro_lib import *
from scipy.signal import argrelextrema
import shutil
import sys
import time as t
import warnings
import webbrowser



# -------------------- functions ------------------------------------

def isofft(data):
    pass




# -------------------- get instrument information --------------------
""" Get specific picarro instrument whose data is being processed as well as some associated information. Populate
this function with your own instrument(s). The function get_instrument() is located in picarro_lib.py."""
instrument = get_instrument()


# -------------------- paths --------------------
python_dir = get_path("python")
project_dir = f"{get_path('project')}{instrument['name'].lower()}/"
run_dir = os.path.join(project_dir, 'runs/')


# -------------------- identify run --------------------
run_list = natsorted(os.listdir(run_dir))
print('\nChoose from the run list below:')
[print(f'    {i}') for i in run_list]
identified_run = 0
while identified_run == 0:
    run_search = input('\nEnter the run you wish to process: ')
    isdir = [run_search[0: len(run_search)] in x for x in run_list]
    if len(np.where(isdir)[0]) == 1:
        identified_run = 1
        run = run_list[np.where(isdir)[0][0]]
        run_dir += f'{run}/'
        print(f'\n    Processing run {run}...')
    else:
        print('\n** More than one run found. **\n')


# -------------------- prepare run directory --------------------
archive_dir = os.path.join(run_dir, 'archive/')
if os.path.isdir(archive_dir) is False:
    os.mkdir(archive_dir)

shutil.copy2(os.path.join(python_dir, 'py_report_style.css'), os.path.join(run_dir, 'py_style.css'))
if os.path.isdir(os.path.join(archive_dir, 'python_archive')) is False:
    os.mkdir(os.path.join(archive_dir, 'python_archive'))
[shutil.copy2(os.path.join(python_dir, script), os.path.join(archive_dir, f"python_archive/{script}_ARCHIVE_COPY")) for script in python_scripts]



# -------------------- identify hdf5 file --------------------
hdf5_list = make_file_list(os.path.join(run_dir), '.hdf5')
if len(hdf5_list) > 1:
    print('\nChoose from the file list below:')
    [print(f'    {i}') for i in hdf5_list]
    identified_file = 0
    while identified_file == 0:
        hdf5_file_search = input('Enter the filename you wish to process: ')
        isfile = [hdf5_file_search[0: len(hdf5_file_search)] in x for x in hdf5_list]
        if len(np.where(isfile)[0]) == 1:
            identified_file = 1
            hdf5_file = hdf5_list[np.where(isfile)[0][0]]
            print(f'\n    Processing file {hdf5_file}...')
        else:
            print('\n** More than one file found. **\n')
else:
    hdf5_file = hdf5_list[0]

# np.seterr(all='raise')


# -------------------- read in data from hdf5 file --------------------

with h5py.File(f'{run_dir}{hdf5_file}', 'r') as hf:
    datasets = list(hf.keys())
    for data in datasets:
        globals()[data] = np.asarray(hf[data])

dD = dDp.copy()


# -------------------- screen data by removing H2O ppm hiccups --------------------


time_diff = np.diff(time)
time_diff = np.append(time_diff, np.mean(time_diff))

kernel_size = 20
kernel = np.ones(kernel_size) / kernel_size

H2O_convolved = np.convolve(H2O, kernel, mode='same')

H2O_diff = np.diff(H2O_convolved)
H2O_diff = np.append(H2O_diff, H2O_diff[-1]) # duplicate last value so length of array is the same as the original length

dH2O_dT = H2O_diff / time_diff

dH2O_dT_convolved = np.convolve(dH2O_dT, kernel, mode='same')
dH2O_dT2 = np.diff(dH2O_dT_convolved)
dH2O_dT2 = np.append(dH2O_dT2, dH2O_dT2[-1])




gdi = np.where((dH2O_dT>-30) & (dH2O_dT<30) & (H2O>15000))[0]




# valco timing
valco_diff = np.diff(valco_pos)
valco_movement = np.where(np.diff(valco_pos))[0]

v1 = np.where(valco_pos==1)[0]+25 # melter
v2 = np.where(valco_pos==2)[0]+25 # heaviest in dD
v3 = np.where(valco_pos==3)[0]+25 # second heaviest in dD
v4 = np.where(valco_pos==4)[0]+25 # second lightest in dD
v5 = np.where(valco_pos==5)[0]+25 # lightest in dD
v6 = np.where(valco_pos==6)[0]+25 # standby water


# campaign timing


sys.exit()


dD = dDp.copy()
# import sys
# sys.exit()

# Find transitions
time_diff = np.diff(time)
time_diff = np.append(time_diff, np.mean(time_diff))

kernel_size = 50
kernel = np.ones(kernel_size) / kernel_size
dD_convolved = np.convolve(dD, kernel, mode='same')
    # this uses zeros to pad the ends. it looks like half the kernel is affected by this. so if kernel size is 50, then the first and last 25 points are affected by this zero padding.

dD_diff = np.diff(dD_convolved)
dD_diff = np.append(dD_diff, dD_diff[-1]) # duplicate last value so length of array is the same as the original length

ddD_dT = dD_diff / time_diff

ddD_dT_convolved = np.convolve(ddD_dT, kernel, mode='same')
ddD_dT2 = np.diff(ddD_dT_convolved)
ddD_dT2 = np.append(ddD_dT2, ddD_dT2[-1])

transitions = np.where(np.logical_or(ddD_dT_convolved > 0.01, ddD_dT_convolved < -0.01))[0]

allindices = np.where(dD)
gdi = np.setdiff1d(allindices, transitions)


# find peak slope during a transition
local_extreme_range_size = 50
local_max = argrelextrema(ddD_dT, np.greater, axis=0, order=local_extreme_range_size, mode='clip')[0]
peak_tops = np.asarray(ddD_dT[local_max] > 0.05).nonzero()[0]
local_max = local_max[peak_tops]

local_min = argrelextrema(ddD_dT, np.less, axis=0, order=local_extreme_range_size, mode='clip')[0]
peak_troughs = np.asarray(ddD_dT[local_min] < -0.05).nonzero()[0]
local_min = local_min[peak_troughs]

peak_transition_slope_index = np.concatenate((local_min, local_max))


# get start of each transition
transitions_reverse = transitions[::-1]
start_of_each_transition = np.where(np.diff(transitions_reverse) < -1000)[0]
start_transition_indices = transitions_reverse[start_of_each_transition]
start_transition_indices = start_transition_indices[::-1]
start_transition_indices = start_transition_indices[0:-1]

# get the end of each transition
end_of_each_transition = np.where(np.diff(transitions)>1000)[0]
end_transition_indices = transitions[end_of_each_transition]
end_transition_indices = end_transition_indices[1:]


# valco timing
# valco_diff = np.diff(valco_pos)
# valco_movement = np.where(np.diff(valco_pos))[0]

# v1 = np.where(valco_pos[start_transition_indices]==1)[0]
# v2 = np.where(valco_pos[start_transition_indices]==2)[0]
# v3 = np.where(valco_pos[start_transition_indices]==3)[0]
# v4 = np.where(valco_pos[start_transition_indices]==4)[0]
# v5 = np.where(valco_pos[start_transition_indices]==5)[0]
# v6 = np.where(valco_pos[start_transition_indices]==6)[0]


# trim gdi
trim_amount = 600
trimmed_indices = [[i for i in range(index, index + trim_amount)] for index in end_transition_indices]
trimmed_indices = [item for sublist in trimmed_indices for item in sublist]

gdi = np.setdiff1d(gdi,trimmed_indices)


fig_dir = 'figures'
fig_num = 0

# Figures
if os.path.isdir(os.path.join(run_dir, fig_dir)) is False:
    os.makedirs(os.path.join(run_dir, fig_dir))

# remove old figures
figlist = make_file_list(os.path.join(run_dir, fig_dir), 'png')
[os.remove(os.path.join(fig_dir, fig)) for fig in figlist]

WIDTH = 8
HEIGHT = 5

fig_num += 1
figname = f'Fig{str(fig_num)}_example_transitions.png'
fig, ax = pplt.subplots(figsize=(WIDTH, HEIGHT), dpi=600, tight_layout=True)
ax.plot(time[440000:500000], dD[440000:500000])
ax.set_xlabel('Time (seconds since Jan 1, 1970)')
ax.set_ylabel('dD raw (permil)')
pplt.savefig(os.path.join(run_dir, fig_dir, figname))
pplt.close()

fig_num += 1
figname = f'Fig{str(fig_num)}_example_transition_ddDdT.png'
fig, ax = pplt.subplots(figsize=(WIDTH, HEIGHT), dpi=600, tight_layout=True)
ax.plot(time[440000:500000], ddD_dT_convolved[440000:500000])
ax.set_xlabel('Time (seconds since Jan 1, 1970)')
ax.set_ylabel('ddD/dT smoothed (permil / second)')
pplt.savefig(os.path.join(run_dir, fig_dir, figname))
pplt.close()

fig_num += 1
figname = f'Fig{str(fig_num)}_example_transition_ddDdT_zoom.png'
fig, ax = pplt.subplots(figsize=(WIDTH, HEIGHT), dpi=600, tight_layout=True)
ax.plot(time[start_transition_indices], ddD_dT_convolved[start_transition_indices], 'g.')
ax.plot(time[end_transition_indices], ddD_dT_convolved[end_transition_indices], 'r.')
ax.plot(time[461500:463000], ddD_dT_convolved[461500:463000])
ax.set_xlim([time[461500], time[463000]])
ax.set_xlabel('Time (seconds since Jan 1, 1970)')
ax.set_ylabel('ddD/dT smoothed (permil / second)')
pplt.savefig(os.path.join(run_dir, fig_dir, figname))
pplt.close()


# fig_num += 1
# figname = f'Fig{str(fig_num)}_Valco_to_Picarro_transit_time.png'
# fig, ax = pplt.subplots(figsize=(WIDTH, HEIGHT), dpi=600, tight_layout=True)
# # ax.plot(time[start_transition_indices[v1]], time[start_transition_indices[v1]] - time[valco_movement[v1]],'o', label='v1')
# ax.plot(time[start_transition_indices[v2]], time[start_transition_indices[v2]] - time[valco_movement[v2]],'o', label='v2')
# ax.plot(time[start_transition_indices[v3]], time[start_transition_indices[v3]] - time[valco_movement[v3]],'o', label='v3')
# ax.plot(time[start_transition_indices[v4]], time[start_transition_indices[v4]] - time[valco_movement[v4]],'o', label='v4')
# ax.plot(time[start_transition_indices[v5]], time[start_transition_indices[v5]] - time[valco_movement[v5]],'o', label='v5')
# # ax.plot(time[start_transition_indices[v6]], time[start_transition_indices[v6]] - time[valco_movement[v6]],'o', label='v6')
# # ax2 = ax.twinx()
# # ax2.plot(time, valco_pos)
# ax.set_xlabel('Time (seconds since Jan 1, 1970)')
# ax.set_ylabel('Valco to picarro travel time (seconds)')
# # ax2.set_xlabel('Valco position')
# ax.legend()
# pplt.savefig(os.path.join(run_dir, fig_dir, figname))
# pplt.close()


fig_num += 1
figname = f'Fig{str(fig_num)}_transition_time_vs_dD.png'
fig, ax = pplt.subplots(figsize=(WIDTH, HEIGHT), dpi=600, tight_layout=True)
ax.plot(dD[end_transition_indices] - dD[start_transition_indices], time[end_transition_indices] - time[start_transition_indices],'k.')
ax.set_title(f"{instrument['name']} - {dt.datetime.utcfromtimestamp(int(round(time[0]))).strftime('%Y-%m-%d')} to {dt.datetime.utcfromtimestamp(int(round(time[-1]))).strftime('%Y-%m-%d')}")
ax.set_ylabel('Reference water transition time (seconds)')
ax.set_xlabel('Reference water dD difference (permil)')
pplt.savefig(os.path.join(run_dir, fig_dir, figname))
pplt.close()


# fig,ax1 = pplt.subplots()
# ax1.plot(time,ddD_dT)
# ax2 = ax1.twinx()
# ax2.plot(time,valco_pos)
# pplt.show()



# t = time
# f = d18O
# dt = np.mean(np.diff(t))/60
# n = len(t)
# fhat = np.fft.fft(f, n)
# PSD = fhat * np.conjugate(fhat) / n
# freq = (1 / (dt * n)) * np.arange(n)
# L = np.arange(1, np.floor(n / 2), dtype='int')

# fig, axs = pplt.subplots(2, 1)
# pplt.sca(axs[0])
# pplt.plot(t, f)
# pplt.xlim(t[0], t[-1])

# pplt.sca(axs[1])
# pplt.plot(freq[L], PSD[L])
# pplt.xlim(freq[L[0]], freq[L[-1]])

# pplt.show()


