#!/usr/bin/env python3
"""
Read in multiple hourly h5 data files from any Picarro instrument. The h5 files (short for HDF5, a type of
Hierarchical Data Format) contain data at about 1 Hz frequency. This script simply reads in a date
range of h5 files, allows the user to trim the ends and then saves the dataset as a new hdf5 file. While
this script was written with Picarro's water isotope instruments in mind, in principle, it should work for
any Picarro CRDS instrument.
"""

__author__ = "Andy Schauer"
__email__ = "aschauer@uw.edu"
__last_modified__ = "2026.01.18"
__copyright__ = "Copyright 2026, Andy Schauer"
__license__ = "Apache 2.0"


# -------------------- imports --------------------
import argparse
from datetime import datetime
import h5py
import matplotlib.pyplot as pplt
from matplotlib.widgets import SpanSelector, Button
import numpy as np
import os
from picarro_lib import *
import time as t



# -------------------- parse the three optional arguments --------------------
"""These arguments will be obtained either by passing them while calling the present script
or by adding them as arguments."""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--instrument", type=str)
parser.add_argument("-st", "--startdate", type=str)
parser.add_argument("-sp", "--stopdate", type=str)

args = parser.parse_args()
instrument_name = args.instrument
start_date = args.startdate
stop_date = args.stopdate


# -------------------- get instrument information --------------------
""" Get specific picarro instrument whose data is being processed as well as some associated information. Populate
this function with your own instrument(s). The function get_instrument() is located in picarro_lib.py."""
instrument = get_instrument()


# -------------------- paths --------------------
if instrument['name'] == 'not_listed':
    project_dir = input('Enter the path to your h5 directory: ')
    run_dir = project_dir[:-3]
    h5_dir = os.path.join(project_dir, 'Picarro/G2000/Log/DataLogger/DataLog_Private/')
else:
    project_dir = f"{get_path('project')}{instrument['name'].lower()}/"
    run_dir = os.path.join(project_dir, 'runs/')
    h5_dir = os.path.join(project_dir, 'h5/Picarro/G2000/Log/DataLogger/DataLog_Private/')


# -------------------- make h5 file list --------------------
"""This is a list of all h5 files that exist for the instrument of interest."""
all_h5_file_list = make_file_list(h5_dir, '.h5')
all_h5_file_list.sort()


# -------------------- get the first file to process --------------------
start_date_found = False
while start_date_found is False:
    if start_date is None:
        start_date = input(f"Enter the start date as yyyymmdd (leave empty for oldest file {all_h5_file_list[0]}): ")
    if start_date == '':
        start_date_index = None
        start_date_found = True
    else:
        start_date_list = [i for i in all_h5_file_list if start_date in i]
        if not start_date_list:
            print(f"The start date you entered ({start_date}) is not in the h5 file list.")
            start_date = input(f"Try again. Enter the start date as yyyymmdd: ")
        else:
            start_date_found = True

        start_date_index = all_h5_file_list.index(start_date_list[0])


# -------------------- get the last file to process --------------------
stop_date_found = False
while stop_date_found is False:
    if stop_date is None:
        stop_date = input(f"Enter the stop date as yyyymmdd (leave empty for most recent file {all_h5_file_list[-1]}): ")
    if stop_date == '':
        stop_date_index = None
        stop_date_found = True
    else:
        stop_date_list = [i for i in all_h5_file_list if str(int(stop_date)) in i]
        if not stop_date_list:
            print(f"The stop date you entered ({stop_date}) is not in the h5 file list.")
            stop_date = input(f"Try again. Enter the stop date as yyyymmdd: ")
        else:
            stop_date_found = True

        stop_date_index = all_h5_file_list.index(stop_date_list[-1])


# -------------------- make file list based on those start and stop dates --------------------
file_list = all_h5_file_list[start_date_index:stop_date_index]


# -------------------- open the first h5 file in the list and get the headers and data types --------------------
h5_file = h5py.File(f'{h5_dir}{file_list[0]}')
headers = list(h5_file['results'].dtype.names)
h5_dtype = h5_file['results'].dtype


# -------------------- create an empty numpy array for every header --------------------

preallocate_size = instrument['h5_shape'] * len(file_list)  # The length of a one hour h5 file, if the spectral duration were 0.8 seconds, is 4500.
data = np.asarray(np.zeros(preallocate_size), dtype=h5_dtype)


# -------------------- read in data --------------------
total_size = 0
start = 0

for file in file_list:
    h5_file = h5py.File(f'{h5_dir}{file}')
    h5_data = h5_file['results']
    if len(h5_data.dtype) == len(headers):
        print(f"{file}    =>    {len(h5_data.dtype)} data fields. Shape = {h5_data.shape}")
        total_size += h5_data.shape[0]
        stop = start + h5_data.shape[0]
        data[start:stop] = h5_data[:]
        start = stop
    else:
        print(f"{file}    =>    {len(h5_data.dtype)} data fields.    ** Can't import different sized files. Super sorry. ** ")
        t.sleep(0.1)



# -------------------- Choose the exact start and stop data indices for the run --------------------
def _decimate(x, y, max_points=5000):
    n = len(x)
    if n <= max_points:
        return np.asarray(x), np.asarray(y)
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return np.asarray(x)[idx], np.asarray(y)[idx]

def _nearest_index(x_all, x_val):
    x_all = np.asarray(x_all)
    i = int(np.searchsorted(x_all, x_val))
    if i <= 0:
        return 0
    if i >= len(x_all):
        return len(x_all) - 1
    return i if abs(x_all[i] - x_val) < abs(x_all[i-1] - x_val) else i-1

def _coarse_window_two_axes(x, y1, y2, title):
    xo, y1o = _decimate(x, y1)
    _,  y2o = _decimate(x, y2)

    sel = {"xmin": float(xo[0]), "xmax": float(xo[-1])}

    def onselect(xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        sel["xmin"], sel["xmax"] = float(xmin), float(xmax)

    fig, (ax1, ax2) = pplt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(xo, y1o, rasterized=True)
    ax1.set_ylabel('H2O (ppm)')
    ax2.plot(xo, y2o, rasterized=True)
    ax2.set_ylabel('dD (‰)')
    ax2.set_xlabel('h5_data point index')
    fig.suptitle(f"{title}: drag to bracket region; click 'Done' or close this window when finished.")

    pplt.subplots_adjust(bottom=0.18)
    done_ax = fig.add_axes([0.75, 0.05, 0.13, 0.07])
    done_btn = Button(done_ax, 'Done')

    def finish(event=None):
        pplt.close(fig)

    done_btn.on_clicked(finish)
    span1 = SpanSelector(ax1, onselect, direction='horizontal', useblit=True, interactive=True,
                         props=dict(alpha=0.25, facecolor='orange'))
    span2 = SpanSelector(ax2, onselect, direction='horizontal', useblit=True, interactive=True,
                         props=dict(alpha=0.25, facecolor='orange'))
    pplt.show()

    xmin, xmax = sel["xmin"], sel["xmax"]
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        mid = float(xo[len(xo)//2])
        span = (xo[-1] - xo[0]) * 0.02
        xmin, xmax = mid - span, mid + span
    return xmin, xmax

def _fine_pick(x_all, y1, y2, xmin, xmax, title):
    fig, (ax1, ax2) = pplt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(x_all, y1)
    ax1.set_ylabel('H2O (ppm)')
    ax2.plot(x_all, y2)
    ax2.set_ylabel('dD (‰)')
    ax2.set_xlabel('h5_data point index')
    for a in (ax1, ax2):
        a.set_xlim(xmin, xmax)

    pplt.subplots_adjust(bottom=0.18)
    pick_ax = fig.add_axes([0.12, 0.05, 0.18, 0.07])
    done_ax = fig.add_axes([0.75, 0.05, 0.13, 0.07])
    pick_btn  = Button(pick_ax,  'Pick point')
    done_btn  = Button(done_ax,  'Done')
    picked = {"idx": None}
    vlines = []

    def _draw_vline(xv):
        if not vlines:
            vlines.extend([ax1.axvline(xv, linestyle='--', linewidth=1.5),
                           ax2.axvline(xv, linestyle='--', linewidth=1.5)])
        else:
            for vl in vlines:
                vl.set_xdata([xv, xv])
        fig.canvas.draw_idle()

    def do_pick(event):
        pts = pplt.ginput(1, timeout=0)
        if not pts:
            return
        xv = pts[0][0]
        idx = _nearest_index(x_all, xv)
        picked["idx"] = int(idx)
        _draw_vline(x_all[picked["idx"]])

    def finish(event):
        pplt.close(fig)

    pick_btn.on_clicked(do_pick)
    done_btn.on_clicked(finish)

    fig.suptitle(f"{title}: zoom/pan freely; when ready, click 'Pick point', click on the figure where the point is located; then click Done.")
    pplt.show()

    if picked["idx"] is None:
        picked["idx"] = _nearest_index(x_all, ax1.get_xlim()[0])

    return int(picked["idx"])


def pick_start_stop(data_index, h2o, dD, *, overview_max_points=5000):
    x = np.asarray(data_index)
    y1 = np.asarray(h2o)
    y2 = np.asarray(dD)

    # START
    xmin, xmax = _coarse_window_two_axes(x, y1, y2, title="DRAG TO CHOOSE START REGION")
    start_idx = _fine_pick(x, y1, y2, xmin, xmax, title="CHOOSE START POINT")

    # STOP
    xmin, xmax = _coarse_window_two_axes(x, y1, y2, title="DRAG TO CHOOSE STOP REGION")
    stop_idx = _fine_pick(x, y1, y2, xmin, xmax, title="CHOOSE STOP POINT")

    if start_idx > stop_idx:
        start_idx, stop_idx = stop_idx, start_idx

    return start_idx, stop_idx


data_index = np.arange(len(data['H2O']))
start_idx, stop_idx = pick_start_stop(
    data_index=data_index,
    h2o=data['H2O'],
    dD=data['Delta_D_H'],
    overview_max_points=8000  # optional; bump if you want more detail in the overview
)
print(f"\nSTART: {start_idx}, STOP: {stop_idx}")


# -------------------- Populate numpy arrays --------------------
"""Numpy arrays were preallocated above for each header. Populate those arrays with data within the start and stop range."""
for header in headers:
    globals()[header] = data[header][start_idx:stop_idx]


# -------------------- delta value calculation  --------------------
"""Picarro provides delta values within the h5 files (e.g. Delta_18_16). However, it is
    instructional to understand the calculation Picarro is making. This section attempts
    to replicate Picarro's delta calculations and then makes sure these new delta values
    make their way into the data set for hdf5 export.

    Raw delta calculations - John Hoffnagle's calculation - based on strengths and uses laser 2 minor
    masses and laser 1 major mass. The _offset values have been corrected
    for water vapor concentration (e.g. str3_offset). The non offset values
    have not been corrected for water vapor concentration (e.g. strength3).

    As of 2023-01-25, we have discovered an issue with this strategy. Even though we have been using
    these below calculations since the development of the 2140 (c. 2013), we have seen evidence that
    the water vapor correction applied to the strength offset values may have changed or in some way
    they now appear sensitive to H2O ppmv. We continue to look into this issue and for now are formally
    separating the 'Picarro' delta values (e.g., d18Op) from the IsoLab delta values (e.g., d18Oi)."""

hdf5_additions = ['dDp', 'rDHi', 'dDi', 'd18Op', 'r1816i', 'd18Oi']
if instrument['O17_flag']:
    hdf5_additions.extend(['d17Op', 'r1716i', 'd17Oi', 'r1816i_1v2', 'd18Oi_1v2'])

    rDHi = str3_offset / str2_offset
    r1816i = str1_offset / str2_offset
    r1716i = str13_offset / str2_offset
    r1816i_1v2 = str11_offset / str2_offset

    d17Oi = (r1716i / instrument['ref_ratios']['r1716i'] - 1) * 1000
    d18Oi_1v2 = (r1816i_1v2 / instrument['ref_ratios']['r1816i_1v2'] - 1) * 1000

    d17Op = Delta_17_16

else:
    rDHi = peak3_offset / peak2_offset
    r1816i = peak1_offset / peak2_offset

dDi = (rDHi / instrument['ref_ratios']['rDHi'] - 1) * 1000
d18Oi = (r1816i / instrument['ref_ratios']['r1816i'] - 1) * 1000

headers = headers + hdf5_additions

dDp = Delta_D_H
d18Op = Delta_18_16


# -------------------- save h5_data set as single hdf5 file --------------------
start_run = datetime.fromtimestamp(time[0]).strftime('%Y%m%d')
end_run = datetime.fromtimestamp(time[-1]).strftime('%Y%m%d')
run_keyword = input("Enter a descriptive keyword for this h5_data set if you like: ")
if run_keyword == '':
    pass
else:
    run_keyword = f"_{run_keyword}"

run_dir += f"{start_run}{run_keyword}/"
if os.path.isdir(run_dir) is False:
    os.mkdir(run_dir)

if os.path.isdir(os.path.join(run_dir, 'archive')) is False:
    os.mkdir(os.path.join(run_dir, 'archive'))


run_name = f"{start_run}-{end_run}{run_keyword}"

if os.path.isfile(os.path.join(run_dir, run_name)) is False:
    with h5py.File(f'{run_dir}{run_name}.hdf5', 'w') as run_h5_file:
        for header in headers:
            run_h5_file.create_dataset(header, data=eval(header))

else:
    print(f'Looks like the file {run_name} already exists in {run_dir}.')

print(' ')
print(f"{run_name}.hdf5 has been saved to {run_dir}")
