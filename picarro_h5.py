#!/usr/bin/env python3
"""
Read in multiple hourly h5 data files from any Picarro instrument. The h5 files (short for HDF5, a type of
Hierarchical Data Format) contain data at about 1 Hz frequency. This script simply reads in a date
range of h5 files, allows the user to trim the ends and then saves the dataset as a new hdf5 file. While
this script was written with Picarro's water isotope instruments in mind, in principle, it should work for
any Picarro CRDS instrument.
"""

__author__ = "Andy Schauer"
__copyright__ = "Copyright 2022, Andy Schauer"
__license__ = "Apache 2.0"
__version__ = "1.0"
__email__ = "aschauer@uw.edu"


# -------------------- imports --------------------
import argparse
from datetime import datetime
import h5py
import numpy as np
import matplotlib.pyplot as pplt
import os
import re
import time as t


# -------------------- functions --------------------
def fig_on_key(event):
    """Gets the start and stop data point indices while looking at a figure of H2O and dD."""
    global start, stop
    if event.key == '1':
        start = round(event.xdata)
        print(f'start chosen as {start}')
    elif event.key == '2':
        stop = round(event.xdata)
        print(f'stop chosen as {stop}')


def make_h5_file_list(directory):
    """Create and return a list of files contained within a directory."""
    filelist = []
    initial_list = os.listdir(directory)
    for file in initial_list:
        if re.search('h5', file):
            filelist.append(file)
    return filelist


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
this function with your own instrument(s)."""

instrument_list = 'abel, desoto, mildred, phoenix'
name_recognized = False

while name_recognized is False:
    if (instrument_name is None) or (instrument_name not in instrument_list):
        entered_name = input(f"\nEnter the name of the instrument ({instrument_list}): ")
    else:
        entered_name = instrument_name

    if entered_name == 'abel':
        instrument = {
            'name': 'Abel',
            'model': 'L2130i',
            'O17_flag': False}
        print(' ')
        print(' ** Using placeholder isotopic reference values that need to be updated to this specific instrument. **')
        print(' ')
        t.sleep(0.5)
        rDH_ref = 0.1509
        r1816_ref = 1.6954
        r1716_ref = 0.5854
        r1816b_ref = 0.9623

        name_recognized = True

    elif entered_name == 'desoto':
        instrument = {
            'name': 'DeSoto',
            'model': 'L2120i',
            'O17_flag': False}
        name_recognized = True

    elif entered_name == 'mildred':
        instrument = {
            'name': 'Mildred',
            'model': 'L2140i',
            'O17_flag': True}
        print(' ')
        print(' ** Using placeholder isotopic reference values that need to be updated to this specific instrument. **')
        print(' ')
        t.sleep(0.5)
        rDH_ref = 0.1509
        r1816_ref = 1.6954
        r1716_ref = 0.5854
        r1816b_ref = 0.9623

        name_recognized = True

    elif entered_name == 'phoenix':
        instrument = {
            'name': 'Phoenix',
            'model': 'L2140i',
            'O17_flag': True}

        rDH_ref = 0.1509
        r1816_ref = 1.6954
        r1716_ref = 0.5854
        r1816b_ref = 0.9623

        name_recognized = True

    else:
        print('\nInstrument not recognized.')


# -------------------- directory setup --------------------
"""Make your life easier with this section. I think the only common path segment we all have is the h5_dir, but the
rest will be completely different depending on how you organize yourself. I have assumed the simplest, but not my
preferred, approach which is to have the present script in a project directory that also has "runs" and "h5"
directories. I am also assuming you will comment out or remove the current "project_dir" line in favor of
a hard-coded line like the example shown below it."""
project_dir = os.getcwd()
# project_dir = f"/path/to/your/project/"
run_dir = os.path.join(project_dir, 'runs/')
h5_dir = os.path.join(project_dir, 'h5/Picarro/G2000/Log/DataLogger/DataLog_Private/')


# -------------------- make h5 file list --------------------
"""This is a list of all h5 files that exist for the instrument of interest."""
all_h5_file_list = make_h5_file_list(h5_dir)
all_h5_file_list.sort()


# -------------------- ask user for start and stop dates --------------------
if start_date is None:
    start_date = input(f"Enter the start date as yyyymmdd (leave empty for oldest file {all_h5_file_list[0]}): ")

if stop_date is None:
    stop_date = input(f"Enter the stop date as yyyymmdd (leave empty for most recent file {all_h5_file_list[-1]}): ")

start_time = t.time()


# -------------------- make file list based on those start and stop dates --------------------
if start_date == '':
    start_date_index = None
else:
    start_date_list = [i for i in all_h5_file_list if start_date in i]
    start_date_index = all_h5_file_list.index(start_date_list[0])

if stop_date == '':
    stop_date_index = None
else:
    stop_date_list = [i for i in all_h5_file_list if str(int(stop_date) + 1) in i]
    stop_date_index = all_h5_file_list.index(stop_date_list[0])

file_list = all_h5_file_list[start_date_index:stop_date_index]


# -------------------- open the first h5 file in the list and get the headers and data types --------------------
h5_file = h5py.File(f'{h5_dir}{file_list[0]}')
headers = list(h5_file['results'].dtype.names)
h5_dtype = h5_file['results'].dtype


# -------------------- create an empty numpy array for every header --------------------
preallocate_size = 4500 * len(file_list)  # The length of a one hour h5 file, if the spectral duration were 0.8 seconds, is 4500.
data = np.asarray(np.zeros(preallocate_size), dtype=h5_dtype)


# -------------------- read in data --------------------
total_size = 0
start = 0

for file in file_list:
    h5_file = h5py.File(f'{h5_dir}{file}')
    h5_data = h5_file['results']
    if len(h5_data.dtype) == len(headers):
        print(f"{file}    =>    {len(h5_data.dtype)} data fields.")
        total_size += h5_data.shape[0]
        stop = start + h5_data.shape[0]
        data[start:stop] = h5_data[:]
        start = stop
    else:
        print(f"{file}    =>    {len(h5_data.dtype)} data fields.    ** Can't import different sized files. Super sorry. ** ")
        t.sleep(0.1)

print("--- %s seconds ---" % (t.time() - start_time))


# -------------------- Choose the exact start and stop data points by using the figures --------------------
data_index = [i for i in range(0, len(data['H2O']))]

WIDTH = 10
HEIGHT = 8

fig = pplt.subplots(figsize=(WIDTH, HEIGHT))
ax_top = pplt.subplot(211)
ax_top.set_title("""Use either figure to pick where your dataset should start and stop.
                 Hover over where you want the dataset to start and press 1. Then hover over where it should stop and press 2. Close the figure when you are done.""", wrap=True)
ax_top.set_xlabel('h5_data point index')
ax_top.set_ylabel('H2O (ppm)')
ax_top.plot(data_index, data['H2O'])
ax_bottom = pplt.subplot(212)
ax_bottom.set_xlabel('h5_data point index')
ax_bottom.set_ylabel('dD (permil)')
ax_bottom.plot(data_index, data['Delta_D_H'])

cid = fig[0].canvas.mpl_connect('key_press_event', fig_on_key)

pplt.show()


# -------------------- Populate numpy arrays --------------------
"""Numpy arrays were preallocated above for each header. Populate those arrays with data within the start and stop range."""
for header in headers:
    globals()[header] = data[header][start:stop]


# -------------------- delta value calculation  --------------------
"""Picarro provides delta values within the h5 files (e.g. Delta_18_16). However, it is
    instructional to understand the calculation Picarro is making. This section attempts
    to replicate Picarro's delta calculations and then makes sure these new delta values
    make their way into the data set for hdf5 export.

    Reference Values - The below "_ref" values are user defined reference values based
    on empirical data. These values use Kona Deep drinking water standard (KD) from
    Phoenix (the prototype and original L2140i) picarro_vial_calibrate.py, using KD
    data, calculate ratio values using strength offset data as below under "Raw delta
    calculations". Values are to four decimal places because 2 sigma was <0.0002. I
    chose KD because of its similarity with VSMOW. I chose calibrated vial level
    data because this is an effort to make delta values as close to reality as possible.

    Raw delta calculations - John Hoffnagle's calculation - based on strengths and uses laser 2 minor
    masses and laser 1 major mass. The _offset values have been corrected
    for water vapor concentration (e.g. str3_offset). The non offset values
    have not been corrected for water vapor concentration (e.g. strength3)."""

if instrument['O17_flag']:
    hdf5_additions = ['rDH', 'dD', 'r1816', 'd18O', 'r1716', 'd17O', 'r1816b', 'd18Ob']

    if 'rDH_ref' not in globals():  # if reference values are not defined above, then here are some place holders
        rDH_ref = 0.1509
        r1816_ref = 1.6954
        r1716_ref = 0.5854
        r1816b_ref = 0.9623

    rDH = str3_offset / str2_offset
    r1816 = str1_offset / str2_offset
    r1716 = str13_offset / str2_offset
    r1816b = str11_offset / str2_offset

    dD = (rDH / rDH_ref - 1) * 1000
    d18O = (r1816 / r1816_ref - 1) * 1000
    d17O = (r1716 / r1716_ref - 1) * 1000
    d18Ob = (r1816b / r1816b_ref - 1) * 1000

else:
    hdf5_additions = ['rDH', 'dD', 'r1816', 'd18O']

    """Reference values are from Abel (an L2130i) 20220901 calibrated vial level data using KD."""

    if 'rDH_ref' not in globals():  # if reference values are not defined above, then here are some place holders
        rDH_ref = 0.1744  # np.mean(vial['peak3_offset']['mean'][kd['index']]/vial['peak2_offset']['mean'][kd['index']])
        r1816_ref = 1.7540  # np.mean(vial['peak1_offset']['mean'][kd['index']]/vial['peak2_offset']['mean'][kd['index']])

    rDH = peak3_offset / peak2_offset
    r1816 = peak1_offset / peak2_offset

    dD = (rDH / rDH_ref - 1) * 1000
    d18O = (r1816 / r1816_ref - 1) * 1000

headers = headers + hdf5_additions


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

run_name = f"{start_run}-{end_run}{run_keyword}"

if os.path.isfile(os.path.join(run_dir, run_name)) is False:
    with h5py.File(f'{run_dir}{run_name}.hdf5', 'w') as run_h5_file:
        for header in headers:
            run_h5_file.create_dataset(header, data=eval(header))

else:
    print(f'Looks like the file {run_name} already exists in {run_dir}.')

print(' ')
print(f"{run_name}.hdf5 has been saved to {run_dir}")
