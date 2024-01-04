#!/usr/bin/env python3
"""
Library of functions used by the IsoLab picarro_* suite of python scripts.

As of version 1.3 I am trying to investigate how the IsoLab delta value calculations are sensitive to
H2O ppmv when they did not used to be, for phoenix at least. As such, I have added an 'i' to all variable
names that are associated with IsoLab's delta calculations to indicate they are isolab's. A 'p' on the end
of a delta or ratio indicates Picarro's calculation.

Version 1.4 adds a picarro_path.txt to save me time in moving the picarro_lib.py file around and dealing
with all the different directory structures on different computers.

Version 1.5 adds the ability to import data from an instrument that is not explicitly listed.

Version 1.6 changes min_H2O to 5000 from 10000

Version 1.7 has inj_peak, inj_quality, and vial_quality removed. These have been replaced by dictionaries in
picarro_inj.py and picarro_vial.py files. Moved calc_mode() from picarro_inj.py to here so it is available to
picarro_vial.py.
"""

__author__ = "Andy Schauer"
__email__ = "aschauer@uw.edu"
__last_modified__ = "2023-07-24"
__version__ = "1.7"
__copyright__ = "Copyright 2023, Andy Schauer"
__license__ = "Apache 2.0"
__acknowledgements__ = "M. Sliwinski, H. Lowes-Bicay, N. Brown"


# -------------------- imports --------------------
import numpy as np
import os
import re
import time as t


# -------------------- functions --------------------
def calc_mode(rd, rnd):
    """Calculate the mode of a raw dataset rd after having rounded it to 
    the nearest number of specified decimal places, rnd."""
    rd = rd[~np.isnan(rd)]
    d = [round(i, rnd) for i in rd]
    v,c = np.unique(d, return_counts=True)
    i = np.argmax(c)
    return v[i]


def get_path(desired_path):
    """Make your life easier with this section. These are the paths that seem to change depending on the computer we are working on."""
    picarro_path_file = os.path.join(os.getcwd(), 'picarro_path.txt')
    if os.path.isfile(picarro_path_file):
        # print(' :-) Using existing picarro path file for a warm and fuzzy experience. (-:')
        with open(picarro_path_file, 'r') as ppf:
            python_path, project_path, standards_path = ppf.readline().split(',')

    else:
        python_path_check = False
        project_path_check = False
        standards_path_check = False
        print(' )-: Picarro path file does not exist yet. :-(')
        print(" Let's make one... :-| ")
        while python_path_check is False:
            python_path = input(f'Enter the current path to the picarro python scripts. Perhaps it is {os.getcwd()}. ')
            if os.path.isdir(python_path):
                python_path_check = True
                if python_path[-1] != '/':
                    python_path += '/'
            else:
                print(f'oops, try typing that in again (you typed {python_path}): ')

        while project_path_check is False:
            project_path = input('Enter the current path to your projects: ')
            if os.path.isdir(project_path):
                project_path_check = True
                if project_path[-1] != '/':
                    project_path += '/'
            else:
                print(f'oops, try typing that in again (you typed {project_path}): ')

        while standards_path_check is False:
            standards_path = input('Enter the current path and filename to your reference materials file: ')
            if os.path.isfile(standards_path):
                standards_path_check = True
            else:
                print(f'oops, try typing that in again (you typed {standards_path}): ')

        with open(picarro_path_file, 'w') as ppf:
            ppf.write(f'{python_path},{project_path},{standards_path}')

    if desired_path == "project":
        return project_path
    elif desired_path == "python":
        return python_path
    elif desired_path == "standards":
        return standards_path
    else:
        unknown_path = input('Enter the path to your project: ')
        return unknown_path


def get_instrument():
    """ Get specific picarro instrument whose data is being processed and associated information.

        Reference Values - The below "ref_ratios" values are user defined reference values based
        on empirical data. They are calculated when an in-house reference water with known VSMOW
        values is being measured and believed to be memory free. Values are to four decimal places
        because 2 sigma was <0.0002. High resolution, injection level, or vial level data may be
        used provided we know that a particular standard is entering the cavity. See picarro_h5.py
        for the ratio calculation."""

    instrument_list = 'abel, desoto, mildred, phoenix, not_listed'
    name_recognized = False

    while name_recognized is False:
        entered_name = input(f"\nEnter the name of the instrument ({instrument_list}): ")
        if entered_name == 'abel':
            name_recognized = True
            instrument = {
                'name': 'Abel',
                'model': 'L2130i',
                'O17_flag': False}
            instrument['ref_ratios'] = {
                'rDHi': 0.1744,
                'r1816i': 1.7540,
                'notes': """Reference values are from Abel (an L2130i) 20220901 calibrated vial level data using KD.
                            np.mean(vial['peak3_offset']['mean'][kd['index']]/vial['peak2_offset']['mean'][kd['index']])"""}

        elif entered_name == 'desoto':
            name_recognized = True
            instrument = {
                'name': 'DeSoto',
                'model': 'L2120i',
                'O17_flag': False}

        elif entered_name == 'mildred':
            name_recognized = True
            instrument = {
                'name': 'Mildred',
                'model': 'L2140i',
                'O17_flag': True}

        elif entered_name == 'phoenix':
            name_recognized = True
            instrument = {
                'name': 'Phoenix',
                'model': 'L2140i',
                'O17_flag': True}
            instrument['ref_ratios'] = {
                'rDHi': 0.1509,
                'r1816i': 1.6954,
                'r1716i': 0.5854,
                'r1816i_1v2': 0.9623}

        elif entered_name == 'not_listed':
            name_recognized = True
            name = 'not_listed'
            valid_answer = False
            while not valid_answer:
                O17_flag = input('Does your instrument have O17 capability? (y or n) ')
                if O17_flag == 'y':
                    O17_flag = True
                    valid_answer = True
                elif O17_flag == 'n':
                    O17_flag = False
                    valid_answer = True
                else:
                    print('type y or n')
            instrument = {
                'name': name,
                'model': 'unknown',
                'O17_flag': O17_flag}

        else:
            print('\nInstrument not recognized.')

        if 'ref_ratios' not in instrument:
            instrument['ref_ratios'] = {
                'rDHi': 0.1500,
                'r1816i': 1.7000,
                    'r1716i': 0.5900,
                'r1816i_1v2': 0.9600}
            print(' ')
            print(' **** Using placeholder isotopic reference values that need to be updated to this specific instrument. ****')
            print(' ')


    return instrument


def make_file_list(directory, filetype):
    """Create and return a list of files contained within a directory
    of file type."""
    filelist = []
    initial_list = os.listdir(directory)
    for file in initial_list:
        if re.search(filetype, file):
            filelist.append(file)
    return filelist


def read_file(file_to_import, delim=None, header_row=1):
    """Read in a delimited text file containing a single header row
    followed by data and return those headers as a list and the data
    as a dictionary."""
    with open(file_to_import, 'r') as f:
        if header_row > 1:
            f.readline()
        headers = f.readline().split(delim)

        # remove unwanted characters from headers using a regular expression
        p = re.compile(r'[./\s()]')  # list of characters to match
        for ii in range(len(headers)):
            m = p.findall(headers[ii])
            for em in m:
                headers[ii] = headers[ii].replace(em, '')

        data = {}
        for h in headers:
            data[h] = []

        for line in f:
            row = line.split(delim)
            if delim is None:
                if len(row) < len(headers):
                    row.append(0)
            # populate dictionary with all data in all rows
            for h, v in zip(headers, row):
                if v == '':
                    v = None
                data[h].append(v)

    return headers, data


# -------------------- python scripts --------------------
python_dir = get_path("python")
python_scripts = {'picarro_lib.py': '', 'picarro_h5.py': '', 'picarro_inj.py': '', 'picarro_vial.py': ''}
python_scripts = {key: (t.strftime('%Y-%m-%d %H:%M:%S', t.localtime(os.path.getmtime(f'{python_dir}{key}')))) for key, value in python_scripts.items()}
