#!/usr/bin/env python3
"""
Library of functions used by the IsoLab picarro_* suite of python scripts.
"""

__author__ = "Andy Schauer"
__copyright__ = "Copyright 2022, Andy Schauer"
__license__ = "Apache 2.0"
__version__ = "1.1"
__email__ = "aschauer@uw.edu"


# -------------------- imports --------------------
import os
import re


def get_instrument():
    """ Get specific picarro instrument whose data is being processed and associated information.

        Reference Values - The below "ref_ratios" values are user defined reference values based
        on empirical data. They are calculated when an in-house reference water with known VSMOW
        values is being measured and believed to be memory free. Values are to four decimal places
        because 2 sigma was <0.0002. High resolution, injection level, or vial level data may be
        used provided we know that a particular standard is entering the cavity. See picarro_h5.py
        for the ratio calculation."""

    instrument_list = 'abel, desoto, mildred, phoenix'
    name_recognized = False

    while name_recognized is False:
        entered_name = input(f"\nEnter the name of the instrument ({instrument_list}): ")
        if entered_name == 'abel':
            name_recognized = True
            instrument = {
                'name': 'Abel',
                'model': 'L2130i',
                'O17_flag': False}
            ref_ratios = {
                'rDH': 0.1744,
                'r1816': 1.7540,
                'notes': """Reference values are from Abel (an L2130i) 20220901 calibrated vial level data using KD.
                            np.mean(vial['peak3_offset']['mean'][kd['index']]/vial['peak2_offset']['mean'][kd['index']])"""}
            inj_peak = {
                'h2o_detection_limit': 5000,
                'trim_from_start': 35,
                'trim_from_end': 50}
            inj_quality = {
                'max_H2O_std': 1000,
                'max_d18O_std': 1.0,
                'max_dD_std': 2.0}
            vial_quality = {
                'max_H2O_std': 700,
                'max_d18O_std': 0.33,
                'max_dD_std': 1.33}

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
            ref_ratios = {
                'rDH': 0.1509,
                'r1816': 1.6954,
                'r1716': 0.5854,
                'r1816b': 0.9623}
            inj_peak = {
                'H2O_detection_limit': 5000,
                'trim_from_start': 35,
                'trim_from_end': 50}
            inj_quality = {
                'max_H2O_std': 1500,
                'max_d18O_std': 1.0,
                'max_d17O_std': 0.5,
                'max_dD_std': 2.0}
            vial_quality = {
                'max_H2O_std': 400,
                'max_d18O_std': 0.2,
                'max_dD_std': 2.00}

        else:
            print('\nInstrument not recognized.')

        if 'ref' not in locals():
            ref_ratios = {
                'rDH': 0.1500,
                'r1816': 1.7000,
                'r1716': 0.5900,
                'r1816b': 0.9600}
            print(' ')
            print(' **** Using placeholder isotopic reference values that need to be updated to this specific instrument. ****')
            print(' ')

        if 'inj_peak' not in locals():
            inj_peak = {
                'H2O_detection_limit': 5000,
                'trim_from_start': 35,
                'trim_from_end': 50}
            print(' ')
            print(' **** Using placeholder injection peak detection values that may need to be updated to this specific instrument. ****')
            print(' ')

        if 'inj_quality' not in locals():
            inj_quality = {
                'max_H2O_std': 1000,
                'max_d18O_std': 1.0,
                'max_dD_std': 2.0}
            print(' ')
            print(' **** Using placeholder injection quality values that may need to be updated to this specific instrument. ****')
            print(' ')

        # These additions to inj_quality are reasonable for all instruments
        inj_quality['max_CAVITYPRESSURE_std'] = 0.056
        inj_quality['min_H2O'] = 10000

        if 'vial_quality' not in locals():
            vial_quality = {
                'max_H2O_std': 500,
                'max_d18O_std': 0.1,
                'max_dD_std': 1.00}

    return instrument, ref_ratios, inj_peak, inj_quality, vial_quality


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
