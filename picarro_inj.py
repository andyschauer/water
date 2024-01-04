#!/usr/bin/env python3
"""
This script reads in the hdf5 file created by picarro_h5.py if the data set exclusively contains discrete samples (injections).
This script will find each injection and reduce the h5 level data (~ 1 Hz data) to injection level summaries. A tray
description file is required. This script effectively creates Picarro's Coordinator output, albeit in json format
and with all of the data fields present that are in the original h5 files.

Version 2.0 has a different strategy employed to find peaks. Previous versions found the tops of peaks using H2O, dH2O_dT, and dH2O_dT2.
This version finds to troughs using the same three parameters. This came from an attempt to be able to account for failed injections and
the desire to show the complete injection from start to finish. Furthermore, I have made an effort to have measured thresholds rather than
static values set by me. They can still be changed to some custom value but the initial thresholds are measured. These are stored in the
peak detection settings file and imported as pds.

Version 2.1 has project and tray removed from tray description file
"""


__author__ = "Andy Schauer"
__email__ = "aschauer@uw.edu"
__last_modified__ = "2023-12-30"
__version__ = "2.1"
__copyright__ = "Copyright 2023, Andy Schauer"
__license__ = "Apache 2.0"
__acknowledgements__ = "M. Sliwinski, H. Lowes-Bicay, N. Brown"


# -------------------- imports --------------------
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN  # , INLINE
import datetime as dt
import h5py
import json
import matplotlib.pyplot as pplt  # for interactive mode
from natsort import natsorted
import numpy as np
import os
from picarro_lib import *
from scipy.signal import find_peaks
import shutil
import sys
import time as t
import warnings
import webbrowser


# -------------------- functions ------------------------------------

def get_peak_detection_settings():
    # Peak detection settings may be customized depending on the instrument or run. If you had odd backgrounds or otherwise a non-optimal
    #    run, you may need to adjust the peak detection settings in order to salvage your data.

    # The H2O concentration distribution has two modes, one at the background level in between injections and a second at the peak height level.
    max_H2O_background = 1000
    lower_H2O_mode = calc_mode(H2O[H2O < max_H2O_background], -1)
    upper_H2O_mode = calc_mode(H2O[H2O > max_H2O_background], -1)
    default_pds = {"kernel_size": 5,
           "dH2O_dT2": 50,
           "max_H2O_background": max_H2O_background,
           "lower_H2O_mode": int(lower_H2O_mode),
           "upper_H2O_mode": int(upper_H2O_mode),
           "trim_start": 10,
           "trim_end": 10,
           "trough_diff": 100}

    # Check for the existence of a settings file to grab existing peak detection settings
    pds_file = os.path.join(run_dir, 'peak_detection_settings.json')
    if os.path.isfile(pds_file):
        with open(pds_file, 'r') as f:
            existing_pds = json.load(f)
        if (set(existing_pds.keys()) == set(default_pds.keys())) is False:
            print('    Archiving existing peak detection settings file and using new default settings.')
            shutil.copy2(pds_file, os.path.join(archive_dir, f"peak_detection_settings_ARCHIVE_{int(os.path.getmtime(os.path.join(run_dir,'peak_detection_settings.json')))}.json"))
            pds = default_pds
        else:
            print('    Using existing peak detection settings.')
            pds = existing_pds
    else:
        print('    Using default peak detection settings.')
        pds = default_pds
    with open(os.path.join(run_dir, 'peak_detection_settings.json'), 'w', encoding='utf-8') as f:
        json.dump(pds, f, ensure_ascii=False, indent=4)

    return pds


def get_quality_control_parameters():
    # Quality control injection parameters may be customized depending on the instrument or run. If you had odd backgrounds or otherwise a non-optimal
    #    run, you may need to adjust the injection quality control parameters in order to salvage your data.
    default_qcp = {'max_H2O_std': round(calc_mode(inj['H2O']['std'], -1) * 4.4, 0),
           'max_d18O_std': round(calc_mode(inj['d18O']['std'], 1) * 4.4, 2),
           'max_dD_std': round(calc_mode(inj['dD']['std'], 1) * 4.4, 1),
           'max_CAVITYPRESSURE_std': 0.056,
           'min_H2O': 5000}
    if instrument['O17_flag']:
        default_qcp['max_d17O_std'] = round(calc_mode(inj['d17O']['std'], 1) * 4.4, 2)

    # Check for the existence of a settings file to grab existing peak detection settings
    qcp_file = os.path.join(run_dir, 'quality_control_parameters_inj.json')
    if os.path.isfile(qcp_file):
        with open(qcp_file, 'r') as f:
            existing_qcp = json.load(f)
        if set(existing_qcp.keys()) == set(default_qcp.keys()) is False:
            print('    Archiving existing quality control parameters file and using new default parameters.')
            shutil.copy2(qcp_file, os.path.join(archive_dir, f"quality_control_parameters_inj_ARCHIVE_{int(os.path.getmtime(os.path.join(run_dir,'quality_control_parameters_inj.json')))}.json"))
            qcp = default_qcp
        else:
            print('    Using existing quality control parameters.')
            qcp = existing_qcp
    else:
        print('    Using default quality control parameters.')
        qcp = default_qcp
    with open(os.path.join(run_dir, 'quality_control_parameters_inj.json'), 'w', encoding='utf-8') as f:
        json.dump(qcp, f, ensure_ascii=False, indent=4)

    return qcp


def plot_this(data):
    pplt.plot(data, '.')
    pplt.hlines([np.mean(data)-np.std(data)*4.4, np.mean(data)+np.std(data)*4.4], xmin=0, xmax=len(data))
    pplt.show()


def plot_H2O():
    fig, ax = pplt.subplots()
    [ax.plot(range(i-i,j-i), H2O[range(i,j)], '-') for i,j in zip(peak['start'], peak['end'])]
    pplt.show()


def plot_residual(data):
    fig, ax = pplt.subplots()
    [ax.plot(range(i-i,j-i), data[range(i,j)]-np.mean(data[range(i,j)]), '.') for i,j in zip(peak['top_start'], peak['top_end'])]
    pplt.show()
    # [pplt.plot(data[j:k]-np.mean(data[j:k]), '.') for j, k in zip(peak['top_start'], peak['top_end'])]
    # pplt.hlines([np.std(dD_residual)*4.4, -np.std(dD_residual)*4.4], xmin=0, xmax=350)


def screen_outliers(data, threshold):
    m = np.mean(data)
    i = np.where((data > m-threshold) & (data < m+threshold))
    return i


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

np.seterr(all='raise')


# -------------------- tray description file --------------------
identified_file = 0
while identified_file == 0:
    tray_descriptions = make_file_list(os.path.join(project_dir, 'TrayDescriptions/'), 'TrayDescription.csv')
    tray_description_file_list = [i for i in tray_descriptions if hdf5_file[0:8] in i[0:8]]
    if tray_description_file_list:
        if len(tray_description_file_list) > 1:
            print(f'\n ** More than one tray description file starting with {hdf5_file[0:8]} was found.')
            [print(f'    {i}') for i in tray_description_file_list]
            tray_description_file = input('Enter the tray description file name you want to use: ')
            if tray_description_file in tray_description_file_list:
                identified_file = 1
                print('    Tray description file found.')
            else:
                print('The file you entered is not in the list. Try again.')
        else:
            print('    Tray description file found.')
            identified_file = 1
            tray_description_file = tray_description_file_list[0]
        shutil.copy2(os.path.join(project_dir, 'TrayDescriptions/', tray_description_file), os.path.join(project_dir, run_dir, tray_description_file))

    else:
        print('\n ** Tray description file not found. **')
        print(f"        - Make sure a file with called '{hdf5_file[0:8]}_[your-project]_TrayDescripion.csv' is in {instrument['name']}'s TrayDescriptions folder .")
        input('\n    Fix this issue, come back, and hit ENTER to continue.')

tray_file_good = False
while tray_file_good is False:
    tray_headers, tray_data = read_file(os.path.join(project_dir, 'TrayDescriptions/', tray_description_file), ',')
    try:
        # Vial,Identifier 1,Injections
        vial = np.asarray(tray_data['Vial'], dtype=int)
        id1 = tray_data['Identifier1']
        expected_inj = np.asarray(tray_data['Injections'], dtype=int)
        print('    Tray description file read in successfully.')
        tray_file_good = True
    except KeyError:
        print('\n ** Problem with TrayDescription file. ** ')
        print('        - Make sure your tray description file has these column headings "Vial, Identifier 1, Injections".')
        print('        - If all the above columns are present, open it in a text editor (NOT EXCEL) and make sure their are no extra commas at the bottom.')
        input('\n    Fix the error, save the tray description file, come back, and hit ENTER to continue.')


# -------------------- read in data from hdf5 file --------------------
with h5py.File(f'{run_dir}{hdf5_file}', 'r') as hf:
    datasets = list(hf.keys())
    for data in datasets:
        globals()[data] = np.asarray(hf[data])


# -------------------- Dictate the delta calculation you are going to use, Picarro's or IsoLab's --------------------
"""dD is a DIY delta calculation using strengths whereas Delta_D_H is picarros calculation. They
   are nominally the same."""

delta_calc_choice = 'picarro'  # change to isolab for IsoLab's delta calculation using the raw absorption peak or strength.


if 'dDi' not in datasets:
    dcc = 'p'
    print('\n    Version 1.2 or lower was used to create the hdf5 file. This is fine, and means that your delta values originate from the Picarro software (rather than from Team IsoLab). If you want the choice, re-run picarro_h5.py.\n')
    dD = Delta_D_H.copy()
    datasets.append('dD')
    d18O = Delta_18_16.copy()
    datasets.append('d18O')
    if instrument['O17_flag']:
        d17O = Delta_17_16.copy()
        datasets.append('d17O')
else:
    if delta_calc_choice == 'picarro':
        dcc = 'p'
        print("\n    Using Picarro delta values. If you want Team IsoLab's, change 'delta_calc_choice' to isolab.\n")
        dD = dDp.copy()
        datasets.append('dD')
        d18O = d18Op.copy()
        datasets.append('d18O')
        if instrument['O17_flag']:
            d17O = d17Op.copy()
            datasets.append('d17O')
    elif delta_calc_choice == 'isolab':
        dcc = 'i'
        print("\n    Using IsoLab delta values. If you want Picarros's, change 'delta_calc_choice' to picarro.\n")
        dD = dDi.copy()
        datasets.append('dD')
        d18O = d18Oi.copy()
        datasets.append('d18O')
        if instrument['O17_flag']:
            d17O = d17Oi.copy()
            datasets.append('d17O')


# -------------------- Find individual injections --------------------
print('\n    Finding injection peaks.')

# Identify the very start of the injection, when the H2O starts to increase

pds = get_peak_detection_settings()

dT = np.diff(time)
dT = np.append(dT, np.mean(dT))

kernel = np.ones(pds['kernel_size']) / pds['kernel_size']
H2O_convolved = np.convolve(H2O, kernel, mode='same')
dH2O = np.diff(H2O_convolved)
dH2O = np.append(dH2O, dH2O[-1])  # duplicate last value so length of array is the same as the original length
dH2O_dT = dH2O / dT
dH2O_dT2 = np.diff(dH2O_dT)
dH2O_dT2 = np.append(dH2O_dT2, dH2O_dT2[-1])

adi = np.arange(0, len(H2O), 1)
peak = {}

peak['trough'] = np.where(np.logical_and(H2O < pds['lower_H2O_mode'] * 2,                                                   # threshold is sufficiently above baseline
                                             np.logical_and(dH2O_dT < 0,                                                    # H2O ppm is decreasing
                                                            np.logical_and(dH2O_dT2 > 0, dH2O_dT2 < pds['dH2O_dT2']))))[0]  # at the low end of acceleration

peak['start'] = peak['trough'][np.where(np.diff(peak['trough'])>pds['trough_diff'])[0]]  # this reduces the above trough to a single point marking the start of an injection
peak['end'] = peak['trough'][np.where(np.diff(peak['trough'])>pds['trough_diff'])[0]+1]  # this reduces the above trough to a single point marking the end of an injection

peak['width'] = peak['end'] - peak['start']

i=0
meanpeakwidth = int(np.round(np.mean(peak['width']), 0))
meanH2Opeak = np.zeros(meanpeakwidth)
while i < meanpeakwidth:
    meanH2Opeak[i] = np.mean(H2O[peak['start']+i])
    i+=1

dmeanH2Opeak = np.diff(meanH2Opeak)
dmeanH2Opeak = np.append(dmeanH2Opeak, dmeanH2Opeak[-1])
max_increase = np.where(dmeanH2Opeak == np.max(dmeanH2Opeak))[0]
# min_decrease = np.where(dmeanH2Opeak == np.min(dmeanH2Opeak))[0] # this is the default, the below line was used to offset some datasets that exhibit a steep decline at the start of the peak
min_decrease = np.where(dmeanH2Opeak[20:] == np.min(dmeanH2Opeak[20:]))[0]+20
dmeanH2Opeak_dT2 = np.diff(dmeanH2Opeak)
dmeanH2Opeak_dT2 = np.append(dmeanH2Opeak_dT2, dmeanH2Opeak_dT2[-1])
meanH2Opeaktop = np.where(abs(dmeanH2Opeak_dT2[max_increase[0]:min_decrease[0]]) < 20)[0]            # I would prefer to have "20" be a measured value, but for now, it is fixed and attempts to estimate a low-ish value for acceleration that seems appropriate for the top of the peak
trim_start = max_increase[0] + meanH2Opeaktop[10]                                                    # the "[10]" is an effort to offset from the start of the measured top
trim_end = len(meanH2Opeak) - meanH2Opeaktop[-2]                                                     # the "[-2]" is an effort to offset from the end of the measured top
peak['top_start'] = peak['start'] + trim_start
peak['top_end'] = peak['end'] - trim_end

peak['separation'] = peak['start'][1:] - peak['end'][0:-1]
peak_separation_mode = calc_mode(peak['separation'], 0)

missing_injections = np.where(np.round(peak['separation'] / peak_separation_mode, 0) > 1)[0]
# total_missing_injections = np.sum(np.round(peak['separation'] / peak_separation_mode, 0)[missing_injections])
total_missing_injections = len(missing_injections)

print(f"    Expected {np.sum(expected_inj)} injections, found {len(peak['start'])}")
if np.sum(expected_inj) != len(peak['start']):
    print(f"    It looks like {total_missing_injections} injection(s) is(are) unaccounted for. Check near injection(s) {missing_injections}, which should be on or near vial(s) {np.round(missing_injections / expected_inj[0],0)+1}.")


# Edge case when peak is erroneously narrow. If the start of the peak is after the end of the peak, remove it from the peak indexing arrays.
to_be_deleted = []
for i in range(len(peak['top_start'])):
    if peak['top_start'][i] > peak['top_end'][i]:
        to_be_deleted.append(i)

if len(to_be_deleted)>0:
    peak['top_start'] = np.delete(peak['top_start'], to_be_deleted)
    peak['top_end'] = np.delete(peak['top_end'], to_be_deleted)



# -------------------- Screen gdi set for outliers --------------------
H2O_residual = [H2O[i:j]-np.mean(H2O[i:j]) for i,j in zip(peak['top_start'], peak['top_end'])]
H2O_residual = np.concatenate(H2O_residual).ravel()
H2O_outlier_threshold = np.std(H2O_residual)*3.3

d18O_residual = [d18O[i:j]-np.mean(d18O[i:j]) for i,j in zip(peak['top_start'], peak['top_end'])]
d18O_residual = np.concatenate(d18O_residual).ravel()
d18O_outlier_threshold = np.std(d18O_residual)*3.3

dD_residual = [dD[i:j]-np.mean(dD[i:j]) for i,j in zip(peak['top_start'], peak['top_end'])]
dD_residual = np.concatenate(dD_residual).ravel()
dD_outlier_threshold = np.std(dD_residual)*3.3

gdi_H2O = [screen_outliers(H2O[i:j], H2O_outlier_threshold) for i,j in zip(peak['top_start'], peak['top_end'])]
gdi_d18O = [screen_outliers(d18O[i:j], d18O_outlier_threshold) for i,j in zip(peak['top_start'], peak['top_end'])]
gdi_dD = [screen_outliers(dD[i:j], dD_outlier_threshold) for i,j in zip(peak['top_start'], peak['top_end'])]
gdi_H2O__gdi_d18O = [np.intersect1d(i,j) for i,j in zip(gdi_H2O, gdi_d18O)]
gdi_H2O__gdi_d18O__gdi_dD = [np.intersect1d(i,j) for i,j in zip(gdi_H2O__gdi_d18O, gdi_dD)]

inj_gdi = [adi[i:j][k] for i, j, k in zip(peak['top_start'], peak['top_end'], gdi_H2O__gdi_d18O__gdi_dD)]

gdi = np.concatenate(inj_gdi).ravel()


# -------------------- Summarize Injection Data --------------------
print('\n    Summarizing injection data.')
dD_short_flag = False  # trim extra long injections to a shorter amount which reduces memory on commercial vaporizers

inj_exclude_list = ['ALARM_STATUS', 'AccelX', 'AccelY', 'AccelZ', 'Battery_Charge', 'Battery_Current', 'Battery_Temperature', 'Battery_Voltage',
                    'CH4_2min', 'CH4_30s', 'CH4_5min', 'CavityTemp1', 'CavityTemp2', 'CavityTemp3', 'CavityTemp4', 'FanState', 'Flow1',
                    'InletValve', 'Laser3Current', 'Laser3Tec', 'Laser3Temp', 'Laser4Current', 'Laser4Tec', 'Laser4Temp', 'ProcessedLoss1',
                    'ProcessedLoss2', 'ProcessedLoss3', 'ProcessedLoss4', 'SchemeTable', 'SchemeVersion', 'SpectrumID', 'cal_enabled', 'n2_flag']
included_datasets = list(set(datasets) - set(inj_exclude_list))

# RuntimeWarnings in this block and caught separately with my own notification
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

    inj = {}
    for data in included_datasets:
        inj[data] = {}
        inj[data]['mean'] = np.array(np.zeros(len(peak['top_start'])))
        inj[data]['std'] = np.array(np.zeros(len(peak['top_start'])))

        for i, j in enumerate(inj_gdi):
            try:
                inj[data]['mean'][i] = np.nanmean(eval(data)[j])
                inj[data]['std'][i] = np.nanstd(eval(data)[j])
            except (FloatingPointError, TypeError) as error:
                print(f'    **** Error with index {i} - {error} ****    ')
                inj[data]['mean'][i] = np.nan
                inj[data]['std'][i] = np.nan

    # Add custom arrays to inj dictionary that are otherwise not appropriate for mean and standard deviation.
    #    These variable names also need to be added to the list inj_extra_list inside picarro_vial.py.
    inj_extra_list = ['n_high_res', 'H2O_time_slope', 'dD_time_slope', 'd18O_time_slope', 'dD_H2O_slope', 'd18O_H2O_slope']

    for data in inj_extra_list:
        inj[data] = np.array(np.zeros(len(peak['top_start'])))

    for i, j in enumerate(inj_gdi):
        try:
            inj['n_high_res'][i] = np.asarray(len(H2O[j]))
            inj['H2O_time_slope'][i] = np.asarray(np.polyfit(H2O[j], time[j], 1)[0])
            inj['dD_time_slope'][i] = np.asarray(np.polyfit(time[j], dD[j], 1)[0])
            inj['d18O_time_slope'][i] = np.asarray(np.polyfit(time[j], d18O[j], 1)[0])
            inj['dD_H2O_slope'][i] = np.asarray(np.polyfit(H2O[j], dD[j], 1)[0])
            inj['d18O_H2O_slope'][i] = np.asarray(np.polyfit(H2O[j], d18O[j], 1)[0])
        except (FloatingPointError, TypeError) as error:
            print(f'    **** Error with index {i} - {error} ****    ')
            for data in inj_extra_list:
                inj[data][i] = np.nan


# -------------------- compare detected injections with expected injections from tray description --------------------
detected_inj = len(inj['H2O']['mean'])


if np.sum(expected_inj) != detected_inj:

    # -------------------- If actual number of injections do not match expected number of injections --------------------
    print(f"\n ** Expecting {np.sum(expected_inj)} injections. Found {detected_inj} injections. Look carefully at the figure to assess what went wrong. **")
    t.sleep(2)

    # save injection detection parameters and tell user to edit them
    with open(os.path.join(run_dir, 'peak_detection_settings.json'), 'w', newline='') as f:
        json.dump(pds, f, indent=2)

    fig_a = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="H2O (ppmv)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig_a.circle(adi, H2O, color="black", legend_label="All data", size=2)
    fig_a.circle(gdi, H2O[gdi], color="yellow", legend_label="Data identified as good but...", size=6)
    fig_a.circle(peak['top_start'], H2O[peak['top_start']], color="green", size=6, legend_label="Start of each injection")
    fig_a.circle(peak['top_end'], H2O[peak['top_end']], color="red", size=6, legend_label="End of each injection")
    fig_a_caption = f"""Figure A. Water concentration during your run."""

    fig_b = figure(width=1100, height=700,
                   y_range=(-100, 10),
                   x_axis_label="data index",
                   y_axis_label="d18O raw (permil)",
                   tools="pan, box_zoom, reset, save",
                   active_drag="box_zoom")
    fig_b.circle(adi, d18O, color="black", legend_label="All data", size=2)
    fig_b.circle(gdi, d18O[gdi], color="yellow", legend_label="Data identified as good but...", size=6)
    fig_b.circle(peak['top_start'], d18O[peak['top_start']], color="green", size=6, legend_label="Start of each injection")
    fig_b.circle(peak['top_end'], d18O[peak['top_end']], color="red", size=6, legend_label="End of each injection")
    fig_b_caption = f"""Figure 3. Oxygen-18 isotope composition (d18O or delta 18 Oh)."""

    # -------------------- make mismatch html page --------------------
    header = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <!-- py by Andrew Schauer -->
            <meta http-equiv="Content-Type" content="text/html charset=UTF-8" />
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <link rel="stylesheet" type="text/css" href="py_style.css">
            <title>{instrument['name']}</title>
        </head>
        <body id="top">\n
        <h2>{instrument['name']} {hdf5_file[0:-5]} Problem with Number of Injections Accounting</h2>
        <div class="text-indent">
        <p>Last updated - ''' + str(dt.datetime.now()) + '''</p>
        <p>Use the two figures below the help determine why the number of expected injections does not match the measured injections.</p>'''

    html_path = os.path.join(run_dir, f'{hdf5_file[0:-5]}_Injection_Accounting_Problem.html')
    with open(html_path, 'w') as html_page:
        html_page.write(header)
        html_page.write(file_html(fig_a, CDN))  # INLINE when no internet, CDN otherwise
        html_page.write('<p>[figure 1 caption]</p>')
        html_page.write(file_html(fig_b, CDN))  # INLINE when no internet, CDN otherwise
        html_page.close()
    webbrowser.open(html_path)

else:
    # set flag that all expected injections from tray file equals the number of detected injections
    accounted_for_all_injs = 1

    # -------------------- remove mismatch file if it exists --------------------
    if os.path.isfile(os.path.join(run_dir, f'{hdf5_file[0:-5]}_Injection_Accounting_Problem.html')):
        os.remove(os.path.join(run_dir, f'{hdf5_file[0:-5]}_Injection_Accounting_Problem.html'))

    # -------------------- put sample IDs and vial inj number into inj dictionary --------------------
    inj['id1'] = [[i] * j for i, j in zip(id1, expected_inj)]
    inj['id1'] = [x for xs in inj['id1'] for x in xs]
    inj['vial_num'] = [[i] * j for i, j in zip(vial, expected_inj)]
    inj['vial_num'] = [x for xs in inj['vial_num'] for x in xs]
    inj['inj_num'] = [list(range(1, ei + 1)) for ei in expected_inj]
    inj['inj_num'] = [x for xs in inj['inj_num'] for x in xs]




    # -------------------- quality control injections --------------------
    print('\n    Checking quality of injections.')

    qcp = get_quality_control_parameters()

    if qcp['max_H2O_std'] > 2000:
        print(f" ** Check your injections, the H2O ppm seems noisier than normal.")
    if qcp['max_d18O_std'] > 0.3*4.4:
        print(f" ** Check your injections, the d18O seems noisier than normal.")
    if qcp['max_dD_std'] > 0.8*4.4:
        print(f" ** Check your injections, the dD seems noisier than normal.")

    inj['flag'] = np.ones(len(inj['H2O']['mean']))
    inj['flag_reason'] = [' ' for i in inj['flag']]
    for i, _ in enumerate(inj['H2O']['std']):
        if inj['H2O']['std'][i] > qcp['max_H2O_std']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'above max_H2O_std'
            print(f"\n    Injection {i} had high H2O standard deviation ({round(inj['H2O']['std'][i], 0)} ppm). Threshold is {qcp['max_H2O_std']} ppm.")
        elif inj['d18O']['std'][i] > qcp['max_d18O_std']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'above max_d18O_std'
            print(f"\n    Injection {i} had high d18O standard deviation ({round(inj['d18O']['std'][i], 3)} permil). Threshold is {qcp['max_d18O_std']} permil.")
        elif inj['dD']['std'][i] > qcp['max_dD_std']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'above max_dD_std'
            print(f"\n    Injection {i} had high dD standard deviation ({round(inj['dD']['std'][i], 3)} permil). Threshold is {qcp['max_dD_std']} permil.")
        elif inj['H2O']['mean'][i] < qcp['min_H2O']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'below min_H2O'
            print(f"\n    Injection {i} had low water concentration ({round(inj['H2O']['mean'][i], 0)} ppmv). Threshold is {qcp['min_H2O']} ppmv.")
        elif inj['CavityPressure']['std'][i] > qcp['max_CAVITYPRESSURE_std']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'above max_CAVITYPRESSURE_std'
            print(f"\n    Injection {i} had high cavity pressure standard deviation ({round(inj['CavityPressure']['std'][i],3)} Torr). Threshold is {qcp['max_CAVITYPRESSURE_std']} Torr.")
    flag0_index = np.where(inj['flag']==0)[0]
    if len(flag0_index)>0:
        fdi = np.concatenate([inj_gdi[i] for i in flag0_index]).ravel()
    else:
        fdi = []
    flagged_reason = [inj['flag_reason'][i] for i in flag0_index]
    flagged_id1 = [inj['id1'][i] for i in flag0_index]
    flagged_vial = [inj['vial_num'][i] for i in flag0_index]
    flagged_inj = [inj['inj_num'][i] for i in flag0_index]

    # -------------------- make high resolution data figures --------------------
    fig1 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="H2O (ppmv)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig1.circle(adi, H2O, color="black", legend_label="All data", size=2)
    fig1.circle(gdi, H2O[gdi], color="green", legend_label="Good data", size=6)
    fig1.circle(fdi, H2O[fdi], color="yellow", legend_label="Flagged data", size=6)
    fig1_caption = f"""Figure 1. Water concentration during your run where each injection peak top is shown
                       in green. Reasonable injection water concentrations range from 17000 to 23000 ppmv.
                       Injections with an H2O standard deviation above {qcp['max_H2O_std']} are <a href="#flagged_injections">flagged</a>."""

    fig2 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="H2O (ppmv)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    [fig2.line(range(i-i,j-i), H2O[range(i,j)]) for i,j in zip(peak['start'], peak['end'])]
    fig2_caption = f"""Figure 2. Water concentration for each pulse plotted on top of each other. Zero is the maximum derivative of H2O during
                       the beginning of an injection. If these data are from a HotTee vaporizer, then the errant spikes are most likely from
                       bubbles in the syringe. Bubbles are evil."""

    fig3 = figure(width=1100, height=700,
                  y_range=(-500,50),
                  x_axis_label="data index",
                  y_axis_label="dD raw (permil)",
                  tools="pan, box_zoom, reset, save",
                  active_drag="box_zoom")
    fig3.circle(adi, dD, color="black", legend_label="All data", size=2)
    fig3.circle(gdi, dD[gdi], color="green", legend_label="Good data", size=6)
    fig3.circle(fdi, dD[fdi], color="yellow", legend_label="Flagged data", size=6)
    fig3_caption = f"""Figure 3. Hydrogen isotope composition (dD or delta Dee). Injections with a standard
                       deviation greater than {qcp['max_dD_std']} permil are <a href="#flagged_injections">flagged</a>."""

    fig4 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="dD - departure from the last 20 measurements (permil)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    [fig4.circle(range(i-i,j-i), dD[range(i,j)]-np.mean(dD[range(j-20,j)]), size=2) for i,j in zip(peak['top_start'], peak['top_end'])]
    fig4_caption = f"""Figure 4. dD departure from the final 20 measurements of each top of injection peak."""

    fig5 = figure(width=1100, height=700,
                  y_range=(-100,10),
                  x_axis_label="data index",
                  y_axis_label="d18O raw (permil)",
                  tools="pan, box_zoom, reset, save",
                  active_drag="box_zoom")
    fig5.circle(adi, d18O, color="black", legend_label="All data", size=2)
    fig5.circle(gdi, d18O[gdi], color="green", legend_label="Good data", size=6)
    fig5.circle(fdi, d18O[fdi], color="yellow", legend_label="Flagged data", size=6)
    fig5_caption = f"""Figure 5. Oxygen-18 isotope composition (d18O or delta 18 Oh). Injections with a standard
                       deviation greater than {qcp['max_d18O_std']} permil are <a href="#flagged_injections">flagged</a>."""

    fig6 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="d18O - departure from the last 20 measurements (permil)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    [fig6.circle(range(i-i,j-i), d18O[range(i,j)]-np.mean(d18O[range(j-20,j)]), size=2) for i,j in zip(peak['top_start'], peak['top_end'])]
    fig6_caption = f"""Figure 6. d18O departure from the final 20 measurements of each top of injection peak."""

    # -------------------- make html page --------------------
    print('\nMaking html page...')
    header = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <!-- py by Andrew Schauer -->
            <meta http-equiv="Content-Type" content="text/html charset=UTF-8" />
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <link rel="stylesheet" type="text/css" href="py_style.css">
            <title>{instrument['name']} - High Res Data</title>
        </head>
        <body id="top">\n

        <div class="created-date">Created - {str(dt.datetime.now())}</div>

        <h2>{instrument['name']} {hdf5_file[0:-5]} High Resolution Data</h2>
        <div class="text-indent">
        <p>High resolution data. Use this page to assess the quality of the high res data.</p>
        <hr><p>'''

    flagged_block = str([f"<tr><td>{flagged_vial[i]}</td><td>{flagged_inj[i]}</td><td>{flagged_id1[i]}</td><td>{flagged_reason[i]}</td></tr>" for i, _ in enumerate(flagged_inj)]).replace("[", "").replace("'", "").replace("]", "").replace(", ", "")

    python_scripts_block = str([f'<li><a href="python/{key}_ARCHIVE_COPY">{key}</a> - {value}</li>' for key, value in python_scripts.items()]).replace("[", "").replace("'", "").replace("]", "").replace(", ", "")

    footer = f"""
        <h2 id="refs">References</h2>
        <div class="references">
        <ul>
            <li>Python scripts - modification date:
                <ul>
                    {python_scripts_block}
                </ul>
        </ul>
        </div>
        </body></html>"""

    html_path = os.path.join(run_dir, f'{hdf5_file[0:-5]}_high_res_figures.html')
    with open(html_path, 'w') as html_page:
        html_page.write(header)
        html_page.write(file_html(fig1, CDN))  # INLINE when no internet, CDN otherwise
        html_page.write(f"{fig1_caption}<hr><br>")
        html_page.write(file_html(fig2, CDN))  # INLINE when no internet, CDN otherwise
        html_page.write(f"{fig2_caption}<hr><br>")
        html_page.write(file_html(fig3, CDN))  # INLINE when no internet, CDN otherwise
        html_page.write(f"{fig3_caption}<hr><br>")
        html_page.write(file_html(fig4, CDN))  # INLINE when no internet, CDN otherwise
        html_page.write(f"{fig4_caption}<hr><br>")
        html_page.write(file_html(fig5, CDN))  # INLINE when no internet, CDN otherwise
        html_page.write(f"{fig5_caption}<hr><br>")
        html_page.write(file_html(fig6, CDN))  # INLINE when no internet, CDN otherwise
        html_page.write(f"{fig6_caption}<hr><br>")
        html_page.write("""<h2>Flagged injections</h2>
                        <div class="text-indent">
                        <div id="flagged_injections">These injections were flagged:<table>
                        <tr><th>Vial<br>Number</th><th>Injection<br>Number</th><th>Sample<br>ID</th><th>Reason</th></tr>""")
        html_page.write(flagged_block)
        html_page.write('</table></div></div>')
        html_page.write(footer)
        html_page.close()
    webbrowser.open(html_path)

    # -------------------- write injection level data to json file --------------------
    inj_export = inj.copy()
    for data in included_datasets:
        inj_export[data]['mean'] = inj[data]['mean'].tolist()
        inj_export[data]['std'] = inj[data]['std'].tolist()
    for data in inj_extra_list:
        inj_export[data] = inj[data].tolist()

    inj_export['flag'] = inj_export['flag'].tolist()
    inj_export['vial_num'] = [int(i) for i in inj['vial_num']]

    with open(os.path.join(run_dir, f'{hdf5_file[0:-5]}_{dcc}_injections.json'), 'w') as fp:
        json.dump(inj_export, fp)
