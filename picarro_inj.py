#!/usr/bin/env python3
"""
This script reads in the hdf5 file created by picarro_h5.py if the data set exclusively contains discrete samples (injections).
This script will find each injection and reduce the h5 level data (~ 1 Hz data) to injection level summaries. A tray
description file is required. This script effectively creates Picarro's Coordinator output, albeit in json format
and with all of the data fields present that are in the original h5 files.

Version 1.4 from 2023-02-14 has the the tray description file parsing higher in the code so that if it errors out we don't have
to wait for the injection level data to completely process. Also, it attempts to help the user understand that a tray description
file error is happening and prompts for the issue to be fixed before continuing.

Version 1.5 from 2023-06-07 has the ability to deal with the new Hot Tee injection method. Stay tuned for a publication on this.
The short version is, the picarro vaporizer died and I took the opportunity to test a method I had been wondering about. Injecting
very slowly into a hot tee. Turns out it works quite well. More on this later. Also in this version, and in an attempt to make a single
script allow for many different types and sizes of injections, I have tried to make the peak detection settings (pds dictionary)
more flexible. Version 1.51 corrects typo in figure legend.
"""

__author__ = "Andy Schauer"
__email__ = "aschauer@uw.edu"
__last_modified__ = "2023-06-24"
__version__ = "1.51"
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
import shutil
import time as t
import warnings
import webbrowser


# -------------------- functions ------------------------------------
def calc_mode(rd, rnd):
    """Calculate the mode of a raw dataset rd after having rounded it to 
    the nearest number of specified decimal places, rnd."""
    d = [round(i, rnd) for i in rd]
    v,c = np.unique(d, return_counts=True)
    i = np.argmax(c)
    return v[i]


# -------------------- get instrument information --------------------
""" Get specific picarro instrument whose data is being processed as well as some associated information. Populate
this function with your own instrument(s). The function get_instrument() is located in picarro_lib.py."""
instrument, ref_ratios, inj_peak, inj_quality, vial_quality = get_instrument()


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
        # Project,Tray,Vial,Identifier 1,Injections
        project = tray_data['Project']
        tray = tray_data['Tray']
        vial = np.asarray(tray_data['Vial'], dtype=int)
        id1 = tray_data['Identifier1']
        expected_inj = np.asarray(tray_data['Injections'], dtype=int)
        print('    Tray description file read in successfully.')
        tray_file_good = True
    except KeyError:
        print('\n ** Problem with TrayDescription file. ** ')
        print('        - Make sure your tray description file has these column headings "Project, Tray, Vial, Identifier 1, Injections".')
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


# get mode of H2O so we know where to center the H2O thresholds
#    - this assumes one injection size, still working on getting multiple injection sizes
H2Omode = calc_mode(H2O[H2O>1000], -1) # round to nearest 10s position
H2Omode_range = 10000


# check for the existence of a settings file to grab custom peak detection settings
if os.path.isfile(os.path.join(run_dir, 'peak_detection_settings.json')):
    existing_pds = True
    with open(os.path.join(run_dir, 'peak_detection_settings.json'), 'r') as f:
        pds = json.load(f)

else:
    existing_pds = False
    pds = {"H2O_threshold_min": H2Omode - H2Omode_range/2,
           "H2O_threshold_max": H2Omode + H2Omode_range/2,
           "dH2O_dT2_threshold": 50,
           "kernel_size": 5,
           "min_pts_in_pks": 70,
           "min_pts_between_pks": 60,  # about 60 seconds between peaks
           "trim_end": 5,
           "trim_start": 5}

    with open(os.path.join(run_dir, 'peak_detection_settings.json'), 'w', encoding='utf-8') as f:
        json.dump(pds, f, ensure_ascii=False, indent=4)

time_diff = np.diff(time)
time_diff = np.append(time_diff, np.mean(time_diff))
kernel = np.ones(pds['kernel_size']) / pds['kernel_size']
H2O_convolved = np.convolve(H2O, kernel, mode='same')
H2O_diff = np.diff(H2O_convolved)
H2O_diff = np.append(H2O_diff, H2O_diff[-1])  # duplicate last value so length of array is the same as the original length
dH2O_dT = H2O_diff / time_diff
dH2O_dT_convolved = np.convolve(dH2O_dT, kernel, mode='same')
dH2O_dT2 = np.diff(dH2O_dT_convolved)
dH2O_dT2 = np.append(dH2O_dT2, dH2O_dT2[-1])
dH2O_dT2_convolved = np.convolve(dH2O_dT2, kernel, mode='same')

# update pds dictionary if one did not already exist
if existing_pds is False:
    pds['dH2O_dT2_threshold'] = np.std(dH2O_dT2_convolved)


pks = np.where(np.logical_and(abs(dH2O_dT2_convolved) < pds['dH2O_dT2_threshold'], 
                              np.logical_and(H2O > pds['H2O_threshold_min'],
                                             H2O < pds['H2O_threshold_max'])))[0]

adi = np.arange(0, len(H2O), 1)
pks_end = np.asarray(np.diff(pks) > pds['min_pts_between_pks']).nonzero()[0]
end_of_each_inj = adi[pks[pks_end]]
end_of_each_inj = np.append(end_of_each_inj, pks[-1])
end_of_each_inj = end_of_each_inj - pds['trim_end']
pks_start = pks_end + 1
start_of_each_inj = adi[pks[pks_start]]
start_of_each_inj = np.insert(start_of_each_inj, 0, pks[0])
start_of_each_inj = start_of_each_inj + pds['trim_start']

# if the start of the peak is after the end of the peak, remove it from the peak indexing arrays
to_be_deleted = []
for i in range(len(start_of_each_inj)):
    if start_of_each_inj[i] > end_of_each_inj[i]:
        to_be_deleted.append(i)

if len(to_be_deleted)>0:
    start_of_each_inj = np.delete(start_of_each_inj, to_be_deleted)
    end_of_each_inj = np.delete(end_of_each_inj, to_be_deleted)


inj_gdi = [adi[i:j] for i, j in zip(start_of_each_inj, end_of_each_inj)]
gdi = np.concatenate(inj_gdi).ravel()



# -------------------- Summarize Injection Data --------------------
print('\n    Summarizing injection data.')

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
        inj[data]['mean'] = np.array(np.zeros(len(start_of_each_inj)))
        inj[data]['std'] = np.array(np.zeros(len(start_of_each_inj)))

        for i, _ in enumerate(start_of_each_inj):
            if data == 'dD' and (end_of_each_inj[i] - start_of_each_inj[i]) > 120:
                curr_range = range(start_of_each_inj[i], start_of_each_inj[i] + 120)
            else:
                curr_range = range(start_of_each_inj[i], end_of_each_inj[i])
            try:
                inj[data]['mean'][i] = np.nanmean(eval(data)[curr_range])
                inj[data]['std'][i] = np.nanstd(eval(data)[curr_range])
            except (FloatingPointError, TypeError) as error:
                print(f'    **** Error with index {i} - {error} ****    ')
                inj[data]['mean'][i] = np.nan
                inj[data]['std'][i] = np.nan

    # Add custom arrays to inj dictionary that are otherwise not appropriate for mean and standard deviation.
    #    These variable names also need to be added to the list inj_extra_list inside picarro_vial.py.
    inj_extra_list = ['n_high_res', 'H2O_time_slope', 'dD_time_slope', 'd18O_time_slope', 'dD_H2O_slope', 'd18O_H2O_slope']

    for data in inj_extra_list:
        inj[data] = np.array(np.zeros(len(start_of_each_inj)))

    for i, _ in enumerate(start_of_each_inj):
        curr_range = range(start_of_each_inj[i], end_of_each_inj[i])
        try:
            inj['n_high_res'][i] = np.asarray(len(H2O[curr_range]))
            inj['H2O_time_slope'][i] = np.asarray(np.polyfit(H2O[curr_range], time[curr_range], 1)[0])
            inj['dD_time_slope'][i] = np.asarray(np.polyfit(time[curr_range], dD[curr_range], 1)[0])
            inj['d18O_time_slope'][i] = np.asarray(np.polyfit(time[curr_range], d18O[curr_range], 1)[0])
            inj['dD_H2O_slope'][i] = np.asarray(np.polyfit(H2O[curr_range], dD[curr_range], 1)[0])
            inj['d18O_H2O_slope'][i] = np.asarray(np.polyfit(H2O[curr_range], d18O[curr_range], 1)[0])
        except (FloatingPointError, TypeError) as error:
            print(f'    **** Error with index {i} - {error} ****    ')
            for data in inj_extra_list:
                inj[data][i] = np.nan


# -------------------- compare detected injections with expected injections from tray description --------------------
detected_inj = len(inj['H2O']['mean'])


if np.sum(expected_inj) != detected_inj:

    # -------------------- If actual number of injections do not match expected number of injections --------------------
    print(f"\n** Expecting {np.sum(expected_inj)} injections. Found {detected_inj} injections. Look carefully at the figure to assess what went wrong. **")
    t.sleep(2)

    # save injection detection parameters and tell user to edit them
    with open(os.path.join(run_dir, 'peak_detection_settings.json'), 'w', newline='') as f:
        json.dump(pds, f, indent=2)

    fig_a = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="H2O (ppmv)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig_a.circle(adi, H2O, color="black", legend_label="All data", size=2)
    fig_a.circle(gdi, H2O[gdi], color="yellow", legend_label="Data identified as good but...", size=6)
    fig_a.circle(start_of_each_inj, H2O[start_of_each_inj], color="green", size=6, legend_label="Start of each injection")
    fig_a.circle(end_of_each_inj, H2O[end_of_each_inj], color="red", size=6, legend_label="End of each injection")
    fig_a_caption = f"""Figure A. Water concentration during your run."""

    fig_b = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="d18O raw (permil)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig_b.circle(adi, d18O, color="black", legend_label="All data", size=2)
    fig_b.circle(gdi, d18O[gdi], color="yellow", legend_label="Data identified as good but...", size=6)
    fig_b.circle(start_of_each_inj, d18O[start_of_each_inj], color="green", size=6, legend_label="Start of each injection")
    fig_b.circle(end_of_each_inj, d18O[end_of_each_inj], color="red", size=6, legend_label="End of each injection")
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
    inj['project'] = [[i] * j for i, j in zip(project, expected_inj)]
    inj['project'] = [x for xs in inj['project'] for x in xs]
    inj['id1'] = [[i] * j for i, j in zip(id1, expected_inj)]
    inj['id1'] = [x for xs in inj['id1'] for x in xs]
    inj['vial_num'] = [[i] * j for i, j in zip(vial, expected_inj)]
    inj['vial_num'] = [x for xs in inj['vial_num'] for x in xs]
    inj['inj_num'] = [list(range(1, ei + 1)) for ei in expected_inj]
    inj['inj_num'] = [x for xs in inj['inj_num'] for x in xs]

    # -------------------- quality control injections --------------------
    inj_quality['max_H2O_std'] = round(calc_mode(inj['H2O']['std'], -1) * 3.3, 0)
    inj_quality['max_d18O_std'] = round(calc_mode(inj['d18O']['std'], 1) * 3.3, 2)
    inj_quality['max_dD_std'] = round(calc_mode(inj['dD']['std'], 1) * 3.3, 1)
    
    if inj_quality['max_H2O_std'] > 2000:
        print(f" ** Check your injections, the H2O ppm seems noisier than normal.")
    if inj_quality['max_d18O_std'] > 1.0:
        print(f" ** Check your injections, the d18O seems noisier than normal.")
    if inj_quality['max_dD_std'] > 2.0:
        print(f" ** Check your injections, the dD seems noisier than normal.")


    # # used for all campcentury runs
    # inj_quality['max_H2O_std'] = 5000
    # inj_quality['max_d18O_std'] = 5
    # inj_quality['max_dD_std'] = 5


    inj['flag'] = np.ones(len(inj['H2O']['mean']))
    inj['flag_reason'] = [' ' for i in inj['flag']]
    for i, _ in enumerate(inj['H2O']['std']):
        if inj['H2O']['std'][i] > inj_quality['max_H2O_std']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'above max_H2O_std'
            print(f"\n    Injection {i} had high H2O standard deviation ({round(inj['H2O']['std'][i], 0)} ppm). Threshold is {inj_quality['max_H2O_std']} ppm.")
        elif inj['d18O']['std'][i] > inj_quality['max_d18O_std']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'above max_d18O_std'
            print(f"\n    Injection {i} had high d18O standard deviation ({round(inj['d18O']['std'][i], 3)} permil). Threshold is {inj_quality['max_d18O_std']} permil.")
        elif inj['dD']['std'][i] > inj_quality['max_dD_std']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'above max_dD_std'
            print(f"\n    Injection {i} had high dD standard deviation ({round(inj['dD']['std'][i], 3)} permil). Threshold is {inj_quality['max_dD_std']} permil.")
        elif inj['H2O']['mean'][i] < inj_quality['min_H2O']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'below min_H2O'
            print(f"\n    Injection {i} had low water concentration ({round(inj['H2O']['mean'][i], 0)} ppmv). Threshold is {inj_quality['min_H2O']} ppmv.")
        elif inj['CavityPressure']['std'][i] > inj_quality['max_CAVITYPRESSURE_std']:
            inj['flag'][i] = 0
            inj['flag_reason'][i] = 'above max_CAVITYPRESSURE_std'
            print(f"\n    Injection {i} had high cavity pressure standard deviation ({round(inj['CavityPressure']['std'][i],3)} Torr). Threshold is {inj_quality['max_CAVITYPRESSURE_std']} Torr.")
    fdi = np.concatenate([inj_gdi[i] for i in np.where(inj['flag'] == 0)[0]]).ravel()
    flagged_reason = [inj['flag_reason'][i] for i in np.where(inj['flag'] == 0)[0]]
    flagged_id1 = [inj['id1'][i] for i in np.where(inj['flag'] == 0)[0]]
    flagged_vial = [inj['vial_num'][i] for i in np.where(inj['flag'] == 0)[0]]
    flagged_inj = [inj['inj_num'][i] for i in np.where(inj['flag'] == 0)[0]]

    # -------------------- make high resolution data figures --------------------
    fig1 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="H2O (ppmv)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig1.circle(adi, H2O, color="black", legend_label="All data", size=2)
    fig1.circle(gdi, H2O[gdi], color="green", legend_label="Good data", size=6)
    fig1.circle(fdi, H2O[fdi], color="yellow", legend_label="Flagged data", size=6)
    fig1_caption = f"""Figure 1. Water concentration during your run where each injection peak top is shown
                       in green. Reasonable injection water concentrations range from 17000 to 23000 ppmv.
                       Any injection below 10000 ppmv is <a href="#flagged_injections">flagged</a>."""

    fig2 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="Cavity Pressure (Torr)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig2.circle(adi, CavityPressure, color="black", legend_label="All data", size=2)
    fig2.circle(gdi, CavityPressure[gdi], color="green", legend_label="Good data", size=6)
    fig2.circle(fdi, CavityPressure[fdi], color="yellow", legend_label="Flagged data", size=6)
    fig2_caption = f"""Figure 2. Cavity pressure is carefully controlled. Injections with a standard deviation
                       greater than {inj_quality['max_CAVITYPRESSURE_std']} Torr are <a href="#flagged_injections">flagged</a>."""

    fig3 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="dD raw (permil)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig3.circle(adi, dD, color="black", legend_label="All data", size=2)
    fig3.circle(gdi, dD[gdi], color="green", legend_label="Good data", size=6)
    fig3.circle(fdi, dD[fdi], color="yellow", legend_label="Flagged data", size=6)
    fig3_caption = f"""Figure 3. Hydrogen isotope composition (dD or delta Dee). Injections with a standard
                       deviation greater than {inj_quality['max_dD_std']} permil are <a href="#flagged_injections">flagged</a>."""

    fig4 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="d18O raw (permil)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig4.circle(adi, d18O, color="black", legend_label="All data", size=2)
    fig4.circle(gdi, d18O[gdi], color="green", legend_label="Good data", size=6)
    fig4.circle(fdi, d18O[fdi], color="yellow", legend_label="Flagged data", size=6)
    fig4_caption = f"""Figure 3. Oxygen-18 isotope composition (d18O or delta 18 Oh). Injections with a standard
                       deviation greater than {inj_quality['max_d18O_std']} permil are <a href="#flagged_injections">flagged</a>."""

    # -------------------- prepare run directory --------------------
    shutil.copy2(os.path.join(python_dir, 'py_report_style.css'), os.path.join(run_dir, 'py_style.css'))
    if os.path.isdir(os.path.join(run_dir, 'python_archive')) is False:
        os.mkdir(os.path.join(run_dir, 'python_archive'))
    [shutil.copy2(os.path.join(python_dir, script), os.path.join(run_dir, f"python_archive/{script}_ARCHIVE_COPY")) for script in python_scripts]

    # -------------------- make html page --------------------
    print('Making html page...')
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
