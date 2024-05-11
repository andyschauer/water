#!/usr/bin/env python3
"""
This is the final script for discrete injection type data. It summarizes injections from individual vials
and then calibrates dD, d18O, (and d17O if using a L2140i) to the VSMOW-SLAP scale. Ideally, picarro_h5.py and
picarro_inj.py were run prior to the present script.

Version 1.3 mod date 2023-02-14 => found bug in no drift correction on L2130 instrument. Fixed.
Version 1.4 mod date 2023-05-07 => found bug when a reference water had a null value. Changed to np.nan. Seems to work fine now.
Version 1.41 mod date 2023.05.30 => minor change of "water" to "waters" with respect to the reference materials json file. I
    updated the refmat file for broader reach.
Version 1.5 mod date 2023.06.28 => added tray description file to report directory and report
Version 1.6 mod date 2023.08.01 => added memory calculation
Version 1.61 mod date 2023.10.17 => bug in memory calculation when considering analyses as a set, fix is creating combined injection array for deltas
Version 1.7 mod date 2023.11.28 => removed project from data stream
version 1.8 mod date 2024.05.03 => renamed name to names from change in reference_materials.json file
"""

__author__ = "Andy Schauer"
__email__ = "aschauer@uw.edu"
__last_modified__ = "2024-05-03"
__version__ = "1.8"
__copyright__ = "Copyright 2024, Andy Schauer"
__license__ = "Apache 2.0"
__acknowledgements__ = "M. Sliwinski, H. Lowes-Bicay, N. Brown"


# -------------------- Imports --------------------
import csv
import datetime as dt
import json
import matplotlib.pyplot as pplt
from natsort import natsorted
import numpy as np
import os
from picarro_lib import *
import shutil
import webbrowser


# -------------------- Functions --------------------
def fig_as_thumb(fhp, figname, thumbcaption):
    """Write html for figure thumb linked at top of web page."""
    thumb = f'<div class="figThumb">\
        <div class="thumbcaption">{thumbcaption}</div>\
        <a href="#{figname}"><img src="{figname}"/></a></div>\n'
    fhp.write(thumb)


def fig_in_html(fhp, figname, caption):
    """Write html for large figure view."""
    imgtag = f'<p><div class="largeFigs">\
            <div class="caption" id="{figname}"><br>{caption}\
                <a href="#top"><br>Jump back to top.</a>\
            </div><img src="{figname}"/>\
        </div></p><hr>\n'
    fhp.write(imgtag)


def get_default_quality_control_parameters():
    # qcp = {'max_H2O_std': round(calc_mode(vial['H2O']['std'], -1) * 4.4, 0),
    #        'max_d18O_std': round(calc_mode(vial['d18O']['std'], 2) * 4.4, 3),
    #        'max_dD_std': round(calc_mode(vial['dD']['std'], 2) * 4.4, 2)}

    # if instrument['O17_flag']:
    #     qcp['max_d17O_std'] = round(calc_mode(vial['d17O']['std'], 2) * 4.4, 3)
    qcp = {'max_H2O_std': round(np.nanmean(vial['H2O']['std']) * 4.4, 0),
           'max_d18O_std': round(np.nanmean(vial['d18O']['std']) * 4.4, 3),
           'max_dD_std': round(np.nanmean(vial['dD']['std']) * 4.4, 2)}

    if instrument['O17_flag']:
        qcp['max_d17O_std'] = round(np.nanmean(vial['d17O']['std']) * 4.4, 3)

    with open(os.path.join(run_dir, 'quality_control_parameters_vial.json'), 'w', encoding='utf-8') as f:
        json.dump(qcp, f, ensure_ascii=False, indent=4)

    return qcp


def memory_calc(delta):
    """This memory, or carry-over calculation attempts to estimate the proportion of water from the current injection
    that is presently being measured. Some of the previously injected water is still present in the system. In this
    calculation, a memory estimate of 0% would imply no carry-over at all and indicate that the only water presently
    being measured is from the current injection. Memory estimates vary depending on the delta used. d18O tends to provide
    a lower memory estimate compared with dD. As such, my default is to use the hydrogen isotope composition as a memory
    estimate. Furthermore, this estimate seems to be only valid when the range is sufficiently large. This makes intuitive
    sense because as the two waters become more similar, a memory estimate becomes more challenging to measure, and indeed,
    less meaningful. I have placed a 10 permil difference threshold to cut off the erroneous estimates from those waters
    with a less than 10 permil difference. Mathmatically, as the difference between the two waters goes to zero, the
    proportion goes to zero.
    """
    memory = {
        'vial_to_vial_range': np.zeros(len(vial['vial_num'])),
        'within_vial_range': np.zeros(len(vial['vial_num'])),
        'vial_memory': np.zeros(len(vial['vial_num']))}
    n=0
    for i in range(len(vial['vial_num'])):
        # get index of all injections for the current (v1) and previous (v0) vials
        if i==0:
            pass
        else:
            v0_injs = int(vial['total_inj'][i-1])
            v0 = np.asarray(range(n, n + v0_injs))
            v1_injs = int(vial['total_inj'][i])
            v1 = np.asarray(range(n + v0_injs, n + v0_injs + v1_injs))
            memory['vial_to_vial_range'][i] = np.abs(np.mean(np.asarray(delta)[v1][-2:]) - np.mean(np.asarray(delta)[v0][-2:]))
            memory['within_vial_range'][i] = np.abs(np.asarray(delta)[v1][0] - np.mean(np.asarray(delta)[v1][-2:]))
            memory['vial_memory'][i] = 1 - ((memory['vial_to_vial_range'][i] - memory['within_vial_range'][i]) / memory['vial_to_vial_range'][i])
            memory['mean'] = np.mean(memory['vial_memory'][np.where(memory['vial_to_vial_range']>10)])
            # fig, ax = pplt.subplots()
            # ax.plot(v0, np.asarray(inj[delta]['mean'])[v0], 'ro')
            # ax.plot(v1, np.asarray(inj[delta]['mean'])[v1], 'bo')
            # pplt.show()
            n+=(v0_injs)

    return memory




# -------------------- CONSTANTS --------------------
FIRST_INJECTIONS_TO_IGNORE = 2
DRIFT_CORRECTION = True


# -------------------- get instrument information --------------------
""" Get specific picarro instrument whose data is being processed as well as some associated information. Populate
this function with your own instrument(s). The function get_instrument() is located in picarro_lib.py."""
instrument = get_instrument()


# -------------------- paths --------------------
python_dir = get_path("python")
project_dir = f"{get_path('project')}{instrument['name'].lower()}/"
fig_dir = 'figures/'

run_or_set = input("\nDo you want to look at a single run or a set of runs? (type 'run' or 'set'): ")
if run_or_set == 'run':
    run_dir = os.path.join(project_dir, 'runs/')

elif run_or_set == 'set':
    run_dir = os.path.join(project_dir, 'sets/')

archive_dir = os.path.join(run_dir, 'archive/')


# ---------- Identify json data file(s) to be loaded ----------
run_list = natsorted(os.listdir(run_dir))
print('\nChoose from the run / set list below:')
[print(f'    {i}') for i in run_list]
identified_run = 0
while identified_run == 0:
    run_search = input('Enter the run / set you wish to process: ')
    isdir = [run_search[0: len(run_search)] in x for x in run_list]
    if len(np.where(isdir)[0]) == 1:
        identified_run = 1
        run = run_list[np.where(isdir)[0][0]]
        run_dir += f'{run}/'
        print(f'    Processing run {run}...')
        report_dir = f"{run_dir}report/"
    else:
        print('\n** More than one run / set found. **\n')

inj_file_list = make_file_list(run_dir, 'json')
exclude_file_list = ['peak_detection_settings.json', 'quality_control_parameters_inj.json', 'quality_control_parameters_vial.json']
[inj_file_list.remove(i) for i in exclude_file_list if i in inj_file_list]

if run_or_set == 'run':
    """If we are processing a single run, then reduce the injection json file list to a single file but keep it as a list."""
    if len(inj_file_list) > 1:
        print('\nChoose from the file list below:')
        [print(f'    {i}') for i in inj_file_list]
        identified_file = 0
        while identified_file == 0:
            inj_file_search = input('Enter the filename you wish to process: ')
            isfile = [inj_file_search[0: len(inj_file_search)] in x for x in inj_file_list]
            if len(np.where(isfile)[0]) == 1:
                identified_file = 1
                inj_file = inj_file_list[np.where(isfile)[0][0]]
                print(f'    Processing file {inj_file}...')
            else:
                print('\n** More than one file found. **\n')
    else:
        inj_file = inj_file_list[0]
    inj_file_list = [inj_file]


# -------------------- Load injection level data from json data file(s) and summarize to vial level data --------------------
inj_extra_list = ['n_high_res', 'H2O_time_slope', 'dD_time_slope', 'd18O_time_slope', 'dD_H2O_slope', 'd18O_H2O_slope',
                  'id1', 'inj_num', 'flag', 'vial_num', 'flag_reason']


dD_for_memory = np.empty(0)
total_vials = 0
for inj_file in inj_file_list:
    # read in injections file(s)
    with open(os.path.join(run_dir, inj_file), 'r') as jdf:
        inj = json.load(jdf)
        
        if 'project' in inj.keys():
            del inj['project']

    if 'timestamp' not in locals() or 'timestamp' not in globals():  # if this is the first injection file, create vial dictionary and keys
        vial = {}
        for key in inj.keys():
            globals()[key] = np.empty(0)
            if key not in inj_extra_list:
                vial[key] = {}
                vial[key]['mean'] = np.empty(0)
                vial[key]['std'] = np.empty(0)
            else:
                vial[key] = np.empty(0)
        vial['total_inj'] = np.empty(0)
        vial['n_inj'] = np.empty(0)
        vial['inj_file'] = np.empty(0)
        vial['n_high_res'] = np.empty(0)

    for key in inj.keys():  # create numpy arrays for all injection level data
        if key not in inj_extra_list:
            globals()[key] = np.asarray(inj[key]['mean'])
        else:
            globals()[key] = np.asarray(inj[key])

    dD_for_memory = np.append(dD_for_memory, inj['Delta_D_H']['mean'])

    # -------------------- summarize vial level data --------------------
    vial_set = list(set(inj['vial_num']))
    total_vials += len(vial_set)
    vials_without_injections = []
    
    for i in vial_set:
        curr_indices = np.where((vial_num == i) & (flag >= 0) & (inj_num > FIRST_INJECTIONS_TO_IGNORE))[0]
        for key in inj.keys():
            if key == 'vial_num':
                vial[key] = np.append(vial[key], i)
            elif key not in inj_extra_list:
                if len(curr_indices) == 0:
                    vials_without_injections.append(i)
                    vial[key]['mean'] = np.append(vial[key]['mean'], np.nan)
                    vial[key]['std'] = np.append(vial[key]['std'], np.nan)
                else:
                    vial[key]['mean'] = np.append(vial[key]['mean'], np.nanmean(eval(key)[curr_indices]))
                    vial[key]['std'] = np.append(vial[key]['std'], np.nanstd(eval(key)[curr_indices]))
            else:
                pass
    
    vials_without_injections = list(set(vials_without_injections))

    vial['id1'] = np.append(vial['id1'], id1[np.where(inj_num == 1)[0]])
    vial['total_inj'] = np.append(vial['total_inj'], [np.size(time[np.where((vial_num == i))[0]]) for i in vial_set])
    vial['n_inj'] = np.append(vial['n_inj'], [np.size(time[np.where((vial_num == i) & (flag == 1) & (inj_num > FIRST_INJECTIONS_TO_IGNORE))[0]]) for i in vial_set])
    vial['n_high_res'] = np.append(vial['n_high_res'], [np.sum(n_high_res[np.where((vial_num == i) & (flag == 1) & (inj_num > FIRST_INJECTIONS_TO_IGNORE))[0]]) for i in vial_set])
    vial['inj_file'] = np.append(vial['inj_file'], ([inj_file for i in vial_set]))



vial['set_vial_num'] = np.asarray(list(range(1, len(vial['id1']) + 1)))


# -------------------- Check for presence of dD etc vs Delta_D_H etc --------------------
# dD is a manual delta calculation using strengths whereas Delta_D_H is part of Picarro's calculation software. The manual delta calculations ultimately came from Picarro anyway.
if 'dD' not in vial.keys():
    vial['dD'] = vial['Delta_D_H'].copy()
    vial['d18O'] = vial['Delta_18_16'].copy()
    if instrument['O17_flag']:
        vial['d17O'] = vial['Delta_17_16'].copy()


# -------------------- Quality control vial level data --------------------
print('\n    Checking quality of vial level data.')


# Quality control vial parameters may be customized depending on the instrument or run. If you had odd backgrounds or otherwise a non-optimal
#    run, you may need to adjust the vial quality control parameters in order to salvage your data.

# List of keys that are currently expected to be in the vial quality control parameters file. If they are different from this list, archive
#    the file and make a new one.
quality_control_parameters_vial_keys = ['max_H2O_std',
                                        'max_d18O_std',
                                        'max_dD_std']
if instrument['O17_flag']:
    quality_control_parameters_vial_keys.append('max_d17O_std')

# Check for the existence of a settings file to grab existing peak detection settings
qcp_file = os.path.join(run_dir, 'quality_control_parameters_vial.json')
if os.path.isfile(qcp_file):
    with open(qcp_file, 'r') as f:
        qcp = json.load(f)

    if set(qcp.keys()).issubset(quality_control_parameters_vial_keys) is False:
        # archive existing file
        shutil.copy2(qcp_file, os.path.join(archive_dir, f"quality_control_parameters_vial_ARCHIVE_{int(os.path.getmtime(os.path.join(run_dir,'quality_control_parameters_vial.json')))}.json"))
        qcp = get_default_quality_control_parameters()

else:
    qcp = get_default_quality_control_parameters()



#    Flag 1 == good data, Flag 0 == bad data. Notes indicate reason for bad data.
print('\nScreening vial level data for poor quality...')
vial['flag'] = np.ones(len(vial['id1']))
vial['notes'] = ['' for i in vial['id1']]
for i in range(len(vial['id1'])):
    if vial['H2O']['std'][i] > qcp['max_H2O_std']:
        # vial['flag'][i] = 0  # commented out because it is usually not justifiable to exclude based on this threshold
        vial['notes'][i] += f"Vial {i+1} had high within vial H2O standard deviation (1 sigma = {round(vial['H2O']['std'][i], 0)}; threshold = {qcp['max_H2O_std']})."

    if vial['d18O']['std'][i] > qcp['max_d18O_std']:
        vial['flag'][i] = 0
        vial['notes'][i] += f"Vial {i+1} had high within vial d18O standard deviation (1 sigma = {round(vial['d18O']['std'][i], 3)}; threshold = {qcp['max_d18O_std']})."

    if vial['dD']['std'][i] > qcp['max_dD_std']:
        vial['flag'][i] = 0
        vial['notes'][i] += f"Vial {i+1} had high within vial dD standard deviation (1 sigma = {round(vial['dD']['std'][i], 3)}; threshold = {qcp['max_dD_std']})."

    if vial['vial_num'][i] in vials_without_injections:
        vial['flag'][i] = 0
        vial['notes'][i] += f"Vial {i+1} had no good injections."

    if vial['flag'][i] == 0:
        print(f"    {vial['notes'][i]}")

vial['notes'] = np.asarray(vial['notes'])


# -------------------- Remove superfluous vial keys and sort based on time --------------------
remove_from_vial_dict = ['inj_num', 'flag_reason', 'H2O_time_slope', 'dD_time_slope', 'd18O_time_slope', 'dD_H2O_slope', 'd18O_H2O_slope']
vial_extra_list = ['id1', 'flag', 'n_inj', 'notes', 'vial_num', 'inj_file', 'set_vial_num', 'total_inj', 'n_high_res']

for i in remove_from_vial_dict:
    if i in vial.keys():
        del vial[i]

vial_sort_order = np.argsort(vial['time']['mean'])
for i, j in vial.items():
    if i not in vial_extra_list:
        vial[i]['mean'] = vial[i]['mean'][vial_sort_order]
        vial[i]['std'] = vial[i]['std'][vial_sort_order]
    else:
        vial[i] = vial[i][vial_sort_order]

vial['vial_num'] = np.asarray(list(range(1, len(vial['id1']) + 1)))

vial_index_flag0 = [i for i, e in enumerate(vial['flag']) if int(e) == 0]


# -------------------- Reference Waters --------------------
#    - read in json file, make dictionaries for each reference water, and create an index list within each dictionary
with open(get_path("standards"), 'r') as refmat_file:
    refmat = json.load(refmat_file)

refmat_keys = refmat['waters'].keys()
for i in refmat_keys:
    globals()[i] = refmat['waters'][i]
    globals()[i]['index'] = np.empty(0, dtype="int16")
    if eval(i)['D17O'] is None:
        eval(i)['D17O'] = np.nan

# -------------------- Find vials containing reference waters --------------------
vial_index_all = np.asarray(vial['vial_num']) - 1
ref_wat = {'vial_index': np.empty(0, dtype="int16"),
           'id1': []}
for i in range(len(vial['id1'])):
    if vial['n_inj'][i]>0:
        for j, k in refmat['waters'].items():
            if vial['id1'][i].lower() == k['names'][0].lower() and vial['flag'][i] == 1:
                ref_wat['id1'].append(vial['id1'][i])
                k['index'] = np.append(k['index'], int(i))
                ref_wat['vial_index'] = np.append(ref_wat['vial_index'], int(i))


# -------------------- Vials containing conditioning waters --------------------
vial_index_cndtnr = [i for i in range(len(vial['id1'])) if '*' in vial['id1'][i]]
vial_index_flag0 = np.setdiff1d(vial_index_flag0, vial_index_cndtnr)


# -------------------- Vials containing sample waters --------------------
vial_index_sam = np.setdiff1d(vial_index_all, ref_wat['vial_index'])
vial_index_sam = np.setdiff1d(vial_index_sam, vial_index_flag0)
vial_index_sam = np.setdiff1d(vial_index_sam, vial_index_cndtnr)

if len(vial_index_sam) > 0:
    sam_flag = True
else:
    sam_flag = False


# -------------------- Raw Residuals --------------------
ref_wat['id1_set'] = np.asarray(list(set(ref_wat['id1'])))
ref_wat['resid_index'] = np.empty(0)
ref_wat['dD_resid_raw'] = np.empty(0)
ref_wat['d18O_resid_raw'] = np.empty(0)
if instrument['O17_flag']:
    ref_wat['d17O_resid_raw'] = np.empty(0)

for i in ref_wat['id1_set']:
    i = i.upper()
    eval(i)['dD_resid_raw'] = vial['dD']['mean'][eval(i)['index']] - np.mean(vial['dD']['mean'][eval(i)['index']])
    ref_wat['dD_resid_raw'] = np.append(ref_wat['dD_resid_raw'], eval(i)['dD_resid_raw'])
    eval(i)['d18O_resid_raw'] = vial['d18O']['mean'][eval(i)['index']] - np.mean(vial['d18O']['mean'][eval(i)['index']])
    ref_wat['d18O_resid_raw'] = np.append(ref_wat['d18O_resid_raw'], eval(i)['d18O_resid_raw'])
    ref_wat['resid_index'] = np.append(ref_wat['resid_index'], eval(i)['index'])
    if instrument['O17_flag']:
        eval(i)['d17O_resid_raw'] = vial['d17O']['mean'][eval(i)['index']] - np.mean(vial['d17O']['mean'][eval(i)['index']])
        ref_wat['d17O_resid_raw'] = np.append(ref_wat['d17O_resid_raw'], eval(i)['d17O_resid_raw'])


# -------------------- Drift Correction --------------------
if DRIFT_CORRECTION:
    print(" *** Your data have been drift corrected *** ")
    ref_wat['dDresid_fit'] = np.polyfit(ref_wat['resid_index'], ref_wat['dD_resid_raw'], 1)
    vial['dD_drift_corr_factor'] = np.asarray(ref_wat['dDresid_fit'][0] * vial['vial_num'] + ref_wat['dDresid_fit'][1])
    vial['dD_drift_corr'] = np.asarray(vial['dD']['mean'] - vial['dD_drift_corr_factor'])

    if instrument['O17_flag']:
        ref_wat['d17Oresid_fit'] = np.polyfit(ref_wat['resid_index'], ref_wat['d17O_resid_raw'], 1)
        vial['d17O_drift_corr_factor'] = np.asarray(ref_wat['d17Oresid_fit'][0] * vial['vial_num'] + ref_wat['d17Oresid_fit'][1])
        vial['d17O_drift_corr'] = np.asarray(vial['d17O']['mean'] - vial['d17O_drift_corr_factor'])

    ref_wat['d18Oresid_fit'] = np.polyfit(ref_wat['resid_index'], ref_wat['d18O_resid_raw'], 1)
    vial['d18O_drift_corr_factor'] = np.asarray(ref_wat['d18Oresid_fit'][0] * vial['vial_num'] + ref_wat['d18Oresid_fit'][1])
    vial['d18O_drift_corr'] = np.asarray(vial['d18O']['mean'] - vial['d18O_drift_corr_factor'])
else:
    print("No drift Correction was applied.")
    vial['dD_drift_corr'] = np.asarray(vial['dD']['mean'])
    vial['d18O_drift_corr'] = np.asarray(vial['d18O']['mean'])
    if instrument['O17_flag']:
        vial['d17O_drift_corr'] = np.asarray(vial['d17O']['mean'])


# -------------------- Normalize to vsmow-slap --------------------
print(f'\nThese reference waters were included in your run / set:')
[print(f'    {i}') for i in ref_wat['id1_set']]
print('Choose reference waters from the list above you wish to normalize to.')
ref_wat['chosen'] = input(f"Enter at least 2 and at most {len(ref_wat['id1_set']) - 1} (e.g. PW, WW, AW): ")
if ',' in ref_wat['chosen']:
    ref_wat['chosen'] = ref_wat['chosen'].split(',')
    ref_wat['chosen'] = [i.strip() for i in ref_wat['chosen']]
else:
    ref_wat['chosen'] = ref_wat['chosen'].split()
    ref_wat['chosen'] = [i.strip() for i in ref_wat['chosen']]
ref_wat['qaqc'] = list(np.setdiff1d(ref_wat['id1_set'], ref_wat['chosen']))

ref_wat['dDacc'] = [eval(i.upper())['dD'] for i in ref_wat['chosen']]
ref_wat['dDmeas'] = [np.nanmean(vial['dD_drift_corr'][eval(i.upper())['index']]) for i in ref_wat['chosen']]
ref_wat['dD_fit'] = np.polyfit(ref_wat['dDmeas'], ref_wat['dDacc'], 1)
vial['dD_vsmow'] = np.asarray(ref_wat['dD_fit'][0] * vial['dD_drift_corr'] + ref_wat['dD_fit'][1])

ref_wat['d18Oacc'] = [eval(i.upper())['d18O'] for i in ref_wat['chosen']]
ref_wat['d18Omeas'] = [np.nanmean(vial['d18O_drift_corr'][eval(i.upper())['index']]) for i in ref_wat['chosen']]
ref_wat['d18O_fit'] = np.polyfit(ref_wat['d18Omeas'], ref_wat['d18Oacc'], 1)
vial['d18O_vsmow'] = np.asarray(ref_wat['d18O_fit'][0] * vial['d18O_drift_corr'] + ref_wat['d18O_fit'][1])

vial['dxs_vsmow'] = np.asarray(vial['dD_vsmow'] - 8 * vial['d18O_vsmow'])

if sam_flag:
    idx = np.isfinite(vial['d18O_vsmow'][vial_index_sam[0:-1]]) & np.isfinite(vial['dD_vsmow'][vial_index_sam[0:-1]])
    dD_v_d18_fit = np.polyfit(vial['d18O_vsmow'][vial_index_sam[0:-1]][idx], vial['dD_vsmow'][vial_index_sam[0:-1]][idx], 1)
    dD_v_d18_fit_str = 'samples'
else:
    dD_v_d18_fit = np.polyfit(vial['d18O_vsmow'], vial['dD_vsmow'], 1)
    dD_v_d18_fit_str = 'analyses'

if instrument['O17_flag']:
    ref_wat['d17Oacc'] = [(np.exp(eval(i.upper())['D17O'] / 10**6 + 0.528 * np.log(eval(i.upper())['d18O'] / 1000 + 1)) - 1) * 1000 for i in ref_wat['chosen']]  # equation 9 in Schoenemann et al 2013
    ref_wat['d17Omeas'] = [np.nanmean(vial['d17O_drift_corr'][eval(i.upper())['index']]) for i in ref_wat['chosen']]
    ref_wat['d17O_fit'] = np.polyfit(ref_wat['d17Omeas'], ref_wat['d17Oacc'], 1)
    vial['d17O_vsmow'] = np.asarray(ref_wat['d17O_fit'][0] * vial['d17O_drift_corr'] + ref_wat['d17O_fit'][1])

    # derived values when measuring O17
    vial['D17O_vsmow'] = np.asarray((np.log(vial['d17O_vsmow'] / 1000 + 1) - 0.528 * np.log(vial['d18O_vsmow'] / 1000 + 1)) * 10**6)
    vial['d17O_vsmow_prime'] = np.asarray(np.log(vial['d17O_vsmow'] / 1000 + 1))
    vial['d18O_vsmow_prime'] = np.asarray(np.log(vial['d18O_vsmow'] / 1000 + 1))

    if sam_flag:
        idx = np.isfinite(vial['d18O_vsmow_prime'][vial_index_sam[0:-1]]) & np.isfinite(vial['d17O_vsmow_prime'][vial_index_sam[0:-1]])
        d17_v_d18_fit = np.polyfit(vial['d18O_vsmow_prime'][vial_index_sam[0:-1]][idx], vial['d17O_vsmow_prime'][vial_index_sam[0:-1]][idx], 1)
        d17_v_d18_fit_str = 'samples'
    else:
        d17_v_d18_fit = np.polyfit(vial['d18O_vsmow_prime'], vial['d17O_vsmow_prime'], 1)
        d17_v_d18_fit_str = 'analyses'


# -------------------- vsmow residuals --------------------
ref_wat['id1_set'] = np.asarray(list(set(ref_wat['id1'])))
ref_wat['resid_index'] = np.empty(0)
ref_wat['dD_resid_vsmow'] = np.empty(0)
ref_wat['d18O_resid_vsmow'] = np.empty(0)
if instrument['O17_flag']:
    ref_wat['d17O_resid_vsmow'] = np.empty(0)

for i in ref_wat['id1_set']:
    i = i.upper()
    eval(i)['dD_resid_vsmow'] = vial['dD_vsmow'][eval(i)['index']] - np.mean(vial['dD_vsmow'][eval(i)['index']])
    ref_wat['dD_resid_vsmow'] = np.append(ref_wat['dD_resid_vsmow'], eval(i)['dD_resid_vsmow'])
    eval(i)['d18O_resid_vsmow'] = vial['d18O_vsmow'][eval(i)['index']] - np.mean(vial['d18O_vsmow'][eval(i)['index']])
    ref_wat['d18O_resid_vsmow'] = np.append(ref_wat['d18O_resid_vsmow'], eval(i)['d18O_resid_vsmow'])
    if instrument['O17_flag']:
        eval(i)['d17O_resid_vsmow'] = vial['d17O_vsmow'][eval(i)['index']] - np.mean(vial['d17O_vsmow'][eval(i)['index']])
        ref_wat['d17O_resid_vsmow'] = np.append(ref_wat['d17O_resid_vsmow'], eval(i)['d17O_resid_vsmow'])


# -------------------- Convert desired keys in vial dictionary to numpy arrays --------------------
inj_file = np.asarray(vial['inj_file'])
id1 = np.asarray(vial['id1'])
time = np.asarray(vial['time']['mean'])
vial_num = np.asarray(vial['vial_num'])
set_vial_num = np.asarray(vial['set_vial_num'])
h2o = np.round(np.asarray(vial['H2O']['mean']))
h2o_std = np.round(np.asarray(vial['H2O']['std']))
dD_vsmow = np.round(np.asarray(vial['dD_vsmow']), 2)
dD_std = np.round(np.asarray(vial['dD']['std']), 3)
d18O_vsmow = np.round(np.asarray(vial['d18O_vsmow']), 2)
d18O_std = np.round(np.asarray(vial['d18O']['std']), 3)
dxs_vsmow = np.round(np.asarray(vial['dxs_vsmow']), 2)
n_inj = np.asarray(vial['n_inj'])
flag = np.asarray(vial['flag'])
notes = np.asarray(vial['notes'])

if instrument['O17_flag']:
    salient_vial_data_list = ['inj_file', 'id1', 'time', 'vial_num', 'set_vial_num', 'h2o', 'h2o_std', 'dD_vsmow', 'dD_std', 'd17O_vsmow', 'd17O_std', 'd18O_vsmow', 'd18O_std', 'dxs_vsmow', 'D17O_vsmow', 'n_inj', 'flag', 'notes']
    d17O_vsmow = np.round(np.asarray(vial['d17O_vsmow']), 2)
    d17O_std = np.round(np.asarray(vial['d17O']['std']), 3)
    D17O_vsmow = np.round(np.asarray(vial['D17O_vsmow']), 1)
    d17O_vsmow_prime = np.asarray(vial['d17O_vsmow_prime'])
    d18O_vsmow_prime = np.asarray(vial['d18O_vsmow_prime'])

else:
    salient_vial_data_list = ['inj_file', 'id1', 'time', 'vial_num', 'set_vial_num', 'h2o', 'h2o_std', 'dD_vsmow', 'dD_std', 'd18O_vsmow', 'd18O_std', 'dxs_vsmow', 'n_inj', 'flag', 'notes']


# -------------------- Memory -------------------
# delta = dD_for_memory
# dDmemory = {
	# 'vial_to_vial_range': np.zeros(len(vial['vial_num'])),
	# 'within_vial_range': np.zeros(len(vial['vial_num'])),
	# 'vial_memory': np.zeros(len(vial['vial_num']))}
# n=0
# for i in range(1, len(vial['vial_num'])):
	# v0_injs = int(vial['total_inj'][i-1])
	# v0 = np.asarray(range(n, n + v0_injs))
	# v1_injs = int(vial['total_inj'][i])
	# v1 = np.asarray(range(n + v0_injs, n + v0_injs + v1_injs))
	# dDmemory['vial_to_vial_range'][i] = np.abs(np.mean(np.asarray(delta)[v1][-2:]) - np.mean(np.asarray(delta)[v0][-2:]))
	# dDmemory['within_vial_range'][i] = np.abs(np.asarray(delta)[v1][0] - np.mean(np.asarray(delta)[v1][-2:]))
	# dDmemory['vial_memory'][i] = 1 - ((dDmemory['vial_to_vial_range'][i] - dDmemory['within_vial_range'][i]) / dDmemory['vial_to_vial_range'][i])
	# dDmemory['mean'] = np.mean(dDmemory['vial_memory'][np.where(dDmemory['vial_to_vial_range']>10)])
	# fig, ax = pplt.subplots()
	# ax.plot(v0, delta[v0], 'ro')
	# ax.plot(v1, delta[v1], 'bo')
	# pplt.show()
	# n+=(v0_injs)

dDmemory = memory_calc(dD_for_memory)
dDmemory['mean'] = np.mean(dDmemory['vial_memory'][np.where(dDmemory['vial_to_vial_range']>10)])


# -------------------- Get ready to make report -------------------
# copy report files
if os.path.exists(report_dir):
    shutil.move(report_dir, os.path.join(archive_dir, f"report_{int(dt.datetime.utcnow().timestamp())}"))
shutil.copytree(os.path.join(python_dir, 'report/'), report_dir)
shutil.copy2(os.path.join(python_dir, 'py_report_style.css'), report_dir)
[shutil.copy2(os.path.join(python_dir, script), os.path.join(report_dir, f"python/{script}_REPORT_COPY")) for script in python_scripts]
shutil.copy2(os.path.join(python_dir, 'py_report_style.css'), report_dir)
tray_description_file = make_file_list(run_dir, 'TrayDescription.csv')[0]
shutil.copy2(os.path.join(run_dir, tray_description_file), os.path.join(run_dir, report_dir, tray_description_file))


# -------------------- Figures --------------------
print('    Making figures...')
# remove old figures
figlist = make_file_list(os.path.join(report_dir, fig_dir), 'png')
[os.remove(os.path.join(report_dir, fig_dir, fig)) for fig in figlist]

# make new figures
fig_num = 0
captions = []
thumbcaptions = []
ref_wat['marker_colors'] = [np.random.rand(3) for i in ref_wat['id1_set']]

fig_num += 1
figname = f'Fig{str(fig_num)}_H2O_vs_vial.png'
fig, ax = pplt.subplots(figsize=(6, 3), dpi=200, tight_layout=True)
ax.errorbar(vial_num[vial_index_sam], h2o[vial_index_sam], yerr=h2o_std[vial_index_sam], label='samples', marker='.', markeredgecolor='black', markerfacecolor='black', ecolor='black', ls='None')
for j in range(len(ref_wat['id1_set'])):
    i = ref_wat['id1_set'][j]
    x = vial['vial_num'][eval(i.upper())['index']]
    y = vial['H2O']['mean'][eval(i.upper())['index']]
    ax.errorbar(x, y, yerr=vial['H2O']['std'][eval(i.upper())['index']], marker='o', markeredgecolor='black', c=ref_wat['marker_colors'][j], label=i, ls='None')
ax.set_xlabel('Vial number')
ax.set_ylabel('H2O (ppmv)')
handles, labels = ax.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax.legend(handles, labels, numpoints=1, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
pplt.savefig(os.path.join(report_dir, fig_dir, figname))
pplt.close()
caption = f'''H2O concentration in ppmv versus vial number. Mean of included injections for each vial and one standard deviation are plotted. This may show poor injections within a vial that were not culled by injection quality control tests.'''
captions.append(f'Figure {fig_num}. {caption}')
thumbcaption = 'H2O'
thumbcaptions.append(thumbcaption)

fig_num += 1
figname = f'Fig{str(fig_num)}_dDresid_vs_vial.png'
fig, ax = pplt.subplots(figsize=(6, 3), dpi=200, tight_layout=True)
for j in range(len(ref_wat['id1_set'])):
    i = ref_wat['id1_set'][j]
    x = vial['vial_num'][eval(i.upper())['index']]
    y = eval(i.upper())['dD_resid_raw']
    ax.plot(x, y, 'o', markeredgecolor='black', c=ref_wat['marker_colors'][j], markersize=4, label=i + '_raw')
for j in range(len(ref_wat['id1_set'])):
    i = ref_wat['id1_set'][j]
    x = vial['vial_num'][eval(i.upper())['index']]
    y = eval(i.upper())['dD_resid_vsmow']
    ax.plot(x, y, 'v', markeredgecolor='black', c=ref_wat['marker_colors'][j], markersize=8, label=i + '_vsmow')
ax.set_xlabel('Vial number')
ax.set_ylabel('dD residual (permil)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
pplt.savefig(os.path.join(report_dir, fig_dir, figname))
pplt.close()
caption = f'''dD residual in permil versus vial number. Residual is the individual vial mean subtracted from the mean of all vials within a reference water type. This helps identify drift across the entire run or set of runs.'''
captions.append(f'Figure {fig_num}. {caption}')
thumbcaption = 'dD_resid_raw'
thumbcaptions.append(thumbcaption)

if instrument['O17_flag']:
    fig_num += 1
    figname = f'Fig{str(fig_num)}_d17Oresid_vs_vial.png'
    fig, ax = pplt.subplots(figsize=(6, 3), dpi=200, tight_layout=True)
    for j in range(len(ref_wat['id1_set'])):
        i = ref_wat['id1_set'][j]
        x = vial['vial_num'][eval(i.upper())['index']]
        y = eval(i.upper())['d17O_resid_raw']
        ax.plot(x, y, 'o', markeredgecolor='black', c=ref_wat['marker_colors'][j], markersize=4, label=i + '_raw')
    for j in range(len(ref_wat['id1_set'])):
        i = ref_wat['id1_set'][j]
        x = vial['vial_num'][eval(i.upper())['index']]
        y = eval(i.upper())['d17O_resid_vsmow']
        ax.plot(x, y, 'v', markeredgecolor='black', c=ref_wat['marker_colors'][j], markersize=8, label=i + '_vsmow')
    ax.set_xlabel('Vial number')
    ax.set_ylabel('d17O residual (permil)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
    pplt.savefig(os.path.join(report_dir, fig_dir, figname))
    pplt.close()
    caption = f'''d17O residual in permil versus vial number. Residual is the individual vial mean subtracted from the mean of all vials within a reference water type. This helps identify drift across the entire run or set of runs.'''
    captions.append(f'Figure {fig_num}. {caption}')
    thumbcaption = 'd17O_resid_raw'
    thumbcaptions.append(thumbcaption)

fig_num += 1
figname = f'Fig{str(fig_num)}_d18Oresid_vs_vial.png'
fig, ax = pplt.subplots(figsize=(6, 3), dpi=200, tight_layout=True)
for j in range(len(ref_wat['id1_set'])):
    i = ref_wat['id1_set'][j]
    x = vial['vial_num'][eval(i.upper())['index']]
    y = eval(i.upper())['d18O_resid_raw']
    ax.plot(x, y, 'o', markeredgecolor='black', c=ref_wat['marker_colors'][j], markersize=4, label=i + '_raw')
for j in range(len(ref_wat['id1_set'])):
    i = ref_wat['id1_set'][j]
    x = vial['vial_num'][eval(i.upper())['index']]
    y = eval(i.upper())['d18O_resid_vsmow']
    ax.plot(x, y, 'v', markeredgecolor='black', c=ref_wat['marker_colors'][j], markersize=8, label=i + '_vsmow')
ax.set_xlabel('Vial number')
ax.set_ylabel('d18O residual (permil)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
pplt.savefig(os.path.join(report_dir, fig_dir, figname))
pplt.close()
caption = f'''d18O residual in permil versus vial number. Residual is the individual
              vial mean subtracted from the mean of all vials within a reference water type. This helps
              identify drift across the entire run or set of runs. The residual is calculated using the raw
              measured instrument values. The standard deviation of all non-drift-corrected d18O residual values
              on this figure is {str(round(np.std(ref_wat['d18O_resid_raw']), 3))}. The standard deviation of
              all drift-corrected d18O residual values is {str(round(np.std(ref_wat['d18O_resid_vsmow']), 3))}.'''
captions.append(f'Figure {fig_num}. {caption}')
thumbcaption = 'd18O_resid_raw'
thumbcaptions.append(thumbcaption)

fig_num += 1
figname = f'Fig{str(fig_num)}_d18O_vs_dD.png'
fig, ax = pplt.subplots(figsize=(6, 3), dpi=200, tight_layout=True)
ax.plot(d18O_vsmow[vial_index_sam], dD_vsmow[vial_index_sam], 'k.', label='Samples')
for j in range(len(ref_wat['id1_set'])):
    i = ref_wat['id1_set'][j]
    x = d18O_vsmow[eval(i.upper())['index']]
    y = dD_vsmow[eval(i.upper())['index']]
    ax.plot(x, y, 'o', markeredgecolor='black', c=ref_wat['marker_colors'][j], markersize=8, label=i)
ax.set_xlabel('d18O vs VSMOW (permil)')
ax.set_ylabel('dD vs VSMOW (permil)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
pplt.savefig(os.path.join(report_dir, fig_dir, figname))
pplt.close()
caption = f'''dD vs d18O. The slope of all {dD_v_d18_fit_str} in this plot is {str(np.round(dD_v_d18_fit[0], 1))}.'''
captions.append(f'Figure {fig_num}. {caption}')
thumbcaption = 'dD vs d18O'
thumbcaptions.append(thumbcaption)

if instrument['O17_flag']:
    fig_num += 1
    figname = f'Fig{fig_num}_d17O_vs_d18O.png'
    fig, ax = pplt.subplots(figsize=(6, 3), dpi=200, tight_layout=True)
    ax.plot(d18O_vsmow_prime[vial_index_sam], d17O_vsmow_prime[vial_index_sam], 'k.', label='Samples')
    for j in range(len(ref_wat['id1_set'])):
        i = ref_wat['id1_set'][j]
        x = d18O_vsmow_prime[eval(i.upper())['index']]
        y = d17O_vsmow_prime[eval(i.upper())['index']]
        ax.plot(x, y, 'o', markeredgecolor='black', c=ref_wat['marker_colors'][j], markersize=8, label=i)
    ax.set_xlabel('d18Oprime vs VSMOW')
    ax.set_ylabel('d17Oprime vs VSMOW')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
    pplt.savefig(os.path.join(report_dir, fig_dir, figname))
    pplt.close()
    caption = f'''d17Oprime vs d18Oprime. The slope of all {d17_v_d18_fit_str} in this plot is {str(np.round(d17_v_d18_fit[0], 4))}.'''
    captions.append(f'Figure {fig_num}. {caption}')
    thumbcaption = 'd17O vs d18O'
    thumbcaptions.append(thumbcaption)

figlist = make_file_list(os.path.join(report_dir, fig_dir), 'png')
figlist = natsorted(figlist)


# -------------------- export summary data file --------------------
print(f'\n    Creating summary data file.')
summary_data_filename = f"{instrument['name'].lower()}_summary_data.csv"
summary_data_file = os.path.join(report_dir, 'data/', summary_data_filename)

summary_file_headers = salient_vial_data_list
data_to_write = str([f'{i}[ii]' for i in salient_vial_data_list]).replace("'", '')
with open(summary_data_file, 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    datawriter.writerow(summary_file_headers)
    for ii in ref_wat['vial_index']:
        datawriter.writerow(eval(data_to_write))
    datawriter.writerow('\n')
    for ii in vial_index_sam:
        datawriter.writerow(eval(data_to_write))
    datawriter.writerow('\n')
    for ii in vial_index_flag0:
        datawriter.writerow(eval(data_to_write))


# -------------------- make html summary report --------------------
ref_wat_block_str_1 = str([f"""<tr>
                             <td>{i}</td>
                             <td>{eval(i.upper())['dD']}</td>
                             <td>{eval(i.upper())['d18O']}</td>
                             <td>{eval(i.upper())['D17O']}</td>
                             <td>normalization to VSMOW-SLAP</td>
                         </tr>""" for i in ref_wat['chosen']]).replace("[", "").replace("'", "").replace("]", "").replace(", ", "").replace("\\n", "")
ref_wat_block_str_2 = str([f"""<tr>
                             <td>{i}</td>
                             <td>{eval(i.upper())['dD']}</td>
                             <td>{eval(i.upper())['d18O']}</td>
                             <td>{eval(i.upper())['D17O']}</td>
                             <td>quality assurance / quality control</td>
                         </tr>""" for i in ref_wat['qaqc']]).replace("[", "").replace("'", "").replace("]", "").replace(", ", "").replace("\\n", "")

ref_wat_block = ref_wat_block_str_1 + ref_wat_block_str_2

data_quality_block_str_1 = str([f"""<tr><td>{i}</td><td>dD</td>
                                  <td>{round(np.std(dD_vsmow[eval(i.upper())['index']]) * 2, 3)}</td>
                                  <td>{round(np.mean(dD_vsmow[eval(i.upper())['index']])-eval(i.upper())['dD'], 3)}</td></tr>
                              <tr><td>{i}</td><td>d18O</td>
                                  <td>{round(np.std(d18O_vsmow[eval(i.upper())['index']]) * 2, 3)}</td>
                                  <td>{round(np.mean(d18O_vsmow[eval(i.upper())['index']])-eval(i.upper())['d18O'], 3)}</td></tr>
                           """ for i in ref_wat['qaqc']]).replace("[", "").replace("'", "").replace("]", "").replace(", ", "").replace("\\n", "")

if instrument['O17_flag']:
    data_quality_block_str_2 = str([f"""<tr><td>{i}</td><td>D17O</td>
                                      <td>{round(np.std(D17O_vsmow[eval(i.upper())['index']]) * 2, 1)}</td>
                                      <td>{round(np.mean(D17O_vsmow[eval(i.upper())['index']])-eval(i.upper())['D17O'], 3)}</td></tr>
                               """ for i in ref_wat['qaqc']]).replace("[", "").replace("'", "").replace("]", "").replace(", ", "").replace("\\n", "")

if instrument['O17_flag']:
    data_quality_block = data_quality_block_str_1 + data_quality_block_str_2
else:
    data_quality_block = data_quality_block_str_1

log_summary_page = os.path.join(report_dir, 'report.html')
fhp = open(log_summary_page, 'w')
header = f"""
    <!DOCTYPE html>
    <html lang="en">
        <!-- py by Andy Schauer -->
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <link rel="stylesheet" type="text/css" href="py_report_style.css">
        <title>{instrument['name']} Report</title>
    </head>
    <body id="top">\n

    <div class="created-date">Created - {str(dt.datetime.now())}</div>

    <h2>{instrument['name']} Report - {dt.datetime.utcfromtimestamp( inj['time']['mean'][0]).strftime('%Y-%m-%d')} to {dt.datetime.utcfromtimestamp( inj['time']['mean'][-1]).strftime('%Y-%m-%d')}</h2>

    <h2>Introduction</h2>
    <div class="text-indent">
        <p>This report is meant to be a stand-alone collection of methods,
        data, scripts, and notes related to water isotopic
        analysis using {instrument['name']} (a Picarro {instrument['model']}).
        Analytical details are described in Schauer et al. (2016) as well as the lab
        overview and method (see <a href="#refs">References</a> below).</p>

        <p>The data and python scripts used to generate this page are linked
        and described in the <a href="#refs">References</a> section below. If you wish
        to save this report, copy and paste (or download) the entire 'report' directory to a
        place of your choosing and all html, images, data files, and
        python scripts will be saved. <strong><a href="report.zip">Save a copy if you are finished
        analyzing your samples</a></strong>.</p>
    </div>

    <h2>My data</h2>
    <div class="text-indent">
        <p>All this technical stuff is fine and all but where are <strong><a href="data/{summary_data_filename}">my data</a></strong>?
        This summary file contains sample IDs, times of analysis, water vapor concentrations, isotope delta values normalized to VSMOW-SLAP,
        number of injections, flags, and notes. Each section of data is separated by an empty row. The first section of data are the trusted
        reference waters; the second section of data are trusted samples; the third section of data are untrusted. Under the
        "flag" heading, "1" indicates good, trusted data while "0" indicates poor quality data that should probably be distrusted. Up to you
        if you want to use it. Untrusted data are given the reason for distrust. If you are done analyzing samples, please save a copy of
        the entire report directory elsewhere, not just a copy of your data file.</p>
    </div>

    <h2>Data operations</h2>
    <div class="text-indent">
        <p>A suite of mathmatical operations were completed on these data prior to claiming they are final. High resolution one-second data are
        summarized to provide injection level data. The injection level data are then summarized to provide individual vial level data. This
        particular run or set of runs was set up to complete <strong>{np.max(inj['inj_num'])} injections per vial.</strong> The first
        <strong>{FIRST_INJECTIONS_TO_IGNORE} injections of each vial were ignored</strong> as a simplistic way of dealing with carry-over or memory. The vial is
        considered the sample and a single replicate. Vial level isotopic data are drift corrected - this is a correction based on all reference waters
        and is assumed to be linear with time. Drift corrected vial level isotopic data are then normalized to the VSMOW-SLAP scale using accepted
        values of at least two of the included reference waters. A correction for water vapor concentration is not built into this scheme at present
        given the consistency of injection sizes and the reasonably sound correction that already exists within the picarro software.
        </p>
    </div>

    <h2>Run / set inventory</h2>
    <div class="text-indent">
        <table>
            <tr><td>Total number of runs</td><td>{len(inj_file_list)}</td></tr>
            <tr><td>Total number of analyses</td><td>{len(vial['id1'])}</td></tr>
            <tr><td>Total number of standards analyzed</td><td>{len(ref_wat['id1'])}</td></tr>
            <tr><td>Total number of samples analyzed</td><td>{len(vial_index_sam)}</td></tr>
            <tr><td><br></td></tr>
            <tr><td>Number of conditioners</a></td><td>{len(vial_index_cndtnr)}</td></tr>
            <tr><td>Number of <a href="#excluded">excluded analyses</a></td><td>{len(vial_index_flag0)}</td></tr>
            <tr><td><br></td></tr>
            <tr><td><a href="{tray_description_file}">Original Tray Description File</a></td></tr>
        </table>
    </div>

    <h2>Reference waters</h2>
    <div class="text-indent"><p>The reference waters and their accepted values, normalized to the VSMOW-SLAP scale, included in this run / set are:</p>
        <table>
            <tr><th>Reference<br>water</th><th>dD<br>accepted<br>(permil)</th><th>d18O<br>accepted<br>(permil)</th><th>D17O<br>accepted<br>(permeg)</th><th>Purpose</th></tr>
            {ref_wat_block}
        </table>
    </div>

    <h2>Data quality</h2>
    <div class="text-indent"><p>Precision and accuracy estimates are derived from reference water {eval(ref_wat['qaqc'][0].upper())['names']}. Precision is
        <strong>two standard deviations</strong> over all replicates of the quality control reference water. Accuracy is the difference of the mean of all replicates of the
        quality control reference water from the accepted value.</p>
        <table>
            <tr><th> </th><th> </th><th>Precision</th><th>Accuracy</th></tr>
            {data_quality_block}
        </table>
    </div>
    <div class="text-indent"><h3>Memory / Carry-over</h3>
    <p>This memory, or carry-over calculation attempts to estimate the proportion of water from the current injection
    that is presently being measured. Some of the previously injected water is still present in the system. In this
    calculation, a memory estimate of 0% would imply no carry-over at all and indicate that the only water presently
    being measured is from the current injection. Memory estimates vary depending on the delta used. d18O tends to provide
    a lower memory estimate compared with dD. As such, my default is to use the hydrogen isotope composition as a memory
    estimate. Furthermore, this estimate seems to be only valid when the range is sufficiently large. This makes intuitive
    sense because as the two waters become more similar, a memory estimate becomes more challenging to measure, and indeed,
    less meaningful. I have placed a 10 permil difference threshold to cut off the erroneous estimates from those waters
    with a less than 10 permil difference. Mathmatically, as the difference between the two waters goes to zero, the
    proportion goes to zero.</p>

    <p>The memory estimate from this run, based on dD, is <strong>{round(dDmemory['mean']*100, 1)}%</strong>.</p>
    </div>

    <h2>Figure thumbnails</h2>"""

fhp.write(header)

for fig, caption in zip(figlist, thumbcaptions):
    fig_as_thumb(fhp, os.path.join(fig_dir, fig), caption)

fhp.write('\n<div class="clear-both"></div><hr>\n')

for fig, caption in zip(figlist, captions):
    fig_in_html(fhp, os.path.join(fig_dir, fig), caption)

if len(vial_index_flag0) == 0:
    excluded_analyses_block = '<tr><td>no excluded analyses</td></tr>'
else:
    excluded_analyses_block = str([f"<tr><td>{vial['id1'][i]}</td><td>{vial['notes'][i]}</td></tr>" for i in vial_index_flag0]).replace("[", "").replace("'", "").replace("]", "").replace(", ", "")

excluded_analysis = f"""
    <h2 id="excluded">Excluded analyses</h2>
    <div class="text-indent">
        <p>This is a list of analyses that have been excluded from further data processing showing
        the sample ID and the reason for exclusion.</p>

        <table>
            <tr><th>Sample ID</th><th>Reason for excluding</th></tr>
            {excluded_analyses_block}
        </table>
    </div>
    """
fhp.write(excluded_analysis)

python_scripts_block = str([f'<li><a href="python/{key}_REPORT_COPY">{key}</a> - {value}</li>' for key, value in python_scripts.items()]).replace("[", "").replace("'", "").replace("]", "").replace(", ", "")

if instrument['O17_flag']:
    overview = 'water-dD-d17O-d18O.php'
    method = 'phoenix.php'
else:
    overview = 'water-dD-d18O.php'
    method = 'abel.php'

footer = f"""
    <h2 id="refs">References</h2>
    <div class="references">
    <ul>
        <li><strong>Schauer AJ, Schoenemann SW, Steig EJ</strong>. (2016). Routine high-precision analysis of triple water-isotope ratios using cavity ring-down spectroscopy. <em>Rapid Communications in Mass Spectrometry</em>: 30, 2059–2069. doi: <a href="https://doi.org/10.1002/rcm.7682">10.1002/rcm.7682</a>.</li>
        <li>Python scripts - modification date:
            <ul>
                {python_scripts_block}
            </ul>
        <li></li>
        <li><a href="https://github.com/andyschauer/water">github repository</a></li>
        <li>Data files - <a href="data/{summary_data_filename}">{summary_data_filename}</a></li>
        <li><a href="https://isolab.ess.washington.edu/laboratory/{overview}">IsoLab's water isotope analysis overiew.</a></li>
        <li><a href="https://isolab.ess.washington.edu/SOPs/{method}">IsoLab's water isotope analysis method.</a></li>
        <li><a href="report.zip">Zip file of entire report directory.</a></strong>.</li>
    </ul>
    </div>
    </body></html>"""
fhp.write(footer)
fhp.close()
webbrowser.open(log_summary_page)


# -------------------- make zip of summary report --------------------
shutil.make_archive('report', 'zip', os.path.join(report_dir))
shutil.move('report.zip', os.path.join(report_dir, 'report.zip'))



# -------------------- write data to log file --------------------
print(f'\n    Write data to log file.')
log_filename = f"{instrument['name'].lower()}_run_log.csv"
log_file = os.path.join(project_dir, log_filename)
memory_notes = inj_file_list[0]
log_file_headers = ['inj_file', 'start_time', 'memory', 'notes']
log_data = [inj_file[0], time[0], dDmemory['mean'], memory_notes]
with open(log_file, 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    datawriter.writerow(log_file_headers)
    datawriter.writerow(log_data)



