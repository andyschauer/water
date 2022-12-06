#!/usr/bin/env python3
"""
This script reads in the hdf5 file created by picarro_h5.py if the data set exclusively contains discrete samples (injections).
This script will find each injection and reduce the h5 level data (~ 1 Hz data) to injection level summaries. A tray
description file is required. This script effectively creates Picarro's Coordinator output, albeit in json format
and with all of the data fields present that are in the original h5 files.
"""

__author__ = "Andy Schauer"
__copyright__ = "Copyright 2022, Andy Schauer"
__license__ = "Apache 2.0"
__version__ = "1.1"
__email__ = "aschauer@uw.edu"


# -------------------- imports --------------------
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN  # , INLINE
import datetime as dt
import h5py
import json
from natsort import natsorted
import numpy as np
import os
from picarro_lib import *
import shutil
import time as t
import webbrowser


# -------------------- get instrument information --------------------
""" Get specific picarro instrument whose data is being processed as well as some associated information. Populate
this function with your own instrument(s). The function get_instrument() is located in picarro_lib.py."""
instrument, ref_ratios, inj_peak, inj_quality, vial_quality = get_instrument()


# -------------------- directory setup --------------------
"""Make your life easier with this section. I think the only common path segment we all have is the h5_dir, but the
rest will be completely different depending on how you organize yourself."""
python_dir = '/home/aschauer/python/pybob/'
project_dir = f"/home/aschauer/projects/{instrument['name'].lower()}/"
run_dir = os.path.join(project_dir, 'runs/')


# -------------------- python scripts --------------------
python_scripts = {'picarro_lib.py': '', 'picarro_h5.py': '', 'picarro_inj.py': '', 'picarro_vial_calibrate.py': ""}
python_scripts = {key: (t.strftime('%Y-%m-%d %H:%M:%S', t.localtime(os.path.getmtime(f'{python_dir}{key}')))) for key, value in python_scripts.items()}


# -------------------- identify run --------------------
run_list = natsorted(os.listdir(run_dir))
print('\nChoose from the run list below:')
[print(f'    {i}') for i in run_list]
identified_run = 0
while identified_run == 0:
    run_search = input('Enter the run you wish to process: ')
    isdir = [run_search[0: len(run_search)] in x for x in run_list]
    if len(np.where(isdir)[0]) == 1:
        identified_run = 1
        run = run_list[np.where(isdir)[0][0]]
        run_dir += f'{run}/'
        print(f'    Processing run {run}...')
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
            print(f'    Processing file {hdf5_file}...')
        else:
            print('\n** More than one file found. **\n')
else:
    hdf5_file = hdf5_list[0]

np.seterr(all='raise')


# -------------------- read in data from hdf5 file --------------------
with h5py.File(f'{run_dir}{hdf5_file}', 'r') as hf:
    datasets = list(hf.keys())
    for data in datasets:
        globals()[data] = np.asarray(hf[data])


# -------------------- Check for presence of dD etc vs Delta_D_H etc --------------------
"""dD is a DIY delta calculation using strengths whereas Delta_D_H is picarros calculation. They
   are nominally the same."""
if 'dD' not in datasets:
    dD = Delta_D_H.copy()
    datasets.append('dD')
    d18O = Delta_18_16.copy()
    datasets.append('d18O')
    if instrument['O17_flag']:
        d17O = Delta_17_16.copy()
        datasets.append('d17O')


# -------------------- Find individual injections --------------------
print('\n    Finding injection peaks.')

H2O_THRESHOLD = 3000
dH2O_dT_threshold = 300
dH2O_dT2_threshold = 30
kernel_size = 3

time_diff = np.diff(time)
time_diff = np.append(time_diff, np.mean(time_diff))

kernel = np.ones(kernel_size) / kernel_size
H2O_convolved = np.convolve(H2O, kernel, mode='same')

H2O_diff = np.diff(H2O_convolved)
H2O_diff = np.append(H2O_diff, H2O_diff[-1])  # duplicate last value so length of array is the same as the original length

dH2O_dT = H2O_diff / time_diff
dH2O_dT_convolved = np.convolve(dH2O_dT, kernel, mode='same')

dH2O_dT2 = np.diff(dH2O_dT_convolved)
dH2O_dT2 = np.append(dH2O_dT2, dH2O_dT2[-1])
dH2O_dT2_convolved = np.convolve(dH2O_dT2, kernel, mode='same')

pks = np.where(np.logical_and(np.logical_and(abs(dH2O_dT2_convolved) < dH2O_dT2_threshold, H2O > H2O_THRESHOLD),
                              np.logical_and(dH2O_dT_convolved > 0, dH2O_dT_convolved < dH2O_dT_threshold)))[0]

adi = np.arange(0, len(H2O), 1)
pks_end = np.asarray(np.diff(pks) > 10).nonzero()[0]
end_of_each_inj = adi[pks[pks_end]]
end_of_each_inj = np.append(end_of_each_inj, pks[-1])
pks_start = pks_end + 1
start_of_each_inj = adi[pks[pks_start]]
start_of_each_inj = np.insert(start_of_each_inj, 0, pks[0])


# -------------------- Summarize Injection Data --------------------
print('\n    Summarizing injection data.')

inj = {}
for data in datasets:
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
        except FloatingPointError as error:
            print(f'    **** {error} ****    ')
            inj[data]['mean'][i] = np.nan
            inj[data]['std'][i] = np.nan

inj['n_high_res'] = np.asarray([len(H2O[i:j]) for i, j in zip(start_of_each_inj, end_of_each_inj)])

inj_gdi = [adi[i:j] for i, j in zip(start_of_each_inj, end_of_each_inj)]
gdi = np.concatenate(inj_gdi).ravel()


# -------------------- tray description file --------------------
tray_descriptions = make_file_list(os.path.join(project_dir, 'TrayDescriptions/'), '.csv')
tray_description_file = [i for i in tray_descriptions if hdf5_file[0:8] in i]
shutil.copy2(os.path.join(project_dir, 'TrayDescriptions/', tray_description_file[0]), os.path.join(project_dir, run_dir, tray_description_file[0]))
tray_headers, tray_data = read_file(os.path.join(project_dir, 'TrayDescriptions/', tray_description_file[0]), ',')
vial = np.asarray(tray_data['Vial'], dtype=int)


# -------------------- compare detected injections with expected injections from tray description --------------------
detected_inj = len(inj['H2O']['mean'])
expected_inj = np.asarray(tray_data['Injections'], dtype=int)

if np.sum(expected_inj) != detected_inj:
    # -------------------- If actual number of injections do not match expected number of injections --------------------
    print(f"\n** Expecting {np.sum(expected_inj)} injections. Found {detected_inj} injections. Look carefully at the figure to assess what went wrong. **")
    t.sleep(2)

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
    # -------------------- remove mismatch file if it exists --------------------
    if os.path.isfile(os.path.join(run_dir, f'{hdf5_file[0:-5]}_Injection_Accounting_Problem.html')):
        os.remove(os.path.join(run_dir, f'{hdf5_file[0:-5]}_Injection_Accounting_Problem.html'))

    # -------------------- put sample IDs and vial inj number into inj dictionary --------------------
    inj['project'] = [[i] * j for i, j in zip(tray_data['Project'], expected_inj)]
    inj['project'] = [x for xs in inj['project'] for x in xs]
    inj['id1'] = [[i] * j for i, j in zip(tray_data['Identifier1'], expected_inj)]
    inj['id1'] = [x for xs in inj['id1'] for x in xs]
    inj['vial_num'] = [[i] * j for i, j in zip(vial, expected_inj)]
    inj['vial_num'] = [x for xs in inj['vial_num'] for x in xs]
    inj['inj_num'] = [list(range(1, ei + 1)) for ei in expected_inj]
    inj['inj_num'] = [x for xs in inj['inj_num'] for x in xs]

    # -------------------- quality control injections --------------------
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
                       deviation greatern than {inj_quality['max_dD_std']} permil are <a href="#flagged_injections">flagged</a>."""

    fig4 = figure(width=1100, height=700, x_axis_label="data index", y_axis_label="d18O raw (permil)", tools="pan, box_zoom, reset, save", active_drag="box_zoom")
    fig4.circle(adi, d18O, color="black", legend_label="All data", size=2)
    fig4.circle(gdi, d18O[gdi], color="green", legend_label="Good data", size=6)
    fig4.circle(fdi, d18O[fdi], color="yellow", legend_label="Flagged data", size=6)
    fig4_caption = f"""Figure 3. Oxygen-18 isotope composition (d18O or delta 18 Oh). Injections with a standard
                       deviation greatern than {inj_quality['max_dD_std']} permil are <a href="#flagged_injections">flagged</a>."""

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
    for data in datasets:
        inj_export[data]['mean'] = inj[data]['mean'].tolist()
        inj_export[data]['std'] = inj[data]['std'].tolist()
    inj_export['n_high_res'] = inj['n_high_res'].tolist()
    inj_export['flag'] = inj['flag'].tolist()
    inj_export['vial_num'] = [int(i) for i in inj['vial_num']]

    with open(os.path.join(run_dir, f'{hdf5_file[0:-5]}_injections.json'), 'w') as fp:
        json.dump(inj_export, fp)
