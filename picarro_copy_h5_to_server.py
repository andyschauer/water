"""
Unzip and copy h5 files produced by any Picarro instrument
to a data server. This script is designed to be placed on the
picarro instrument itself and is necessarily written in python
2.7 because as of 2022, all Picarros still run on and with
python 2.7.

The dependencies for the present script are os, re, and zipfile,
which should already be on your picarro.

Ideally, Windows scheduler is used to run this script periodically
(e.g. once per day).
"""

__author__ = "Andy Schauer"
__copyright__ = "Copyright 2024, Andy Schauer"
__license__ = "Apache 2.0"
__version__ = "1.1"
__email__ = "aschauer@uw.edu"


# ----- IMPORTS -----
import os
import re
import zipfile


# ----- USER SETUP -----
INSTRUMENT = '[instrument_name]'
dest_dir = '/path/to/data/server/' + INSTRUMENT + '/h5'
transfer_list = 'C:/' + INSTRUMENT + '/transferred_h5_files.txt'


# ----- SETUP -----
source_dir = 'C:/Picarro/G2000/Log/Archive/DataLog_Private'


# ----- MAIN -----
with open(transfer_list, "a+") as log:
    curr_log = log.read()

    for dirpath, subdirs, files in os.walk(source_dir):
        for file in files:
            if file not in curr_log:
                try:
                    with zipfile.ZipFile(os.path.join(dirpath, file), 'r') as zip_ref:
                        print 'unzipping ' + file
                        zip_ref.extractall(dest_dir)
                        log.write(file + '\n')

                except:
                    print 'Bad Zip File - ' + file
                    log.write('Bad Zip File - ' + file + '\n')
