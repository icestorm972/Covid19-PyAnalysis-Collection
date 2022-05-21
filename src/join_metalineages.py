# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:58:13 2022

@author: David
"""

import os
from datetime import datetime as dt
import pandas as pd


# Based on Cornelius Roemer bash scripts
# https://github.com/corneliusroemer/desh-data/tree/main/scripts

# Data source 1:
# lineages.csv.xz => https://raw.githubusercontent.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland/master/SARS-CoV-2-Entwicklungslinien_Deutschland.csv.xz
# use cols: IMS_ID,lineage,scorpio_call

# Data source 2:
# metadata.csv.xz => https://raw.githubusercontent.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland/master/SARS-CoV-2-Sequenzdaten_Deutschland.csv.xz
# use cols: IMS_ID,DATE_DRAW,SEQ_REASON,PROCESSING_DATE,SENDING_LAB_PC,SEQUENCING_LAB_PC

# Note:
# join by IMS_ID col

DATE = dt.now().strftime('%Y-%m-%d')

DATA_IN = {
    'lineages': 'https://raw.githubusercontent.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland/master/SARS-CoV-2-Entwicklungslinien_Deutschland.csv.xz',
    'metadata': 'https://raw.githubusercontent.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland/master/SARS-CoV-2-Sequenzdaten_Deutschland.csv.xz'
    }

COLS = {
    'lineages': ['IMS_ID', 'lineage', 'scorpio_call'],
    'metadata': ['IMS_ID', 'DATE_DRAW', 'SEQ_REASON', 'PROCESSING_DATE', 'SENDING_LAB_PC', 'SEQUENCING_LAB_PC']
    }

# Store in "Documents" folder. Should work in Windows & Ubuntu
FPATH = os.path.expanduser(f'~{os.sep}Documents')
FNAME_OUT = f'{FPATH}{os.sep}meta_lineages_{DATE}.csv'

# Download and read CSV files from RKI Github
lineages_data = pd.read_csv(DATA_IN["lineages"], usecols=COLS["lineages"], index_col='IMS_ID')
metadata_data = pd.read_csv(DATA_IN["metadata"], usecols=COLS["metadata"], index_col='IMS_ID')

# join both tables
meta_lineages = metadata_data.join(lineages_data, how='inner')

# some cleanup
meta_lineages.lineage.fillna('Unassigned', inplace=True)
meta_lineages.scorpio_call.fillna('', inplace=True)
meta_lineages.sort_values(['DATE_DRAW', 'lineage'], inplace=True)

# save resulting table as csv
meta_lineages.to_csv(FNAME_OUT)
