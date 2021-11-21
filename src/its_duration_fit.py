# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:43:11 2021

@author: David
"""

from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator 
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np
import scipy.optimize as sopt
import scipy.signal as sig


OUTPUT_FOLDER = '..\\output\\'
OUTPUT_FILE = 'its-liegezeiten.csv'

DOWNLOAD_URL = 'https://diviexchange.blob.core.windows.net/%24web/zeitreihe-bundeslaender.csv'

BL_LIST = [
    'DE', 'BW', 'BY', 'BE', 'BB', 'HB',
    'HH', 'HE', 'MV', 'NI', 'NW', 'RP',
    'SL', 'SN', 'ST', 'SH', 'TH'
    ]

BL_LUT = {
    'BW': 'Baden-Württemberg',
    'BY': 'Bayern',
    'BE': 'Berlin',
    'BB': 'Brandenburg',
    'HB': 'Bremen',
    'HH': 'Hamburg',
    'HE': 'Hessen',
    'MV': 'Mecklenburg-Vorpommern',
    'NI': 'Niedersachsen',
    'NW': 'Nordrhein-Westfalen',
    'RP': 'Rheinland-Pfalz',
    'SL': 'Saarland',
    'SN': 'Sachsen',
    'ST': 'Sachsen-Anhalt',
    'SH': 'Schleswig-Holstein',
    'TH': 'Thüringen',
    'DE': 'Deutschland'
}

ITS_BL_LUT = {
    'BADEN_WUERTTEMBERG': 'BW',
    'BAYERN': 'BY',
    'BERLIN': 'BE',
    'BRANDENBURG': 'BB',
    'BREMEN': 'HB',
    'HAMBURG': 'HH',
    'HESSEN': 'HE',
    'MECKLENBURG_VORPOMMERN': 'MV',
    'NIEDERSACHSEN': 'NI',
    'NORDRHEIN_WESTFALEN': 'NW',
    'RHEINLAND_PFALZ': 'RP',
    'SAARLAND': 'SL',
    'SACHSEN': 'SN',
    'SACHSEN_ANHALT': 'ST',
    'SCHLESWIG_HOLSTEIN': 'SH',
    'THUERINGEN': 'TH'
}

KERNEL_LEN = 25
EXPECTED_DAYS = 12
CONST_LEN = 7


divi_date_parser = lambda x: datetime.strptime(x[:10], '%Y-%m-%d')

all_data = pd.read_csv(DOWNLOAD_URL, 
                       parse_dates = ['Datum'],
                       date_parser = divi_date_parser, 
                       usecols = ['Datum', 'Bundesland', 
                                  'Aktuelle_COVID_Faelle_ITS', 
                                  'faelle_covid_erstaufnahmen'] )

all_data_af = all_data.pivot(index='Datum', 
                             columns='Bundesland', 
                             values='Aktuelle_COVID_Faelle_ITS').rename(columns=ITS_BL_LUT)
all_data_af['DE'] = all_data_af.sum(axis=1, skipna=False)

all_data_ea = all_data.pivot(index='Datum', 
                             columns='Bundesland', 
                             values='faelle_covid_erstaufnahmen').rename(columns=ITS_BL_LUT)
all_data_ea['DE'] = all_data_ea.sum(axis=1, skipna=False)

liegezeiten = {}

for bl in BL_LIST:
    
    data_ea = all_data_ea[bl].dropna()
    data_af = all_data_af[bl].dropna()
    
    start_date = data_ea.index[0] + pd.DateOffset(days=KERNEL_LEN-1)
    end_date = data_ea.index[-1]
    data_af = data_af.loc[start_date:].iloc[:-1]
    
    weeklst = data_af.index.isocalendar().apply(lambda x: '{:d}-{:02d}'.format(x[0], x[1]), axis=1)
    uwl =  weeklst.unique()
    num_var = uwl.size
    
    def errfnc(x0, daf, dea):        
        weeklst = daf.index.isocalendar().apply(lambda x: '{:d}-{:02d}'.format(x[0], x[1]), axis=1)
        uwl =  weeklst.unique()
        num_var = uwl.size
        
        scale_m = np.zeros((daf.size, dea.size))
        
        for i in range(num_var):
            kernlst = np.zeros((KERNEL_LEN,))
            
            sa = x0[i]
            sb = KERNEL_LEN-1
                        
            sai = int(np.ceil(sa))
            sbi = int(np.floor(sb))
            sar = sai-sa
            sbr = sb-sbi
            
            if sai>0:
                kernlst[sai-1] = sar
            if sbi<KERNEL_LEN-1:
                kernlst[sbi+1] = sbr
            kernlst[sai:sbi+1] = 1.0
            
            for j in np.nonzero((weeklst==uwl[i]).to_numpy())[0]:
                scale_m[j, j:j+KERNEL_LEN] = kernlst
            
        daf_model = scale_m @ dea
        return daf.to_numpy() - daf_model
     
        
    print('{:s}...'.format(BL_LUT[bl]))
    test_res = sopt.least_squares(
       errfnc, 
       np.full((num_var,), EXPECTED_DAYS),
       method = 'trf',
       bounds = (0, KERNEL_LEN-1),
       jac = '3-point',
       args = (data_af, data_ea))
    
    plt.figure(figsize=(0.6*16, 0.6*9))
    plt.plot(data_af, 'k-', label='gemeldete aktive Fälle')
    plt.plot(data_af + errfnc(test_res.x, data_af, data_ea),
     'r--', label='Berechnet aus Erstaufnahmen')
    plt.title('COVID-19 - {:s} - Mittlere ITS-Liegezeit ca. {:.1f} Tage'.format(
    BL_LUT[bl], test_res.x.mean()))
    plt.xlabel('Berichtdatum')
    plt.ylabel('aktive Fälle auf ITS')
    plt.grid(True, which='major', axis='both', color=(0.85, 0.85, 0.85))
    plt.grid(True, which='minor', axis='both',color=(0.95, 0.95, 0.95))
    
    date_form = DateFormatter("%d.%m.")
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=0))
    
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.xlim([data_af.index[0], data_af.index[-1]])
    plt.legend(loc='upper left')
    plt.show()
    
    liegezeiten[bl] = pd.DataFrame(data = test_res.x, index=uwl, columns=[bl])
    
liegezeiten_df = pd.concat([l for l in liegezeiten.values()], axis=1)
liegezeiten_df.index.rename('KW',inplace=True)

plt.figure(figsize=(0.6*20, 0.6*9))
plt.title('COVID-19 - Deutschland - Schätzung der ITS-Liegezeiten nach KW')
plt.plot(liegezeiten_df.DE, 'k.', markersize=14)
plt.ylim([0, 25])
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.xlabel('Kalenderwoche')
plt.ylabel('Mittlere Liegezeit in Tagen')
plt.grid(True, which='major', axis='y', color=(0.85, 0.85, 0.85))
plt.grid(True, which='minor', axis='y',color=(0.95, 0.95, 0.95))
plt.show()

liegezeiten_df.to_csv(OUTPUT_FOLDER + OUTPUT_FILE, float_format='%.1f')