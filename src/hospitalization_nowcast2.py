# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:33:48 2021

@author: David
"""

import sys
sys.path.append('.')

# import os
# import inspect
from datetime import date
from pathlib import Path
import locale

import pandas as pd
import numpy as np
import scipy.signal as sig

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import fig_util
from IPython.display import display, Image



INPUT_PATH = r'..\data\RKI\Hospitalisierungen'
OUTPUT_PATH = r'..\output\Hospitalization_Nowcast2'
FILE_PATTERN = '{year:04d}-{month:02d}-{day:02d}_Deutschland_COVID-19-Hospitalisierungen.csv'
START_DATE = '2021-07-29' #'2021-08-01'
END_DATE = date.today().strftime('%Y-%m-%d') # '2021-11-12'

MAX_TRI_LEN = 21

ALL_DATE_RANGE = ['2020-03-03', END_DATE]
#END_DATE = '2021-11-09'
BL_FILTER = 'Thüringen'

BL_FILTER = 'Sachsen'               # 200
BL_FILTER = 'Rheinland-Pfalz'       # 80
BL_FILTER = 'Berlin'                # 160
BL_FILTER = 'Schleswig-Holstein'    # 90
BL_FILTER = 'Brandenburg'           # 160



BL_FILTER = 'Hessen'                # 140
BL_FILTER = 'Niedersachsen'         # 70
BL_FILTER = 'Hamburg'               # 120
BL_FILTER = 'Baden-Württemberg'     # 100
BL_FILTER = 'Nordrhein-Westfalen'   # 100
BL_FILTER = 'Bayern'                # 140
BL_FILTER = 'Bundesgebiet'          # 100

yscale_table = {
    '00-04': 4,
    '05-14': 2.5,
    '15-34': 7,
    '35-59': 12,
    '60-79': 25,
    '80+': 60,
    '00+': 15,
    'all': 100
    }

DO_SEPERATE_TOTAL = True

# SHOW_ONLY_THESE_AG = None
# SHOW_ONLY_THESE_AG = [
#     '35-59',
#     '60-79',
#     '80+',
#     '00+'
    # ]
SHOW_ONLY_THESE_AG = [
    '00+'
    ]

if DO_SEPERATE_TOTAL:
    AG_LIST = [
        '00-04',
        '05-14',
        '15-34',
        '35-59',
        '60-79',
        '80+'
        ]
else:
    AG_LIST = [
        '00-04',
        '05-14',
        '15-34',
        '35-59',
        '60-79',
        '80+',
        '00+'
        ]
    
SLATE = (0.15, 0.15, 0.15)


# POP_LUT = {
#     '00-04': 39.69100, 
#     '05-14': 75.08700,
#     '15-34': 189.21300,
#     '35-59': 286.66200,
#     '60-79': 181.53300,
#     '80+': 59.36400,
#     '00+': 831.55000
#     }



ytck_table = {
    '00-04': 0.1,
    '05-14': 0.05,
    '15-34': 0.2,
    '35-59': 0.25,
    '60-79': 0.5,
    '80+': 1,
    '00+': 0.25,
    'all': 2
    }

plt_col_table = {
    '00-04': (0.8, 0.0, 0.8),
    '05-14': (0, 0.5, 0.5),
    '15-34': (1, 0.7, 0),
    '35-59': (1, 0, 0),
    '60-79': (0.6, 0.6, 1),
    '80+':   (0, 0, 1),
    '00+':   (0, 0, 0)
    }
    
# %%

plt.rc('axes', axisbelow=True)
locale.setlocale(locale.LC_TIME, 'de-DE')


assert(Path(INPUT_PATH).is_dir())
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

POP_LUT = pd.read_csv(r'../data/LUT/Bundeslaender2.tsv', sep='\t', comment='#', index_col='Gebiet')


data_input_date_range = pd.date_range(START_DATE, END_DATE, freq='D')

last_year_date_range = pd.date_range(
    START_DATE.replace('2021', '2020'), 
    END_DATE.replace('2021', '2020'), freq='D')

last_year_date_range = pd.date_range(
    START_DATE.replace('2021', '2020'), 
    END_DATE.replace('2021', '2020'), freq='D')


all_date_range = pd.date_range(ALL_DATE_RANGE[0], ALL_DATE_RANGE[1], freq='D')

# pd.DataFrame(index=[], columns={
        # 'Datum': pd.Series(dtype='datetime64[ns]'),
        # '7T_Hospitalisierung_Faelle': pd.Series(dtype=np.float64),
        # 'Datenalter': pd.Series(dtype=np.int64)})

rep_tri_table = { ag: None for ag in AG_LIST }

last_working_date = ''
for dt in data_input_date_range:
    fname = INPUT_PATH + '\\' + FILE_PATTERN.format(
        year=dt.year, month=dt.month, day=dt.day)
    
    try:
        data = pd.read_csv(fname, sep=',', decimal='.', parse_dates=['Datum'])  
        last_working_date = dt.strftime('%Y-%m-%d')
    except FileNotFoundError:
        END_DATE = last_working_date
        data_input_date_range = pd.date_range(START_DATE, END_DATE, freq='D')
        break
        
    print(dt.strftime('%Y-%m-%d'))
    data = data.loc[data.Bundesland == BL_FILTER]
    
    data['Datenalter'] = data.apply(
        lambda r: (dt - r.Datum).days, 
        axis=1)
    
    last_year_data = data.loc[(data.Datum >= last_year_date_range[0]) & (data.Datum <= last_year_date_range[-1]), 
                    ['Datum', '7T_Hospitalisierung_Faelle', 'Altersgruppe', 'Datenalter']]
    
    all_data = data.loc[(data.Datum >= all_date_range[0]) & (data.Datum <= all_date_range[-1]), 
                    ['Datum', '7T_Hospitalisierung_Faelle', 'Altersgruppe', 'Datenalter']]
    
    data = data.loc[data.Datum >= data_input_date_range[0], 
                    ['Datum', '7T_Hospitalisierung_Faelle', 'Altersgruppe', 'Datenalter']]
        
    data_flt = data.loc[data.Datenalter <= MAX_TRI_LEN, :]
    
    for ag in AG_LIST:
        subdata = data_flt.loc[data_flt.Altersgruppe == ag, 
                                   ['Datum', '7T_Hospitalisierung_Faelle', 'Datenalter']].copy()
        
        if rep_tri_table[ag] is None:
            rep_tri_table[ag] = subdata
        else:    
            rep_tri_table[ag] = pd.concat([rep_tri_table[ag], subdata], ignore_index=True)

# %%


rep_tri_table2 = {}
approx_dist_table = {}
latest_curves = {}

tot_tst = [None, None, None, None]

all_plots = {}

for ag in AG_LIST:
    hdata = data.loc[data.Altersgruppe == ag, 
                                 ['Datum', '7T_Hospitalisierung_Faelle']].copy()
    hdata.set_index('Datum', inplace=True)
    hdata = pd.DataFrame(
        sig.correlate(hdata['7T_Hospitalisierung_Faelle'], 
                      1.0/7 * np.ones((7,)), mode='valid', method='direct'),
        index=hdata.index[3:-3].copy(),
        columns = ['Faelle'])
    latest_curves[ag] = hdata
    
    tdata = rep_tri_table[ag].pivot(
        index='Datum', 
        columns='Datenalter', 
        values='7T_Hospitalisierung_Faelle')    
    tdata = pd.DataFrame(
        sig.correlate(tdata, 1.0/7 * np.ones((7,1)), mode='valid', method='direct'),
        index=tdata.index[3:-3].copy(),
        columns = tdata.columns)
    rep_tri_table2[ag] = tdata
    tdata2 = tdata.div(tdata.iloc[:, -1], axis=0)
    approx_dist_table[ag] = {
        'med': tdata2.median(axis=0),
        'max': tdata2.max(axis=0),
        'min': tdata2.min(axis=0)
        }
    
    tst1 = latest_curves[ag].copy()
    tst1.iloc[:22] = latest_curves[ag].iloc[:22].div(approx_dist_table[ag]['min'].to_numpy(), axis=0)
    tst2 = latest_curves[ag].copy()
    tst2.iloc[:22] = latest_curves[ag].iloc[:22].div(approx_dist_table[ag]['med'].to_numpy(), axis=0)
    tst3 = latest_curves[ag].copy()
    tst3.iloc[:22] = latest_curves[ag].iloc[:22].div(approx_dist_table[ag]['max'].to_numpy(), axis=0)
    tst4 = latest_curves[ag].copy()
    
    if tot_tst[0] is None:
        tot_tst = [tst1.copy(), tst2.copy(), tst3.copy(), tst4.copy()]
    else:
        tot_tst[0] += tst1
        tot_tst[1] += tst2
        tot_tst[2] += tst3
        tot_tst[3] += tst4
        
    
    tst1 /= POP_LUT[ag][BL_FILTER]
    tst2 /= POP_LUT[ag][BL_FILTER]
    tst3 /= POP_LUT[ag][BL_FILTER]
    tst4 /= POP_LUT[ag][BL_FILTER]
    
    all_plots[ag] = [
        tst2.copy(),
        tst1.copy(),
        tst3.copy(),
        tst4.copy()
        ]
        
    plt.figure(figsize=(16*0.6,9*0.6))
    ax = plt.gca()
    plt.fill_between(tst1.index, tst1.Faelle.to_numpy(), tst3.Faelle.to_numpy(), facecolor=(0.0, 0.0, 0.0), alpha=0.2, label='Nowcast, Min/Max Schätzer')
    plt.plot(tst2, 'k-', linewidth=3, label='Nowcast, Median Schätzer')
    plt.plot(tst4, 'b:', linewidth=2, label='Ohne Nowcast')
    plt.title('COVID-19 - Hospitalisierungen - Altersgruppe {:s} Jahre - 7-Tage-Inzidenzen & Nowcast'.format(ag))
    plt.legend(loc='upper left')
    plt.grid()
    plt.ylim(0, yscale_table[ag])
    plt.xlim(pd.to_datetime('2021-08-01'), pd.to_datetime('2021-11-09'))
    
    date_form = DateFormatter("%d.%m.\n%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=6))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(ytck_table[ag]))
    
    ax.tick_params(which='minor', length=0, width=0)
    ax.tick_params(axis=u'both', labelsize=10, labelcolor = SLATE)
    ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
    ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))
    plt.show()
    plt.close()


# %%

if DO_SEPERATE_TOTAL:
    ag = '00+'
    
    tst1 = tot_tst[0].copy()
    tst2 = tot_tst[1].copy()
    tst3 = tot_tst[2].copy()
    tst4 = tot_tst[3].copy()
    
    tst1 /= POP_LUT[ag][BL_FILTER]
    tst2 /= POP_LUT[ag][BL_FILTER]
    tst3 /= POP_LUT[ag][BL_FILTER]
    tst4 /= POP_LUT[ag][BL_FILTER]
    
    all_plots[ag] = [
            tst2.copy(),
            tst1.copy(),
            tst3.copy(),
            tst4.copy()
            ]
    
    
    plt.figure(figsize=(16*0.6,9*0.6))
    ax = plt.gca()
    plt.fill_between(tst1.index, tst1.Faelle.to_numpy(), tst3.Faelle.to_numpy(), facecolor=(0.0, 0.0, 0.0), alpha=0.2, label='Nowcast, Min/Max Schätzer')
    plt.plot(tst2, 'k-', linewidth=3, label='Nowcast, Median Schätzer')
    plt.plot(tst4, 'b:', linewidth=2, label='Ohne Nowcast')
    plt.title('COVID-19 - Hospitalisierungen - Altersgruppe {:s} Jahre - 7-Tage-Inzidenzen & Nowcast'.format(ag))
    plt.legend(loc='upper left')
    plt.grid()
    plt.ylim(0, yscale_table[ag])
    plt.xlim(pd.to_datetime('2021-08-01'), pd.to_datetime('2021-11-09'))
    
    date_form = DateFormatter("%d.%m.\n%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=6))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(ytck_table[ag]))
    
    ax.tick_params(which='minor', length=0, width=0)
    ax.tick_params(axis=u'both', labelsize=10, labelcolor = SLATE)
    ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
    ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))
    plt.close()
# %%

ag = 'all'





fig = plt.figure(figsize=(16,9))
gs = gridspec.GridSpec(2, 1, figure=fig, 
                   height_ratios = [12, 3],
                   hspace = 0.1)
ax = fig.add_subplot(gs[0, 0])


fig.suptitle('COVID-19 - 7-Tage-Hospitalisierungsinzidenzen nach Altersgruppe mit Nowcast - {:s}\n{:s}'.format(
    data_input_date_range[-1].strftime('%d.%m.%Y'), BL_FILTER),
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=24, color=SLATE, y=0.925)


# 
# plt.plot(tst2, 'k-', linewidth=3, label='Nowcast, Median Schätzer')
# plt.plot(tst4, 'b:', linewidth=2, label='Ohne Nowcast')

if (SHOW_ONLY_THESE_AG is None) or ('80+' in SHOW_ONLY_THESE_AG):
    y1 = all_plots['80+'][1]
    y2 = all_plots['80+'][2]
    plt.fill_between(y1.index, y1.Faelle.to_numpy(), y2.Faelle.to_numpy(), 
                     facecolor = (0, 0, 1), alpha=0.2)

if (SHOW_ONLY_THESE_AG is None) or ('60-79' in SHOW_ONLY_THESE_AG):
    y1 = all_plots['60-79'][1]
    y2 = all_plots['60-79'][2]
    plt.fill_between(y1.index, y1.Faelle.to_numpy(), y2.Faelle.to_numpy(), 
                     facecolor = (0.6, 0.6, 1), alpha=0.2)

if (SHOW_ONLY_THESE_AG is None) or ('35-59' in SHOW_ONLY_THESE_AG):    
    y1 = all_plots['35-59'][1]
    y2 = all_plots['35-59'][2]
    plt.fill_between(y1.index, y1.Faelle.to_numpy(), y2.Faelle.to_numpy(), 
                     facecolor = (1, 0, 0), alpha=0.2)

if (SHOW_ONLY_THESE_AG is None) or ('15-34' in SHOW_ONLY_THESE_AG):    
    y1 = all_plots['15-34'][1]
    y2 = all_plots['15-34'][2]
    plt.fill_between(y1.index, y1.Faelle.to_numpy(), y2.Faelle.to_numpy(), 
                     facecolor = (1, 0.7, 0), alpha=0.2)

if (SHOW_ONLY_THESE_AG is None) or ('00-04' in SHOW_ONLY_THESE_AG):    
    y1 = all_plots['00-04'][1]
    y2 = all_plots['00-04'][2]
    plt.fill_between(y1.index, y1.Faelle.to_numpy(), y2.Faelle.to_numpy(), 
                     facecolor = (0.8, 0.0, 0.8), alpha=0.2)

if (SHOW_ONLY_THESE_AG is None) or ('05-14' in SHOW_ONLY_THESE_AG):    
    y1 = all_plots['05-14'][1]
    y2 = all_plots['05-14'][2]
    plt.fill_between(y1.index, y1.Faelle.to_numpy(), y2.Faelle.to_numpy(), 
                     facecolor = (0, 0.5, 0.5), alpha=0.2)
    
if (SHOW_ONLY_THESE_AG is None) or ('00+' in SHOW_ONLY_THESE_AG):    
    y1 = all_plots['00+'][1]
    y2 = all_plots['00+'][2]
    plt.fill_between(y1.index, y1.Faelle.to_numpy(), y2.Faelle.to_numpy(), 
                     facecolor = (0, 0, 0), alpha=0.2)

dalpha = 0.2
if (SHOW_ONLY_THESE_AG is None) or ('80+' in SHOW_ONLY_THESE_AG):
    plt.plot(all_plots['80+'][3], color = (0, 0, 1), linestyle=':', linewidth=2, dash_capstyle='round', alpha=dalpha)
    plt.plot(all_plots['80+'][0], color = (0, 0, 1), linestyle='-', linewidth=2, label='80+ Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('60-79' in SHOW_ONLY_THESE_AG):
    plt.plot(all_plots['60-79'][3], color = (0.6, 0.6, 1), linestyle=':', linewidth=2, dash_capstyle='round', alpha=dalpha)
    plt.plot(all_plots['60-79'][0], color = (0.6, 0.6, 1), linestyle='-', linewidth=2, label='60-79 Jahre')
    
if (SHOW_ONLY_THESE_AG is None) or ('35-59' in SHOW_ONLY_THESE_AG):   
    plt.plot(all_plots['35-59'][3], color = (1, 0, 0), linestyle=':', linewidth=2, dash_capstyle='round', alpha=dalpha)
    plt.plot(all_plots['35-59'][0], color = (1, 0, 0), linestyle='-', linewidth=2, label='35-59 Jahre')
    
if (SHOW_ONLY_THESE_AG is None) or ('15-34' in SHOW_ONLY_THESE_AG):    
    plt.plot(all_plots['15-34'][3], color = (1, 0.7, 0), linestyle=':', linewidth=2, dash_capstyle='round', alpha=dalpha)
    plt.plot(all_plots['15-34'][0], color = (1, 0.7, 0), linestyle='-', linewidth=2, label='15-34 Jahre')
    
if (SHOW_ONLY_THESE_AG is None) or ('00-04' in SHOW_ONLY_THESE_AG):
    plt.plot(all_plots['00-04'][3], color = (0.8, 0.0, 0.8), linestyle=':', linewidth=2, dash_capstyle='round', alpha=dalpha)
    plt.plot(all_plots['00-04'][0], color = (0.8, 0.0, 0.8), linestyle='-', linewidth=2, label='0-4 Jahre')
    
if (SHOW_ONLY_THESE_AG is None) or ('05-14' in SHOW_ONLY_THESE_AG):    
    plt.plot(all_plots['05-14'][3], color =  (0, 0.5, 0.5), linestyle=':', linewidth=2, dash_capstyle='round', alpha=dalpha)
    plt.plot(all_plots['05-14'][0], color = (0, 0.5, 0.5), linestyle='-', linewidth=2, label='5-14 Jahre')
    
if (SHOW_ONLY_THESE_AG is None) or ('00+' in SHOW_ONLY_THESE_AG):    
    plt.plot(all_plots['00+'][3], color =  (0, 0, 0), linestyle=':', linewidth=2, dash_capstyle='round', alpha=dalpha)
    plt.plot(all_plots['00+'][0], color = (0, 0, 0), linestyle='-', linewidth=4, label='Gesamt')

plt.legend(loc='upper left', fontsize=15.5, ncol=7)
plt.grid()
plt.ylim(0, yscale_table[ag])
plt.xlim(data_input_date_range[3], data_input_date_range[-4])

date_form = DateFormatter("%d.%m.\n%Y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=6))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.yaxis.set_minor_locator(MultipleLocator(ytck_table[ag]))

ax.tick_params(which='minor', length=0, width=0)
ax.tick_params(axis=u'both', labelsize=20, labelcolor = SLATE, pad=10)
ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))


ax.set_ylabel('7-Tage-Hospitalisierungsinzidenz', fontsize=24, color = SLATE, labelpad=14)
ax.set_xlabel('Meldedatum (Datum der pos. Meldung ans GA)', fontsize=24, color = SLATE, labelpad=14)

# xpatch = mpatches.Rectangle((0.01, 0.75), 0.42, 0.16, 
#                             fc=(1.0, 1.0, 1.0), ec=(0.95, 0.95, 0.05), 
#                             fill=True, 
#                             linestyle = '-', linewidth=1, transform=ax.transAxes)
xpatch =  mpatches.FancyBboxPatch(
    (0.013, 0.76), 0.408, 0.14, 
    boxstyle=mpatches.BoxStyle("Round", pad=0.005),
    fc=(1.0, 1.0, 1.0), ec=(0.85, 0.85, 0.85), 
    fill=True, 
    linestyle = '-', linewidth=1, transform=ax.transAxes)
    
ax.add_patch(xpatch)

plt.text(0.018, 0.88, 
            'gestrichelte Linie:\n'+
            'durchgezogene Linie:\n'+
            'schattierte Fläche:',
            horizontalalignment='left', transform=ax.transAxes,
            color = 'k', verticalalignment='top', fontsize=16.5)

plt.text(0.18, 0.88, 
            'bisheriger bekannter Datenstand\n'+
            'Nowcast, Median-Schätzer\n'+
            'Nowcast, Min-Max Schätzintervall',
            horizontalalignment='left', transform=ax.transAxes,
            color = 'k', verticalalignment='top', fontsize=16.5)



plt.text(0.5, 0.9, '2021', 
            horizontalalignment='center', transform=ax.transAxes,
            color = 'k', verticalalignment='top', fontsize=34)




Ryof = 0.01
Rynum = 0.1
Ryden = 3
#Rtext_date = dataset_date_range[-1] - pd.DateOffset(days = min_lut-2)
Rtext_date = data_input_date_range[-1] + pd.DateOffset(days = 1)
Rtext_date2 = data_input_date_range[-1] + pd.DateOffset(days = 5)
Rtext_date3 = data_input_date_range[-1] + pd.DateOffset(days = 9)

# ax.annotate('MgII', xy=(Rtext_date, all_plots['80+'][0].iloc[0]), xycoords='data', annotation_clip=False)


if (SHOW_ONLY_THESE_AG is None):
    incanno_rows = ['80+', '60-79', '35-59', '15-34', '05-14', '00-04', '00+']
else:
    incanno_rows = SHOW_ONLY_THESE_AG

incanno = pd.DataFrame(data = np.array(
    [all_plots[iag][0].iloc[0,0] for iag in incanno_rows]),
    index = incanno_rows,
    columns=['yvals'])

incanno.sort_values(by='yvals', ascending=True, inplace=True)
incanno['ypos'] = incanno['yvals'].copy()

dminoff = 1.0/30 * yscale_table[ag]



for i in range(1, incanno.index.size):
    k1 = incanno.index[i]
    k0 = incanno.index[i-1]
    ypos1 = incanno.ypos[k1]
    ypos0 = incanno.ypos[k0]
    
    if ypos1 < ypos0 + dminoff:
        incanno.loc[k1, 'ypos'] = ypos0 + dminoff
        
    
for k in incanno.index:
    yval = incanno.yvals[k]
    ypos = incanno.ypos[k]
    plt.text(Rtext_date, ypos, 
             '{:.1f}'.format(yval), horizontalalignment='right',
             color = plt_col_table[k], verticalalignment='center', fontsize=18)

ax2 = fig.add_subplot(gs[1, 0])

ax2.axis('off')


if data_input_date_range[0].year == data_input_date_range[-1].year:
    Datenstand_range_str = (
        data_input_date_range[0].strftime('%d.%m.-') + 
        data_input_date_range[-1].strftime('%d.%m.%Y') )
else:
    Datenstand_range_str = (
        data_input_date_range[0].strftime('%d.%m.%y-') + 
        data_input_date_range[-1].strftime('%d.%m.%Y') )


plt.text(0, 0.05,
    'Datenquelle:\n' + 
    'Robert Koch-Institut (2021): COVID-19-Hospitalisierungen in Deutschland, Berlin: Zenodo. DOI:10.5281/zenodo.5519056.\n'+
    'URL: https://github.com/robert-koch-institut/COVID-19-Hospitalisierungen_in_Deutschland ; ' +
    'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; eigene Berechnung/eigene Darstellung;\n' +
    'Datenlizenz CC-BY 4.0 International',
    fontsize=13)


exp_full_fname = '{:s}\\{:s}_{:s}_{:s}.png'.format(
        OUTPUT_PATH, 'HospInz_Nowcast', BL_FILTER, END_DATE)

print('Saving ' + exp_full_fname)
try:
    fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
except:
    fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)

fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
display(Image(filename=exp_full_fname))
plt.close()


# plt.figure(figsize=(16*0.7,9*0.7))
# ax = plt.gca()

# plt.plot(all_plots['00+'], color = (0, 0, 0), linestyle=':', linewidth=4, label='Gesamt', dash_capstyle='round')
# plt.plot(all_plots['80+'], color = (0, 0, 1), linestyle='-', linewidth=2, label='80+ Jahre')
# plt.plot(all_plots['60-79'], color = (0.6, 0.6, 1), linestyle='-', linewidth=2, label='60-79 Jahre')
# plt.plot(all_plots['35-59'], color = (1, 0, 0), linestyle='-', linewidth=2, label='35-59 Jahre')
# plt.plot(all_plots['15-34'], color = (1, 0.7, 0), linestyle='-', linewidth=2, label='15-34 Jahre')
# plt.plot(all_plots['00-04'], color = (0.8, 0.0, 0.8), linestyle='-', linewidth=2, label='0-4 Jahre')
# plt.plot(all_plots['05-14'], color = (0, 0.5, 0.5), linestyle='-', linewidth=2, label='5-14 Jahre')



# plt.title('COVID-19 - Hospitalisierungen - 7-Tage-Inzidenzen & Nowcast (Median Schätzer)')
# plt.legend(loc='upper left')
# plt.grid()
# plt.ylim(0, yscale_table[ag])
# plt.xlim(pd.to_datetime('2021-08-01'), pd.to_datetime('2021-11-09'))

# date_form = DateFormatter("%d.%m.\n%Y")
# ax.xaxis.set_major_formatter(date_form)
# ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=6))
# ax.xaxis.set_minor_locator(mdates.DayLocator())
# ax.yaxis.set_minor_locator(MultipleLocator(ytck_table[ag]))

# ax.tick_params(which='minor', length=0, width=0)
# ax.tick_params(axis=u'both', labelsize=10, labelcolor = SLATE)
# ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
# ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))

# %%


ag = 'all'

fig = plt.figure(figsize=(16,9))
gs = gridspec.GridSpec(2, 1, figure=fig, 
                   height_ratios = [12, 3],
                   hspace = 0.1)
ax = fig.add_subplot(gs[0, 0])


fig.suptitle('COVID-19 - 7-Tage-Hospitalisierungsinzidenzen nach Altersgruppe im Vorjahr - Stand {:s}\n{:s}'.format(
    data_input_date_range[-1].strftime('%d.%m.%Y'), BL_FILTER),
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=24, color=SLATE, y=0.925)


last_year_data_pvt = last_year_data.copy().pivot(
        index='Datum', 
        columns='Altersgruppe', 
        values='7T_Hospitalisierung_Faelle')

for k in last_year_data_pvt.columns:
    last_year_data_pvt[k] /= POP_LUT[k][BL_FILTER]

last_year_data_pvt = pd.DataFrame(
        sig.correlate(last_year_data_pvt, 
                      1.0/7 * np.ones((7,1)), mode='valid', method='direct'),
        index=last_year_data_pvt.index[3:-3].copy(),
        columns = last_year_data_pvt.columns)

if (SHOW_ONLY_THESE_AG is None) or ('80+' in SHOW_ONLY_THESE_AG):
    plt.plot(last_year_data_pvt['80+'], color = (0, 0, 1), linestyle='-', linewidth=2, label='80+ Jahre')
    
if (SHOW_ONLY_THESE_AG is None) or ('60-79' in SHOW_ONLY_THESE_AG):
    plt.plot(last_year_data_pvt['60-79'], color = (0.6, 0.6, 1), linestyle='-', linewidth=2, label='60-79 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('35-59' in SHOW_ONLY_THESE_AG):
    plt.plot(last_year_data_pvt['35-59'], color = (1, 0, 0), linestyle='-', linewidth=2, label='35-59 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('15-34' in SHOW_ONLY_THESE_AG):
    plt.plot(last_year_data_pvt['15-34'], color = (1, 0.7, 0), linestyle='-', linewidth=2, label='15-34 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('00-04' in SHOW_ONLY_THESE_AG):
    plt.plot(last_year_data_pvt['00-04'], color = (0.8, 0.0, 0.8), linestyle='-', linewidth=2, label='0-4 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('05-14' in SHOW_ONLY_THESE_AG):
    plt.plot(last_year_data_pvt['05-14'], color = (0, 0.5, 0.5), linestyle='-', linewidth=2, label='5-14 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('00+' in SHOW_ONLY_THESE_AG):
    plt.plot(last_year_data_pvt['00+'], color = (0, 0, 0), linestyle='-', linewidth=4, label='Gesamt')


plt.legend(loc='upper left', fontsize=15.5, ncol=7)
plt.grid()
plt.ylim(0, yscale_table[ag])
plt.xlim(last_year_date_range[3], last_year_date_range[-4])

date_form = DateFormatter("%d.%m.\n%Y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=5))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.yaxis.set_minor_locator(MultipleLocator(ytck_table[ag]))

ax.tick_params(which='minor', length=0, width=0)
ax.tick_params(axis=u'both', labelsize=20, labelcolor = SLATE, pad=10)
ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))


ax.set_ylabel('7-Tage-Hospitalisierungsinzidenz', fontsize=24, color = SLATE, labelpad=14)
ax.set_xlabel('Meldedatum (Datum der pos. Meldung ans GA)', fontsize=24, color = SLATE, labelpad=14)


plt.text(0.5, 0.9, '2020', 
            horizontalalignment='center', transform=ax.transAxes,
            color = 'k', verticalalignment='top', fontsize=34)



Ryof = 0.01
Rynum = 0.1
Ryden = 3
Rtext_date = last_year_date_range[-1] + pd.DateOffset(days = 1)



if (SHOW_ONLY_THESE_AG is None):
    incanno_rows = ['80+', '60-79', '35-59', '15-34', '05-14', '00-04', '00+']
else:
    incanno_rows = SHOW_ONLY_THESE_AG

incanno = pd.DataFrame(data = np.array(
    [last_year_data_pvt[iag].iloc[-1] for iag in incanno_rows]),
    index = incanno_rows,
    columns=['yvals'])



incanno.sort_values(by='yvals', ascending=True, inplace=True)
incanno['ypos'] = incanno['yvals'].copy()

dminoff = 1.0/30 * yscale_table[ag]

for i in range(1, incanno.index.size):
    k1 = incanno.index[i]
    k0 = incanno.index[i-1]
    ypos1 = incanno.ypos[k1]
    ypos0 = incanno.ypos[k0]
    
    if ypos1 < ypos0 + dminoff:
        incanno.loc[k1, 'ypos'] = ypos0 + dminoff
        
    
for k in incanno.index:
    yval = incanno.yvals[k]
    ypos = incanno.ypos[k]
    plt.text(Rtext_date, ypos, 
             '{:.1f}'.format(yval), horizontalalignment='right',
             color = plt_col_table[k], verticalalignment='center', fontsize=18)

ax2 = fig.add_subplot(gs[1, 0])

ax2.axis('off')


Datenstand_range_str = data_input_date_range[-1].strftime('%d.%m.%Y')


plt.text(0, 0.05,
    'Datenquelle:\n' + 
    'Robert Koch-Institut (2021): COVID-19-Hospitalisierungen in Deutschland, Berlin: Zenodo. DOI:10.5281/zenodo.5519056.\n'+
    'URL: https://github.com/robert-koch-institut/COVID-19-Hospitalisierungen_in_Deutschland ; ' +
    'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; eigene Berechnung/eigene Darstellung;\n' +
    'Datenlizenz CC-BY 4.0 International',
    fontsize=13)


exp_full_fname = '{:s}\\{:s}_{:s}_Vorjahr_{:s}.png'.format(
        OUTPUT_PATH, 'HospInz_Nowcast', BL_FILTER, END_DATE)

print('Saving ' + exp_full_fname)
try:
    fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
except:
    fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)

fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
display(Image(filename=exp_full_fname))
plt.close()


#%%

ag = 'all'


fig = plt.figure(figsize=(16,9))
gs = gridspec.GridSpec(2, 1, figure=fig, 
                   height_ratios = [12, 3],
                   hspace = 0.1)
ax = fig.add_subplot(gs[0, 0])


fig.suptitle('COVID-19 - 7-Tage-Hospitalisierungsinzidenzen nach Altersgruppe - Stand {:s}\n{:s}'.format(
    data_input_date_range[-1].strftime('%d.%m.%Y'), BL_FILTER),
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=24, color=SLATE, y=0.925)


all_data_pvt = all_data.copy().pivot(
        index='Datum', 
        columns='Altersgruppe', 
        values='7T_Hospitalisierung_Faelle')

for k in all_data_pvt.columns:
    all_data_pvt[k] /= POP_LUT[k][BL_FILTER]
    
all_data_pvt = pd.DataFrame(
        sig.correlate(all_data_pvt, 
                      1.0/7 * np.ones((7,1)), mode='valid', method='direct'),
        index=all_data_pvt.index[3:-3].copy(),
        columns = all_data_pvt.columns)

for k in all_data_pvt.columns:
    all_data_pvt[k].iloc[-21:] = all_plots[k][0].iloc[::-1,0].iloc[-21:]


if (SHOW_ONLY_THESE_AG is None) or ('80+' in SHOW_ONLY_THESE_AG):
    plt.plot(all_data_pvt['80+'], color = (0, 0, 1), linestyle='-', linewidth=2, label='80+ Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('60-79' in SHOW_ONLY_THESE_AG):
    plt.plot(all_data_pvt['60-79'], color = (0.6, 0.6, 1), linestyle='-', linewidth=2, label='60-79 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('35-59' in SHOW_ONLY_THESE_AG):
    plt.plot(all_data_pvt['35-59'], color = (1, 0, 0), linestyle='-', linewidth=2, label='35-59 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('15-34' in SHOW_ONLY_THESE_AG):
    plt.plot(all_data_pvt['15-34'], color = (1, 0.7, 0), linestyle='-', linewidth=2, label='15-34 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('00-04' in SHOW_ONLY_THESE_AG):
    plt.plot(all_data_pvt['00-04'], color = (0.8, 0.0, 0.8), linestyle='-', linewidth=2, label='0-4 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('05-14' in SHOW_ONLY_THESE_AG):
    plt.plot(all_data_pvt['05-14'], color = (0, 0.5, 0.5), linestyle='-', linewidth=2, label='5-14 Jahre')

if (SHOW_ONLY_THESE_AG is None) or ('00+' in SHOW_ONLY_THESE_AG):
    plt.plot(all_data_pvt['00+'], color = (0, 0, 0), linestyle='-', linewidth=4, label='Gesamt')


ax.vlines(all_date_range[-4] - pd.DateOffset(years=1),
          0, yscale_table[ag],
          colors = (0.5, 0.5, 0.5), linestyles='dashed',
          alpha = 0.5, linewidth = 5)
          

xpatch = mpatches.Rectangle((all_date_range[-4-14], 0.0), pd.DateOffset(days=14), yscale_table[ag], 
                            fc=(0.0, 0.0, 0.0, 0.15), ec=None, 
                            fill=True)

ax.add_patch(xpatch)


plt.text(1.0, 1.01, 
         'Nowcastschätzung', horizontalalignment='right', transform=ax.transAxes,
         color = (0.6, 0.6, 0.6), verticalalignment='bottom', fontsize=18)

plt.legend(loc='upper left', fontsize=15.5, ncol=7)
plt.grid()
plt.ylim(0, yscale_table[ag])
plt.xlim(all_date_range[3], all_date_range[-4])

date_form = DateFormatter("%b\n%Y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax.yaxis.set_minor_locator(MultipleLocator(ytck_table[ag]))

ax.tick_params(which='minor', length=0, width=0)
ax.tick_params(axis=u'both', labelsize=20, labelcolor = SLATE, pad=10)
ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))


ax.set_ylabel('7-Tage-Hospitalisierungsinzidenz', fontsize=24, color = SLATE, labelpad=14)
ax.set_xlabel('Meldedatum (Datum der pos. Meldung ans GA)', fontsize=24, color = SLATE, labelpad=14)

for label in ax.xaxis.get_ticklabels():
    label.set_horizontalalignment('left')


Ryof = 0.01
Rynum = 0.1
Ryden = 3
Rtext_date = all_date_range[-1] + pd.DateOffset(days = 21)



if (SHOW_ONLY_THESE_AG is None):
    incanno_rows = ['80+', '60-79', '35-59', '15-34', '05-14', '00-04', '00+']
else:
    incanno_rows = SHOW_ONLY_THESE_AG


incanno = pd.DataFrame(data = np.array(
    [[
      all_data_pvt[iag].loc[:data_input_date_range[3]].max(), 
      all_data_pvt[iag].loc[:data_input_date_range[3]].idxmax()] for iag in incanno_rows]),
    index = incanno_rows,
    columns=['yvals', 'xvals'])


incanno.sort_values(by='yvals', ascending=True, inplace=True)
incanno['ypos'] = incanno['yvals'].copy()

dminoff = 1.0/30 * yscale_table[ag]

for i in range(1, incanno.index.size):
    k1 = incanno.index[i]
    k0 = incanno.index[i-1]
    ypos1 = incanno.ypos[k1]
    ypos0 = incanno.ypos[k0]
    
    if ypos1 < ypos0 + dminoff:
        incanno.loc[k1, 'ypos'] = ypos0 + dminoff
        
    
for k in incanno.index:
    xval = incanno.xvals[k]
    yval = incanno.yvals[k]
    ypos = incanno.ypos[k]
    plt.text(Rtext_date, ypos, 
             '{:.1f}'.format(yval), horizontalalignment='right',
             color = plt_col_table[k], verticalalignment='center', fontsize=18)
    
    plt.plot([xval], [yval], marker='o', markersize=10, color=plt_col_table[k])

ax2 = fig.add_subplot(gs[1, 0])

ax2.axis('off')


Datenstand_range_str = data_input_date_range[-1].strftime('%d.%m.%Y')


plt.text(0, 0.05,
    'Datenquelle:\n' + 
    'Robert Koch-Institut (2021): COVID-19-Hospitalisierungen in Deutschland, Berlin: Zenodo. DOI:10.5281/zenodo.5519056.\n'+
    'URL: https://github.com/robert-koch-institut/COVID-19-Hospitalisierungen_in_Deutschland ; ' +
    'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; eigene Berechnung/eigene Darstellung;\n' +
    'Datenlizenz CC-BY 4.0 International',
    fontsize=13)


exp_full_fname = '{:s}\\{:s}_{:s}_Vergleich_{:s}.png'.format(
        OUTPUT_PATH, 'HospInz_Nowcast', BL_FILTER, END_DATE)

print('Saving ' + exp_full_fname)
try:
    fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
except:
    fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)

fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
display(Image(filename=exp_full_fname))
plt.close()
