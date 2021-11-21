# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 19:01:45 2021

@author: David
"""

from pathlib import Path
from datetime import datetime as dt

import zipfile
import os.path

import numpy as np

import scipy.signal as sig

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
import seaborn as sea

import fig_util
from IPython.display import display, Image

SLATE = (0.15, 0.15, 0.15)

WD_ARR = {
    1: 'Montag',
    2: 'Dienstag',
    3: 'Mittwoch',
    4: 'Donnerstag',
    5: 'Freitag',
    6: 'Samstag',
    7: 'Sonntag'
    }

OUTPUT_DIR = '..\\output\\RNowcast\\anim\\'
OUTPUT_DIR = 'D:\\COVID-19\\output\\RNowcast\\anim\\'

ARCHIVE_FPATH = '..\\data\\RKI\\Nowcasting\\Nowcast_R_{:s}.csv'
ARCHIVE_ZIP_URL = 'https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/archive/refs/heads/main.zip'
#'https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/raw/main/Archiv/Nowcast_R_{:s}.csv'

SPECIFIC_DAY = None
#SPECIFIC_DAY = '2021-09-24'
#SPECIFIC_DAY = '2021-10-08'
#SPECIFIC_DAY = '2021-11-12'


INPUT_DATA_RANGE = ['2021-03-16', dt.now().strftime('%Y-%m-%d')]
PLOT_MAX_DATE = '2021-12-31'

DO_EXTRAPOLATION = False


if not SPECIFIC_DAY is None:
    INPUT_DATA_RANGE[1] = SPECIFIC_DAY

dataset_date_range = pd.date_range(*INPUT_DATA_RANGE)

r_idx_min = dataset_date_range[0] - pd.DateOffset(days=4)

r_idx = pd.date_range(r_idx_min, dataset_date_range[-5].strftime('%Y-%m-%d'))
r_cols = pd.Int64Index(range(4, 4+7*6, 1))

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# %%

rep_tri = pd.DataFrame(
    data=np.zeros((r_idx.size, r_cols.size)),
    index=r_idx,
    columns=r_cols)

datasets = {}

for i in range(dataset_date_range.size):
    dataset_date = dataset_date_range[i]
    dataset_date_str = dataset_date.strftime('%Y-%m-%d')
    print(dataset_date_str)
    
    #if os.path.isfile(ARCHIVE_FPATH.format(dataset_date_str)):
    try:
        data = pd.read_csv(
            ARCHIVE_FPATH.format(dataset_date_str),
            index_col = 'Datum',
            parse_dates = True
            )
    except ValueError:
        # two steps:
        data = pd.read_csv(
            ARCHIVE_FPATH.format(dataset_date_str),
            parse_dates = True,
            sep=';', decimal=',',
            skip_blank_lines=False
            )
        extra_rows = data.index.size - data.index[data.Datum.isna()][0]
        data = pd.read_csv(
            ARCHIVE_FPATH.format(dataset_date_str),
            index_col = 'Datum',
            parse_dates = True,
            sep=';', decimal=',',
            date_parser=lambda x: dt.strptime(x, '%d.%m.%Y'),
            skipfooter=extra_rows, encoding='UTF-8'
            )
        data.rename(columns={'Schätzer_Neuerkrankungen': 'PS_COVID_Faelle'},
                    inplace=True)
        
    last_dataset = data.loc[:,['PS_COVID_Faelle']].copy()
    last_dataset['Iso Weekdays'] = last_dataset.index.map(lambda d: d.isoweekday())
    last_dataset['Date Offset'] = (dataset_date - last_dataset.index).days
    datasets[dataset_date_str] = last_dataset
        
    comm_rows = r_idx.intersection(data.index)
    data = data.loc[comm_rows]
    
    d_cols = (dataset_date-data.index).days
    data['Offset'] = d_cols

    comm_cols = d_cols.intersection(r_cols)
    max_offset = comm_cols.max()
    
    data = data.loc[data['Offset'] <= max_offset, ['Offset', 'PS_COVID_Faelle']]
    data = data.pivot(columns='Offset', values='PS_COVID_Faelle')
    data.fillna(0, inplace=True)
    
    rep_tri.loc[data.index, comm_cols] += data.loc[:, comm_cols]
    
    
    

(na_cols, na_rows) = np.tril_indices(rep_tri.shape[0], -1)
if any(na_cols >= r_cols.size):
    max_cols = np.nonzero(na_cols >= r_cols.size)[0][0]    
    na_cols = na_cols[:max_cols]
    na_rows = na_rows[:max_cols]

rep_tri2 = rep_tri.to_numpy().copy()
rep_tri2[r_idx.size-1-na_rows, na_cols] = np.nan

rep_tri3 = rep_tri.copy()
rep_tri3.loc[:,:] = rep_tri2


rep_tri4 = rep_tri3.iloc[:-14, :].div(rep_tri3.apply(lambda s: s[pd.Series.last_valid_index(s)], axis=1), axis=0)



# %%

q10_dist = pd.DataFrame(index=r_cols, columns=range(1,7,1))
lq_dist = pd.DataFrame(index=r_cols, columns=range(1,7,1))
med_dist = pd.DataFrame(index=r_cols, columns=range(1,7,1))
uq_dist = pd.DataFrame(index=r_cols, columns=range(1,7,1))
q90_dist = pd.DataFrame(index=r_cols, columns=range(1,7,1))

max_days_offset = r_cols.max()


for i in range(7):
    iwd = rep_tri4.index[i].isoweekday()
    rep_tri5 = rep_tri4.iloc[i::7]
    tri5_med = rep_tri5.median(axis=0)
    rep_tri5 = rep_tri5.loc[(((rep_tri5-tri5_med) > 1) | (rep_tri5-tri5_med < -1)).sum(axis=1)==0]
    rep_tri5 *= 100
    test = rep_tri5.iloc[:,0:11].melt(var_name='Datenstand in "n Tage nach Datum des Nowcasts"', value_name='Nowcast Korrekturfaktor in %')
    test = test.loc[~test['Nowcast Korrekturfaktor in %'].isna()]
    
    

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1, figure=fig, 
                           height_ratios = [7, 1],
                           hspace = 0.1)
    ax = fig.add_subplot(gs[0, 0])
    
    fig.suptitle('COVID-19 - Variation des RKI Nowcasts der Fallzahlen über Datenstand-Alter nach Wochentag: {:s}'.format(WD_ARR[iwd]),
                    horizontalalignment='center',
                    verticalalignment='center', 
                    fontsize=21, color=SLATE, y=0.91)
    
    sea.violinplot(x='Datenstand in "n Tage nach Datum des Nowcasts"', 
                   y='Nowcast Korrekturfaktor in %', 
                   data=test,
                   scale="count")
    ax.set_ylim([0, 160])
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlim([-1, 11])
    
    ax.tick_params(which='minor', length=0, width=0, pad=10)
    ax.tick_params(axis=u'both', labelsize=16, labelcolor = SLATE)
    ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
    ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))
    
    ax.set_ylabel('Nowcast Korrekturfaktor in %', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    ax.set_xlabel('Datenstand in "n Tage nach Datum des Nowcasts"', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    
    
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.axis('off')
    

    if dataset_date_range[0].year == dataset_date_range[-1].year:
        Datenstand_range_str = (
            dataset_date_range[0].strftime('%d.%m.-') + 
            dataset_date_range[-1].strftime('%d.%m.%Y') )
    else:
        Datenstand_range_str = (
            dataset_date_range[0].strftime('%d.%m.%y-') + 
            dataset_date_range[-1].strftime('%d.%m.%Y') )
        
    plt.text(0, 0.05,
        'Datenquelle:\n' + 
        'Robert Koch-Institut (RKI), an der Heiden, Matthias (2021): SARS-CoV-2-Nowcasting und -R-Schaetzung, Berlin: Zenodo. DOI:10.5281/zenodo.4680400\n'+
        'URL: https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung ; ' +
        'Abfragedatum/Datenstand: ' + Datenstand_range_str + ';\n' +
        'Datenlizenz CC-BY 4.0 International; eigene Berechnung/eigene Darstellung',
        fontsize=11.5)
    
    
    if True:
        exp_full_fname = '{:s}{:s}_{:d}_{:s}.png'.format(
            OUTPUT_DIR + '..\\', 'Nowcast_Var', iwd, WD_ARR[iwd])
        
        print('Saving ' + exp_full_fname)
        try:
            fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
        except:
            fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)
        
        fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
        display(Image(filename=exp_full_fname))
        plt.close()
    else:
        plt.show()
    
    
    q10_dist.loc[:, iwd] = 0.01 * rep_tri5.quantile(0.1, axis=0)
    lq_dist.loc[:, iwd] = 0.01 * rep_tri5.quantile(0.25, axis=0)
    med_dist.loc[:, iwd] = 0.01 * rep_tri5.median(axis=0)
    uq_dist.loc[:, iwd] = 0.01 * rep_tri5.quantile(0.75, axis=0)
    q90_dist.loc[:, iwd] = 0.01 * rep_tri5.quantile(0.9, axis=0)
#input_matrix[np.tril_indices(input_matrix.shape[0], -1)] = np.nan 

# %%

for i in range(dataset_date_range.size):
    dataset_date = dataset_date_range[i]
    dataset_date_str = dataset_date.strftime('%Y-%m-%d')
    print(dataset_date_str)
    
    
    last_dataset = datasets[dataset_date_str]

    last_dataset['Med NowNowcast'] = last_dataset.apply(lambda r: r['PS_COVID_Faelle'] if r['Date Offset'] > max_days_offset else r['PS_COVID_Faelle'] / med_dist[r['Iso Weekdays']][r['Date Offset']], axis=1)        
    #last_dataset['Q1 NowNowcast'] = last_dataset.apply(lambda r: r['PS_COVID_Faelle'] if r['Date Offset'] > max_days_offset else r['PS_COVID_Faelle'] / lq_dist[r['Iso Weekdays']][r['Date Offset']], axis=1)        
    #last_dataset['Q3 NowNowcast'] = last_dataset.apply(lambda r: r['PS_COVID_Faelle'] if r['Date Offset'] > max_days_offset else r['PS_COVID_Faelle'] / uq_dist[r['Iso Weekdays']][r['Date Offset']], axis=1)        

    
    
    last_dataset['Med NowNowcast 7d MA'] = np.hstack((
        np.full((3), np.nan),
        sig.correlate(last_dataset['Med NowNowcast'], np.full((7), 1.0/7), method='direct', mode='valid'),
        np.full((3), np.nan)))
    
    # last_dataset['Q1 NowNowcast 7d MA'] = np.hstack((
    #     np.full((3), np.nan),
    #     sig.correlate(last_dataset['Q1 NowNowcast'], np.full((7), 1.0/7), method='direct', mode='valid'),
    #     np.full((3), np.nan)))
    
    # last_dataset['Q3 NowNowcast 7d MA'] = np.hstack((
    #     np.full((3), np.nan),
    #     sig.correlate(last_dataset['Q3 NowNowcast'], np.full((7), 1.0/7), method='direct', mode='valid'),
    #     np.full((3), np.nan)))
    
    last_dataset['Nowcast 7d MA'] = np.hstack((
        np.full((3), np.nan),
        sig.correlate(last_dataset['PS_COVID_Faelle'], np.full((7), 1.0/7), method='direct', mode='valid'),
        np.full((3), np.nan)))
    
    
    v = last_dataset['Med NowNowcast 7d MA'].to_numpy()
    v = v[4:] / v[:-4]
    v = np.hstack((
        np.full((6), np.nan),
        v[:-2]))
    last_dataset['R (Med NowNowcast 7d MA)'] = v
    v = 2.0**(sig.correlate(np.log2(v), np.full((7), 1.0/7), method='direct', mode='valid'))
    v = np.hstack((
        np.full((3), np.nan),
        v,
        np.full((3), np.nan)))    
    last_dataset['Rgeom (Med NowNowcast 7d MA)'] = v
    
    # v1 = last_dataset['Q1 NowNowcast 7d MA'].to_numpy()
    # v3 = last_dataset['Q3 NowNowcast 7d MA'].to_numpy()
    # vmin = np.vstack((v1, v3)).max(axis=0)
    # vmax = np.vstack((v1, v3)).max(axis=0)    
    # vlo = vmin[4:] / vmax[:-4]
    # vhi = vmax[4:] / vmin[:-4]
    # vlo = np.hstack((
    #     np.full((3), np.nan),
    #     vlo,
    #     np.full((1), np.nan)))
    # vhi = np.hstack((
    #     np.full((3), np.nan),
    #     vhi,
    #     np.full((1), np.nan)))
    # last_dataset['R (Q3 NowNowcast 7d MA)'] = vhi
    # last_dataset['R (Q1 NowNowcast 7d MA)'] = vlo
    # vlo = 2.0**(sig.correlate(np.log2(vlo), np.full((7), 1.0/7), method='direct', mode='valid'))
    # vhi = 2.0**(sig.correlate(np.log2(vhi), np.full((7), 1.0/7), method='direct', mode='valid'))
    # vlo = np.hstack((
    #     np.full((3), np.nan),
    #     vlo,
    #     np.full((3), np.nan)))    
    # vhi = np.hstack((
    #     np.full((3), np.nan),
    #     vhi,
    #     np.full((3), np.nan)))    
    # last_dataset['Rgeom (Q3 NowNowcast 7d MA)'] = vhi
    # last_dataset['Rgeom (Q1 NowNowcast 7d MA)'] = vlo
    
    v = last_dataset['Nowcast 7d MA'].to_numpy()
    v = v[4:] / v[:-4]
    v = np.hstack((
        np.full((6), np.nan),
        v[:-2]))
    last_dataset['R (Nowcast 7d MA)'] = v
    v = 2.0**(sig.correlate(np.log2(v), np.full((7), 1.0/7), method='direct', mode='valid'))
    v = np.hstack((
        np.full((3), np.nan),
        v,
        np.full((3), np.nan)))
    last_dataset['Rgeom (Nowcast 7d MA)'] = v
    
    datasets[dataset_date_str] = last_dataset
    
# %%

fidz = datasets[INPUT_DATA_RANGE[0]]['Rgeom (Med NowNowcast 7d MA)'].first_valid_index()
total_idz = datasets[INPUT_DATA_RANGE[1]]['Rgeom (Med NowNowcast 7d MA)'].index[12:-4].copy()

test = pd.DataFrame(index=total_idz, columns=dataset_date_range.copy())

for dataset_date in dataset_date_range:
    dataset_date_str = dataset_date.strftime('%Y-%m-%d')
    
    cur_dataset = datasets[dataset_date_str]
    
    comm_idz = total_idz.intersection(cur_dataset.index)
    
    test.loc[comm_idz, dataset_date] = cur_dataset.loc[comm_idz, 'Rgeom (Med NowNowcast 7d MA)']

test_s = test.subtract(test.iloc[:, -1], axis=0)
# if np.isnan(test_s.iloc[0,0]):
#     first_nnz_idx = np.nonzero(~test_s.iloc[:,0].isna().to_numpy())[0][0]
#     test_s = test_s.iloc[first_nnz_idx:,:]
    
first_nz_idx = np.nonzero(test_s.iloc[:,0].isna().to_numpy())[0][0]-1
test_s = test_s.iloc[first_nz_idx:,:-1]
    
test_s['Datum'] = test_s.index.copy()
test_s = test_s.melt(value_name='Error', var_name='Report Date', id_vars='Datum').dropna()
test_s['Offset'] = (test_s['Report Date'] - test_s['Datum']).dt.days
test_s.drop(columns=['Report Date'], inplace=True)
test_s.loc[:, 'Error'] = pd.to_numeric(test_s.Error)
test_s = -test_s.pivot(index='Datum', columns='Offset', values='Error')

max_err = test_s.apply(lambda c: c.dropna().max(), axis=0)
min_err = test_s.apply(lambda c: c.dropna().min(), axis=0)
med_err = test_s.apply(lambda c: c.dropna().median(), axis=0)
q25_err = test_s.apply(lambda c: c.dropna().quantile(0.25), axis=0)
q75_err = test_s.apply(lambda c: c.dropna().quantile(0.75), axis=0)
q025_err = test_s.apply(lambda c: c.dropna().quantile(0.025), axis=0)
q975_err = test_s.apply(lambda c: c.dropna().quantile(0.975), axis=0)
iq50_err = (q75_err - q25_err)
iq95_err = (q975_err - q025_err)

#test2 = test.div(test.iloc[:,-1], axis=0)
#first_nz_idx = np.nonzero((test_s.iloc[:,0]!=1).to_numpy())[0][0]


# test2 = test2.iloc[first_nz_idx:,:]
# test2a = test2.iloc[:-(31+12), :]

# test3 = pd.DataFrame(index = test2a.index, columns = range(12, 100))

# for d in test2a.index:
#     v = pd.DataFrame(data = test2a.loc[d, :].to_numpy().copy(),
#                      index = (test2a.columns - d).days,
#                      columns = ['data'])
    
    
#     com_cols = test3.columns.intersection(v.index)
#     test3.loc[d, com_cols] = v.loc[com_cols, 'data']-1

# error_band_md = test3.apply(lambda c: c.dropna().quantile(0.5) , axis=0)
# error_band_q1 = test3.apply(lambda c: c.dropna().quantile(0.25) , axis=0)
# error_band_q3 = test3.apply(lambda c: c.dropna().quantile(0.75) , axis=0)

# error_band_max = test3.apply(lambda c: c.dropna().max(), axis=0)
# error_band_min = test3.apply(lambda c: c.dropna().min(), axis=0)

# error_band_lo = error_band_md - 1.5 * (error_band_q3 - error_band_q1)
# error_band_hi = error_band_md + 1.5 * (error_band_q3 - error_band_q1)

# %%

band_data_med = pd.DataFrame(index = r_idx, columns=dataset_date_range)


band_data_min = pd.DataFrame(index = r_idx, columns=dataset_date_range)
band_data_max = pd.DataFrame(index = r_idx, columns=dataset_date_range)

band_data_iq95_lo = pd.DataFrame(index = r_idx, columns=dataset_date_range)
band_data_iq95_hi = pd.DataFrame(index = r_idx, columns=dataset_date_range)

band_data_iq50_lo = pd.DataFrame(index = r_idx, columns=dataset_date_range)
band_data_iq50_hi = pd.DataFrame(index = r_idx, columns=dataset_date_range)

max_num_entries = (dataset_date_range[-1]-dataset_date_range[0]).days

max_lut = max_err.index.max()
min_lut = max_err.index.min()

# max_err = test_s.apply(lambda c: c.dropna().max(), axis=0)
# min_err = test_s.apply(lambda c: c.dropna().min(), axis=0)
# med_err = test_s.apply(lambda c: c.dropna().median(), axis=0)
# q25_err = test_s.apply(lambda c: c.dropna().quantile(0.25), axis=0)
# q75_err = test_s.apply(lambda c: c.dropna().quantile(0.75), axis=0)
# iq_err = 1.5 * (q75_err - q25_err)

for i in range(dataset_date_range.size):
    dataset_date = dataset_date_range[i]
    dataset_date_str = dataset_date.strftime('%Y-%m-%d')
    
    v = pd.DataFrame(datasets[dataset_date_str]['Rgeom (Med NowNowcast 7d MA)'].iloc[-max_num_entries:].dropna())
    v.rename(columns={'Rgeom (Med NowNowcast 7d MA)': 'Data'}, inplace=True)
    cur_idx = v.index
    
    com_idx = r_idx.intersection(cur_idx)
    if com_idx.size == 0:
        continue
    
    v = v.loc[com_idx]
    cur_idx = v.index
    v['Offset'] = (dataset_date - cur_idx).days
    cur_idx = v.index
    com_idx = r_idx.intersection(cur_idx)
    
    # vmed = v['Data']  
    vmed = v['Data'] + v.apply(lambda r: 0.0 if r['Offset'] > max_lut else med_err[r['Offset']], axis=1)
    vmax = v['Data'] + v.apply(lambda r: 0.0 if r['Offset'] > max_lut else max_err[r['Offset']], axis=1)
    vmin = v['Data'] + v.apply(lambda r: 0.0 if r['Offset'] > max_lut else min_err[r['Offset']], axis=1)
    vq25 = v['Data'] + v.apply(lambda r: 0.0 if r['Offset'] > max_lut else q25_err[r['Offset']], axis=1)
    vq75 = v['Data'] + v.apply(lambda r: 0.0 if r['Offset'] > max_lut else q75_err[r['Offset']], axis=1)
    vq025 = v['Data'] + v.apply(lambda r: 0.0 if r['Offset'] > max_lut else q025_err[r['Offset']], axis=1)
    vq975 = v['Data'] + v.apply(lambda r: 0.0 if r['Offset'] > max_lut else q975_err[r['Offset']], axis=1)
    
    band_data_med.loc[com_idx, dataset_date] = vmed.loc[com_idx]
    
    band_data_min.loc[com_idx, dataset_date] = vmin.loc[com_idx]
    band_data_max.loc[com_idx, dataset_date] = vmax.loc[com_idx]
    
    band_data_iq50_lo.loc[com_idx, dataset_date] = vq25.loc[com_idx]
    band_data_iq50_hi.loc[com_idx, dataset_date] = vq75.loc[com_idx]

    band_data_iq95_lo.loc[com_idx, dataset_date] = vq025.loc[com_idx]
    band_data_iq95_hi.loc[com_idx, dataset_date] = vq975.loc[com_idx]

# %%


plt.rc('axes', axisbelow=True)


if False:
    
    # testX = test.subtract(test.iloc[:,-1], axis=0)
    # band_max = testX.apply(lambda r: r.dropna().max(), axis=1).max()
    # band_min = testX.apply(lambda r: r.dropna().min(), axis=1).min()
    # band_q75 = testX.apply(lambda r: r.dropna().quantile(0.75), axis=1)
    # band_q25 = testX.apply(lambda r: r.dropna().quantile(0.25), axis=1)
    # band_iq = 1.5 * (band_q75 - band_q25).max()
    # band_pm = np.max([-band_min, band_max])
    
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(2, 1, figure=fig, 
                       height_ratios = [14, 3],
                       hspace = 0.1)
    ax = fig.add_subplot(gs[0, 0])
    
    if dataset_date_range[0].year == dataset_date_range[-1].year:
        Datenstand_range_str = (
            dataset_date_range[0].strftime('%d.%m.-') + 
            dataset_date_range[-1].strftime('%d.%m.%Y') )
    else:
        Datenstand_range_str = (
            dataset_date_range[0].strftime('%d.%m.%y-') + 
            dataset_date_range[-1].strftime('%d.%m.%Y') )
    
    
    fig.suptitle('COVID-19 - Original Punktschätzer des RKI 7-Tage-R nach Erkrankungsdatum - {:s}'.format(
        Datenstand_range_str),
                    horizontalalignment='center',
                    verticalalignment='center', 
                    fontsize=21, color=SLATE, y=0.91)
    
    for i in range(dataset_date_range.size):
        dataset_date = dataset_date_range[i]
        dataset_date_str = dataset_date.strftime('%Y-%m-%d')
        xidz = datasets[dataset_date_str].index[-max_num_entries:]
        v = datasets[dataset_date_str]['R (Nowcast 7d MA)'].iloc[-max_num_entries:]
        
        #y1 = v * (1 - IQmax)
        #y2 = v * (1 + IQmax)
        #y1 = datasets[dataset_date_str]['R (Q1 NowNowcast 7d MA)'].iloc[-56:]
        #y2 = datasets[dataset_date_str]['R (Q3 NowNowcast 7d MA)'].iloc[-56:]
        #plt.fill_between(xidz, y1, y2, facecolor=(0.3, 0.3, 0.3), alpha=0.5)
        plt.plot(v)
        
    
    ax.set_ylim([0.6,1.5])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    
    #ax.set_xlim([r_idx[0], r_idx[-1]])
    ax.set_xlim([
        pd.to_datetime(r_idx[0]),
        pd.to_datetime(PLOT_MAX_DATE)
        ])
    
    date_form = DateFormatter("%d.%m.\n%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2, byweekday=0))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    
    ax.tick_params(which='minor', length=0, width=0)
    ax.tick_params(axis=u'both', labelsize=16, labelcolor = SLATE)
    ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
    ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))
    
    ax.set_ylabel('7-Tage Reproduktionszahl R', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    ax.set_xlabel('Geschätztes Erkrankungsdatum', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    
    
    
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.axis('off')
    
    
    plt.text(0, 0.05,
        'Datenquelle:\n' + 
        'Robert Koch-Institut (RKI), an der Heiden, Matthias (2021): SARS-CoV-2-Nowcasting und -R-Schaetzung, Berlin: Zenodo. DOI:10.5281/zenodo.4680400\n'+
        'URL: https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung ; ' +
        'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; eigene Berechnung/eigene Darstellung; \n' +
        'Datenlizenz CC-BY 4.0 International',
        fontsize=11.5)
    
    exp_full_fname = '{:s}{:s}.png'.format(
        OUTPUT_DIR + '..\\', 'Nowcasts_RKI_orig')
    
    print('Saving ' + exp_full_fname)
    fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
    
    fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
    display(Image(filename=exp_full_fname))
    plt.close()
    
    
    
    
    
    
    
    # testX = test.subtract(test.iloc[:,-1], axis=0)
    # band_max = testX.apply(lambda r: r.dropna().max(), axis=1).max()
    # band_min = testX.apply(lambda r: r.dropna().min(), axis=1).min()
    # band_q75 = testX.apply(lambda r: r.dropna().quantile(0.75), axis=1)
    # band_q25 = testX.apply(lambda r: r.dropna().quantile(0.25), axis=1)
    # band_iq = 1.5 * (band_q75 - band_q25).max()
    # band_pm = np.max([-band_min, band_max])
    
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(2, 1, figure=fig, 
                       height_ratios = [14, 3],
                       hspace = 0.1)
    ax = fig.add_subplot(gs[0, 0])
    
    if dataset_date_range[0].year == dataset_date_range[-1].year:
        Datenstand_range_str = (
            dataset_date_range[0].strftime('%d.%m.-') + 
            dataset_date_range[-1].strftime('%d.%m.%Y') )
    else:
        Datenstand_range_str = (
            dataset_date_range[0].strftime('%d.%m.%y-') + 
            dataset_date_range[-1].strftime('%d.%m.%Y') )
    
    
    fig.suptitle('COVID-19 - Punktschätzer$^{{*)}}$ des RKI 7-Tage-R nach Erkrankungsdatum - {:s}'.format(
        Datenstand_range_str),
                    horizontalalignment='center',
                    verticalalignment='center', 
                    fontsize=21, color=SLATE, y=0.91)
    
    for i in range(dataset_date_range.size):
        dataset_date = dataset_date_range[i]
        dataset_date_str = dataset_date.strftime('%Y-%m-%d')
        xidz = datasets[dataset_date_str].index[-max_num_entries:]
        v = datasets[dataset_date_str]['Rgeom (Nowcast 7d MA)'].iloc[-max_num_entries:]
        
        #y1 = v * (1 - IQmax)
        #y2 = v * (1 + IQmax)
        #y1 = datasets[dataset_date_str]['R (Q1 NowNowcast 7d MA)'].iloc[-56:]
        #y2 = datasets[dataset_date_str]['R (Q3 NowNowcast 7d MA)'].iloc[-56:]
        #plt.fill_between(xidz, y1, y2, facecolor=(0.3, 0.3, 0.3), alpha=0.5)
        plt.plot(v)
        
    
    ax.set_ylim([0.6,1.5])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    
    #ax.set_xlim([r_idx[0], r_idx[-1]])
    ax.set_xlim([
        pd.to_datetime(r_idx[0]),
        pd.to_datetime(PLOT_MAX_DATE)
        ])
    
    date_form = DateFormatter("%d.%m.\n%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2, byweekday=0))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    
    ax.tick_params(which='minor', length=0, width=0)
    ax.tick_params(axis=u'both', labelsize=16, labelcolor = SLATE)
    ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
    ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))
    
    ax.set_ylabel('7-Tage Reproduktionszahl R', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    ax.set_xlabel('Geschätztes Erkrankungsdatum', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    
    
    
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.axis('off')
    
    
    plt.text(0, 0.05,
        'Datenquelle:\n' + 
        'Robert Koch-Institut (RKI), an der Heiden, Matthias (2021): SARS-CoV-2-Nowcasting und -R-Schaetzung, Berlin: Zenodo. DOI:10.5281/zenodo.4680400\n'+
        'URL: https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung ; ' +
        'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; eigene Berechnung/eigene Darstellung; \n' +
        'Datenlizenz CC-BY 4.0 International               '+
        '$^{*)}$ gleitender geometrischer Mittelwert (Wurzel der Produkte)',
        fontsize=11.5)
    
    exp_full_fname = '{:s}{:s}.png'.format(
        OUTPUT_DIR + '..\\', 'Nowcasts_RKI_geom')
    
    print('Saving ' + exp_full_fname)
    fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
    
    fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
    display(Image(filename=exp_full_fname))
    plt.close()
    
    
    
    
    
        
    
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(2, 1, figure=fig, 
                       height_ratios = [14, 3],
                       hspace = 0.1)
    ax = fig.add_subplot(gs[0, 0])
    
    fig.suptitle('COVID-19 - Wochentagkorrigierter Punktschätzer$^{{*)}}$ des RKI 7-Tage-R nach Erkrankungsdatum - {:s}'.format(
        Datenstand_range_str),
                    horizontalalignment='center',
                    verticalalignment='center', 
                    fontsize=21, color=SLATE, y=0.91)
    
    # y1 = band_data_min.apply(lambda r: r.dropna().min(), axis=1).dropna()
    # y2 = band_data_max.apply(lambda r: r.dropna().max(), axis=1).dropna()
    # x = y1.index
    # plt.fill_between(x, y1, y2, facecolor=(0.0, 0.0, 0.0), alpha=0.2)
    
    
    # y1 = band_data_iq_min.apply(lambda r: r.dropna().min(), axis=1).dropna()
    # y2 = band_data_iq_max.apply(lambda r: r.dropna().max(), axis=1).dropna()
    # x = y1.index    
    # plt.fill_between(x, y1, y2, facecolor=(1.0, 0.0, 0.0), alpha=0.8)
    
    # y1 = band_data_q25.apply(lambda r: r.dropna().min(), axis=1).dropna()
    # y2 = band_data_q75.apply(lambda r: r.dropna().max(), axis=1).dropna()
    # x = y1.index    
    # plt.fill_between(x, y1, y2, facecolor=(0.4, 0.4, 1.0), alpha=0.8)
    
    for i in range(dataset_date_range.size):
        dataset_date = dataset_date_range[i]
        dataset_date_str = dataset_date.strftime('%Y-%m-%d')
        xidz = datasets[dataset_date_str].index[-max_num_entries:]
        v = datasets[dataset_date_str]['Rgeom (Med NowNowcast 7d MA)'].iloc[-max_num_entries:]
        
        plt.plot(v) #, 'k-') #, linewidth=0.5)
            
    
    ax.set_ylim([0.6,1.5])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    
    #ax.set_xlim([r_idx[0], r_idx[-1]])
    ax.set_xlim([
        pd.to_datetime(r_idx[0]),
        pd.to_datetime(PLOT_MAX_DATE)
        ])
    
    date_form = DateFormatter("%d.%m.\n%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2, byweekday=0))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    
    ax.tick_params(which='minor', length=0, width=0)
    ax.tick_params(axis=u'both', labelsize=16, labelcolor = SLATE)
    ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
    ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))
    
    ax.set_ylabel('7-Tage Reproduktionszahl R', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    ax.set_xlabel('Geschätztes Erkrankungsdatum', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    
    
    
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.axis('off')
    
    
    plt.text(0, 0.05,
        'Datenquelle:\n' + 
        'Robert Koch-Institut (RKI), an der Heiden, Matthias (2021): SARS-CoV-2-Nowcasting und -R-Schaetzung, Berlin: Zenodo. DOI:10.5281/zenodo.4680400\n'+
        'URL: https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung ; ' +
        'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; eigene Berechnung/eigene Darstellung;\n' +
        'Datenlizenz CC-BY 4.0 International               '+
        '$^{*)}$ gleitender geometrischer Mittelwert (Wurzel der Produkte)',
        fontsize=11.5)
    
    exp_full_fname = '{:s}{:s}.png'.format(
        OUTPUT_DIR + '..\\', 'Nowcasts_RKI_korr')
    
    print('Saving ' + exp_full_fname)
    fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
    
    fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
    display(Image(filename=exp_full_fname))
    plt.close()



 # %%

ext_r_idx = pd.date_range(r_idx[0], r_idx[-1]-pd.DateOffset(days=1))

old_boundaries = pd.DataFrame(index = ext_r_idx, columns=[
    'min', 'max', 'iq50_lo', 'iq50_hi', 'iq95_lo', 'iq95_hi', 'med'])

for i in range(max_num_entries+1):
    # i = max_num_entries
    
    dataset_date = dataset_date_range[i]
    dataset_date_str = dataset_date.strftime('%Y-%m-%d')
        
    if pd.to_numeric(band_data_min.iloc[:,i].dropna()).size == 0:
        continue    
       
    
    y1 = pd.to_numeric(band_data_min.iloc[:,i].dropna())
    y2 = pd.to_numeric(band_data_max.iloc[:,i].dropna())
    old_boundaries.loc[y1.index[-1], 'min'] = y1.iloc[-1]
    old_boundaries.loc[y2.index[-1], 'max'] = y2.iloc[-1]
    
    
    y1 = pd.to_numeric(band_data_iq95_lo.iloc[:,i].dropna())
    y2 = pd.to_numeric(band_data_iq95_hi.iloc[:,i].dropna())
    old_boundaries.loc[y1.index[-1], 'iq95_lo'] = y1.iloc[-1]
    old_boundaries.loc[y2.index[-1], 'iq95_hi'] = y2.iloc[-1]
    
    
    y1 = pd.to_numeric(band_data_iq50_lo.iloc[:,i].dropna())
    y2 = pd.to_numeric(band_data_iq50_hi.iloc[:,i].dropna())
    old_boundaries.loc[y1.index[-1], 'iq50_lo'] = y1.iloc[-1]
    old_boundaries.loc[y2.index[-1], 'iq50_hi'] = y2.iloc[-1]
    
    y = pd.to_numeric(band_data_med.iloc[:,i].dropna())
    old_boundaries.loc[y.index[-1], 'med'] = y.iloc[-1]

# extrapolation of last 3 days

# for i in range(max_num_entries-2, max_num_entries+1):

# for j in range(-10,-4.shape[1]):
#     p = np.polyfit([*range(-9,-3)], pd.to_numeric(old_boundaries.iloc[-9:-3, j]).to_numpy(),2)
#     old_boundaries.iloc[-3:, j] = np.polyval(p, [-3,-2,-1])

# %%

if DO_EXTRAPOLATION:
    p = np.polyfit([*range(-10,-4)], pd.to_numeric(band_data_min.iloc[-10:-4, -1]).to_numpy(), 2)
    band_data_min.iloc[-4:-1, -1] = np.polyval(p, [-4,-3,-2])
    
    p = np.polyfit([*range(-10,-4)], pd.to_numeric(band_data_max.iloc[-10:-4, -1]).to_numpy(), 2)
    band_data_max.iloc[-4:-1, -1] = np.polyval(p, [-4,-3,-2])
    
    p = np.polyfit([*range(-10,-4)], pd.to_numeric(band_data_iq95_lo.iloc[-10:-4, -1]).to_numpy(), 2)
    band_data_iq95_lo.iloc[-4:-1, -1] = np.polyval(p, [-4,-3,-2])
    
    p = np.polyfit([*range(-10,-4)], pd.to_numeric(band_data_iq95_hi.iloc[-10:-4, -1]).to_numpy(), 2)
    band_data_iq95_hi.iloc[-4:-1, -1] = np.polyval(p, [-4,-3,-2])
    
    p = np.polyfit([*range(-10,-4)], pd.to_numeric(band_data_iq50_lo.iloc[-10:-4, -1]).to_numpy(), 2)
    band_data_iq50_lo.iloc[-4:-1, -1] = np.polyval(p, [-4,-3,-2])
    
    p = np.polyfit([*range(-10,-4)], pd.to_numeric(band_data_iq50_hi.iloc[-10:-4, -1]).to_numpy(), 2)
    band_data_iq50_hi.iloc[-4:-1, -1] = np.polyval(p, [-4,-3,-2])

    p = np.polyfit([*range(-10,-4)], pd.to_numeric(band_data_med.iloc[-10:-4, -1]).to_numpy(), 2)
    band_data_med.iloc[-4:-1, -1] = np.polyval(p, [-4,-3,-2])


# for i in range(max_num_entries+1+3):
i = max_num_entries

# dataset_date = dataset_date_range[i]
# dataset_date_str = dataset_date.strftime('%Y-%m-%d')
    
# if pd.to_numeric(band_data_min.iloc[:,i].dropna()).size == 0:
#     continue    

fig = plt.figure(figsize=(16,9))
gs = gridspec.GridSpec(2, 1, figure=fig, 
                   height_ratios = [14, 3],
                   hspace = 0.1)
ax = fig.add_subplot(gs[0, 0])

fig.suptitle('COVID-19 - Wochentagkorrigierter Punktschätzer$^{{*)}}$ des RKI 7-Tage-R nach Erkrankungsdatum - {:s}'.format(dataset_date_str),
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=21, color=SLATE, y=0.91)


y1 = pd.to_numeric(band_data_min.iloc[:,i].dropna())
y2 = pd.to_numeric(band_data_max.iloc[:,i].dropna())
x = y1.index
Rmin = y1.iloc[-1]
Rmax = y2.iloc[-1]

if DO_EXTRAPOLATION:
    plt.fill_between(x[:-3], y1[:-3], y2[:-3], facecolor=(0.0, 0.0, 0.0), alpha=0.2, label='Min/Max Schätz-Intervall')
    plt.fill_between(x[-4:], y1[-4:], y2[-4:], facecolor=(0.0, 0.0, 0.0), alpha=0.05, label='Min/Max S.-I. extrapoliert')
else:
    plt.fill_between(x, y1, y2, facecolor=(0.0, 0.0, 0.0), alpha=0.2, label='Min/Max Schätz-Intervall')
    


y1 = pd.to_numeric(band_data_iq95_lo.iloc[:,i].dropna())
y2 = pd.to_numeric(band_data_iq95_hi.iloc[:,i].dropna())
x = y1.index    
R95lo = y1.iloc[-1]
R95hi = y2.iloc[-1]

if DO_EXTRAPOLATION:
    plt.fill_between(x[:-3], y1[:-3], y2[:-3], facecolor=(1.0, 0.0, 0.0), alpha=0.8, label='95% Schätz-Intervall')
    plt.fill_between(x[-4:], y1[-4:], y2[-4:], facecolor=(1.0, 0.0, 0.0), alpha=0.2, label='95% S.-I. extrapoliert')
else:
    plt.fill_between(x, y1, y2, facecolor=(1.0, 0.0, 0.0), alpha=0.8, label='95% Schätz-Intervall')

y1 = pd.to_numeric(band_data_iq50_lo.iloc[:,i].dropna())
y2 = pd.to_numeric(band_data_iq50_hi.iloc[:,i].dropna())
x = y1.index    
R50lo = y1.iloc[-1]
R50hi = y2.iloc[-1]

if DO_EXTRAPOLATION:
    plt.fill_between(x[:-3], y1[:-3], y2[:-3], facecolor=(0.0, 0.0, 1.0), alpha=0.8, label='50% (IQR) Schätz-Intervall')
    plt.fill_between(x[-4:], y1[-4:], y2[-4:], facecolor=(0.0, 0.0, 1.0), alpha=0.2, label='50% (IQR) S.-I. extrapoliert')
else:
    plt.fill_between(x, y1, y2, facecolor=(0.0, 0.0, 1.0), alpha=0.8, label='50% (IQR) Schätz-Intervall')

y = pd.to_numeric(band_data_med.iloc[:,i].dropna())
x = y.index
Rmed = y.iloc[-1]

plt.plot(old_boundaries['min'].dropna(), color='gray', linestyle=':', label='Historie Min/Max Schätz-Intervall')
plt.plot(old_boundaries['max'].dropna(), color='gray', linestyle=':')
plt.plot(old_boundaries['iq95_lo'].dropna(), color='r', linestyle=':', label='Historie 95% Schätz-Intervall')
plt.plot(old_boundaries['iq95_hi'].dropna(), color='r', linestyle=':')
plt.plot(old_boundaries['iq50_lo'].dropna(), color='b', linestyle=':', label='Historie 50% (IQR) Schätz-Intervall')
plt.plot(old_boundaries['iq50_hi'].dropna(), color='b', linestyle=':')

#for i in range(dataset_date_range.size):
xidz = datasets[dataset_date_str].index[-max_num_entries:]
v = datasets[dataset_date_str]['Rgeom (Med NowNowcast 7d MA)'].iloc[-max_num_entries:]

if DO_EXTRAPOLATION:
    plt.plot(x[:-3], y[:-3], 'k-', label= 'Median')
    plt.plot(x[-4:], y[-4:], 'k-', alpha=0.25, label= 'Median extrapoliert')
else:
    plt.plot(x, y, 'k-', label= 'Median')

plt.plot(old_boundaries['med'].dropna(), 'k--', label= 'Historie Median')

ax.set_ylim([0.6,1.6])
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))

#ax.set_xlim([r_idx[0], r_idx[-1]])
ax.set_xlim([
    pd.to_datetime(r_idx[0]),
    pd.to_datetime(PLOT_MAX_DATE)
    ])

date_form = DateFormatter("%d.%m.\n%Y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2, byweekday=0))
ax.xaxis.set_minor_locator(mdates.DayLocator())


ax.tick_params(which='minor', length=0, width=0)
ax.tick_params(axis=u'both', labelsize=16, labelcolor = SLATE)
ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))

ax.set_ylabel('Wochentagkorrigierte 7-Tage Reproduktionszahl R', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
ax.set_xlabel('Geschätztes Erkrankungsdatum', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)

leg_handles, leg_labels = ax.get_legend_handles_labels()
leg_handles = np.asarray(leg_handles)
leg_labels = np.asarray(leg_labels)

if DO_EXTRAPOLATION:
    leg_order = [6, 8, 10, 3, 7, 9, 11, 4, 0, 1, 2, 5]
    ax.legend(leg_handles[leg_order], leg_labels[leg_order], 
              loc=2, fontsize=14, ncol=3)
else:
    leg_order = [5, 6, 7, 3, 0, 1, 2, 4]    
    ax.legend(leg_handles[leg_order], leg_labels[leg_order], 
              loc=2, fontsize=14, ncol=2)
    
  
# Rmin = old_boundaries['min'].dropna().iloc[-1]
# Rmed = old_boundaries['med'].dropna().iloc[-1]
# Rmax = old_boundaries['max'].dropna().iloc[-1]
# R50lo = old_boundaries['iq50_lo'].dropna().iloc[-1]
# R50hi = old_boundaries['iq50_hi'].dropna().iloc[-1]
# R95lo = old_boundaries['iq95_lo'].dropna().iloc[-1]
# R95hi = old_boundaries['iq95_hi'].dropna().iloc[-1]


Ryof = 0.01
Rynum = 0.1
Ryden = 3
#Rtext_date = dataset_date_range[-1] - pd.DateOffset(days = min_lut-2)
Rtext_date = dataset_date - pd.DateOffset(days = min_lut-2-1*3)

plt.text(Rtext_date, Rmed, 
         '$R_{{med}}: {:.3f}$'.format(Rmed),
         color = 'k', verticalalignment='center', fontsize=18)
         
plt.text(Rtext_date, Rmed + (Ryof + 1.0*Rynum/Ryden), 
         '$R_{{50\%}}^+: {:.2f}$'.format(R50hi),
         color = 'b', verticalalignment='center', fontsize=11.5)

plt.text(Rtext_date, Rmed - (Ryof + 1.0*Rynum/Ryden), 
         '$R_{{50\%}}^-: {:.2f}$'.format(R50lo),
         color = 'b', verticalalignment='center', fontsize=11.5)

plt.text(Rtext_date, Rmed + (Ryof + 2.0*Rynum/Ryden), 
         '$R_{{95\%}}^+: {:.2f}$'.format(R95hi),
         color = 'r', verticalalignment='center', fontsize=11.5)

plt.text(Rtext_date, Rmed - (Ryof + 2.0*Rynum/Ryden), 
         '$R_{{95\%}}^-: {:.2f}$'.format(R95lo),
         color = 'r', verticalalignment='center', fontsize=11.5)

plt.text(Rtext_date, Rmed + (Ryof + 3.0*Rynum/Ryden), 
         '$R_{{max}}: {:.2f}$'.format(Rmax),
         color = (0.5, 0.5, 0.5), verticalalignment='center', fontsize=11.5)

plt.text(Rtext_date, Rmed - (Ryof + 3.0*Rynum/Ryden), 
         '$R_{{min}}: {:.2f}$'.format(Rmin),
         color = (0.5, 0.5, 0.5), verticalalignment='center', fontsize=11.5)
         

ax2 = fig.add_subplot(gs[1, 0])

ax2.axis('off')


if dataset_date_range[0].year == dataset_date_range[-1].year:
    Datenstand_range_str = (
        dataset_date_range[0].strftime('%d.%m.-') + 
        dataset_date_range[-1].strftime('%d.%m.%Y') )
else:
    Datenstand_range_str = (
        dataset_date_range[0].strftime('%d.%m.%y-') + 
        dataset_date_range[-1].strftime('%d.%m.%Y') )


plt.text(0, 0.05,
    'Datenquelle:\n' + 
    'Robert Koch-Institut (RKI), an der Heiden, Matthias (2021): SARS-CoV-2-Nowcasting und -R-Schaetzung, Berlin: Zenodo. DOI:10.5281/zenodo.4680400\n'+
    'URL: https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung ; ' +
    'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; eigene Berechnung/eigene Darstellung;\n' +
    'Datenlizenz CC-BY 4.0 International               '+
    '$^{*)}$ gleitender geometrischer Mittelwert (Wurzel der Produkte)',
    fontsize=11.5)

if False:
    plt.show()
else:
    exp_full_fname = '{:s}{:s}_{:03d}.png'.format(
        OUTPUT_DIR, 'Nowcast_Var', i)
    
    print('Saving ' + exp_full_fname)
    try:
        fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
    except:
        fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)
    #fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)
    
    fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
    display(Image(filename=exp_full_fname))
    plt.close()
    
    
    
# %%
if True:
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(2, 1, figure=fig, 
                       height_ratios = [14, 3],
                       hspace = 0.1)
    ax = fig.add_subplot(gs[0, 0])
    
    fig.suptitle('COVID-19 - Ableitung des Wochentagkorrigierten Punktschätzers$^{{*)}}$ des RKI 7-Tage-R - {:s}'.format(dataset_date_str),
                    horizontalalignment='center',
                    verticalalignment='center', 
                    fontsize=21, color=SLATE, y=0.91)
    
    if DO_EXTRAPOLATION:
        plt.fill_between(x[:-4], np.zeros((x.size-4,)), np.diff(y[:-3]), facecolor=(0.0,0.0,1.0))
    else:
        plt.fill_between(x[:-1], np.zeros((x.size-1,)), np.diff(y), facecolor=(0.0,0.0,1.0))
    
    
    
    ax.set_ylim([-0.05, 0.05])
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.002))
    
    #ax.set_xlim([r_idx[0], r_idx[-1]])
    ax.set_xlim([
        pd.to_datetime(r_idx[0]),
        pd.to_datetime(PLOT_MAX_DATE)
        ])
    
    date_form = DateFormatter("%d.%m.\n%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2, byweekday=0))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    
    ax.tick_params(which='minor', length=0, width=0)
    ax.tick_params(axis=u'both', labelsize=16, labelcolor = SLATE)
    ax.grid(True, which='major', axis='both', linestyle='-',  color=(0.85,0.85,0.85))
    ax.grid(True, which='minor', axis='both', linestyle='-',  color=(0.95,0.95,0.95))
    
    ax.set_ylabel('Tägliche Änderung der W.k. 7-Tage Reproduktionszahl R', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    ax.set_xlabel('Geschätztes Erkrankungsdatum', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    
    
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.axis('off')
    
    
    if dataset_date_range[0].year == dataset_date_range[-1].year:
        Datenstand_range_str = (
            dataset_date_range[0].strftime('%d.%m.-') + 
            dataset_date_range[-1].strftime('%d.%m.%Y') )
    else:
        Datenstand_range_str = (
            dataset_date_range[0].strftime('%d.%m.%y-') + 
            dataset_date_range[-1].strftime('%d.%m.%Y') )
    
    
    plt.text(0, 0.05,
        'Datenquelle:\n' + 
        'Robert Koch-Institut (RKI), an der Heiden, Matthias (2021): SARS-CoV-2-Nowcasting und -R-Schaetzung, Berlin: Zenodo. DOI:10.5281/zenodo.4680400\n'+
        'URL: https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung ; ' +
        'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; eigene Berechnung/eigene Darstellung;\n' +
        'Datenlizenz CC-BY 4.0 International               '+
        '$^{*)}$ gleitender geometrischer Mittelwert (Wurzel der Produkte)',
        fontsize=11.5)
    
    if False:
        plt.show()
    else:
        exp_full_fname = '{:s}{:s}_{:03d}_dRdt.png'.format(
            OUTPUT_DIR, 'Nowcast_Var', i)
        
        print('Saving ' + exp_full_fname)
        try:
            fig_util.set_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
        except:
            fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)
        #fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)
        
        fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
        display(Image(filename=exp_full_fname))
        plt.close()