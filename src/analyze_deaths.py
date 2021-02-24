# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:23:06 2020

@author: David
"""

import sys
sys.path.append('.')

import os
import inspect

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sig
import scipy.stats as stat
import scipy.interpolate as intp
import scipy.optimize as sopt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from matplotlib import gridspec

import hashlib

from npgeo_reader import NpgeoReader

import fig_util
from IPython.display import display, Image

import json

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __dir__ = dict.keys



ALL_AGE_GROUPS = ['A00-A04', 'A05-A14', 'A15-A34', 'A35-A59', 'A60-A79', 'A80+']
AGE_DIRS = {
    'A00-A04': '00-04',
    'A05-A14': '05-14', 
    'A15-A34': '15-34', 
    'A35-A59': '35-59', 
    'A60-A79': '60-79', 
    'A80+': '80+'
    }

SLATE = (0.15, 0.15, 0.15)

SHAPE_MIN = 0.2

NPGEO_FILE_LIST = [*NpgeoReader().get_datafile_list().keys()]

assert len(NPGEO_FILE_LIST) > 0, "No NPGEO/RKI data files found!"

LATEST_FILE = NPGEO_FILE_LIST[-1]


script_full_filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(script_full_filename))
BASE_DIR = os.path.abspath(script_path + '\\..') + '\\'


#INPUT_DATA_RANGE = ['2020-05-26', LATEST_FILE]
INPUT_DATA_RANGE = ['2021-02-01', LATEST_FILE]



MODEL_PARAM = dotdict(
    {
        'INTERPOL_SCALE': 'Const',
        'INTERPOL_SHAPE': 'Const',
        'INTERPOL_LAG': 'Const',
        'SPLINE_R': 12, #12 #26 #5
        'MIN_POINTS_FOR_FIT':  
            {
                'A00-A04': 21,
                'A05-A14': 21,
                'A15-A34': 21,
                'A35-A59': 9, #10,
                'A60-A79': 5, #7,
                'A80+': 5
            }, 
        'SCALE_ERR_VEC': False,
        'DEFORM_FIRST_N_DAYS': 0,
        'WB_SCALE_BASE': 
            {
                'A00-A04': 50,
                'A05-A14': 50,
                'A15-A34': 50,
                'A35-A59': 38,
                'A60-A79': 27,
                'A80+': 17
            }
    })


    
    
    





DO_NOT_OUTPUT_PLOTS = False
DO_ANIMATION_PLOTS = False
DO_LOG_PLOTS = False
DO_DEBUG_PLOTS = False

PLOT_DATE_RANGE = ['2020-10-01', '2021-02-25']
PLOT_DATE_RANGE_WEEKDAY = 3
ANIMATION_START_DATE = '2021-02-01'
ANIMATION_INTERVAL = 1


AGE_GROUPS = ALL_AGE_GROUPS[3:]
BL_FILTER = False
#BL_FILTER = 'Baden-Württemberg'


PLOT_YLIM_LIN_ARR = {
    'A80+': 900,
    'A60-A79': 350,
    'A35-A59': 50,
    'A15-A34': 7,
    'A05-A14': 7,
    'A00-A04': 7
}
PLOT_YLIM_LOG =  {
    'A80+': [1, 10000],
    'A60-A79': [1, 10000],
    'A35-A59': [0.1, 1000],
    'A15-A34': [0.1, 100],
    'A05-A14': [0.1, 100],
    'A00-A04': [0.1, 100]
}

PLOT_LEGEND_NCOLS = 2


# %% Hash function

def to_hash(d):
    return hashlib.md5(json.dumps(d).encode('utf-8')).hexdigest()

# %% Preprocessing / Loading of preprocessed data


assert(Path(BASE_DIR).is_dir())

plot_output_dir = BASE_DIR + 'output\\NPGEO\\nowcast\\'
base_proc_data_dir = BASE_DIR + 'data\\NPGEO\\nowcast\\'

if isinstance(BL_FILTER, str):
    plot_output_dir += BL_FILTER + '\\'
    base_proc_data_dir += BL_FILTER + '\\'

rki_reader = NpgeoReader()

dataset_date_range = pd.date_range(*INPUT_DATA_RANGE)
nowcast_date_range = pd.date_range(
    dataset_date_range[0] - pd.DateOffset(days=1), 
    dataset_date_range[-1] - pd.DateOffset(days=1))

proc_data_dir = base_proc_data_dir + nowcast_date_range[0].strftime('%Y-%m-%d') + '\\'

max_days_delay = (dataset_date_range[-1] - nowcast_date_range[0]).days
days_delay_range = pd.Int64Index(
    range(max_days_delay, 0, -1), 
    name='MeldedatumAlter')

assert(nowcast_date_range.size == max_days_delay)

dataset_template = np.zeros((max_days_delay, max_days_delay))
dataset_template[np.tril_indices(max_days_delay, -1)] = np.nan

if dataset_date_range[0].year == dataset_date_range[-1].year:
    Datenstand_range_str = (
        dataset_date_range[0].strftime('%d.%m.-') + 
        dataset_date_range[-1].strftime('%d.%m.%Y') )
else:
    Datenstand_range_str = (
        dataset_date_range[0].strftime('%d.%m.%y-') + 
        dataset_date_range[-1].strftime('%d.%m.%Y') )

deaths_input_df = {
    ag: pd.DataFrame(
        np.copy(dataset_template), 
        index   = nowcast_date_range, 
        columns = days_delay_range)
    for ag in ALL_AGE_GROUPS
    }

cases_input_df = {
    ag: pd.DataFrame(
        np.copy(dataset_template), 
        index   = nowcast_date_range, 
        columns = days_delay_range)
    for ag in ALL_AGE_GROUPS
    }


# create output/processed data directories, if not existing
Path(plot_output_dir).mkdir(parents=True, exist_ok=True)
Path(proc_data_dir).mkdir(parents=True, exist_ok=True)


for i in range(dataset_date_range.size):
    dataset_date = dataset_date_range[i]
    dataset_date_str = dataset_date.strftime('%Y-%m-%d')
    print(dataset_date_str)
    
    death_proc_files = {
            ag: Path(proc_data_dir + 'DEATHS_' + dataset_date_str + "_" + ag + ".parquet")
            for ag in ALL_AGE_GROUPS
        }
    cases_proc_files = {
            ag: Path(proc_data_dir + 'CASES_' + dataset_date_str + "_" + ag + ".parquet")
            for ag in ALL_AGE_GROUPS
        }
    death_prep_data = {}
    cases_prep_data = {}
    
    all_deaths_preprocessed = all([f.is_file() for f in [*death_proc_files.values()]])
    all_cases_preprocessed = all([f.is_file() for f in [*cases_proc_files.values()]])
    
    if all_deaths_preprocessed and all_cases_preprocessed:
        for ag in ALL_AGE_GROUPS:
            death_prep_data[ag] = pd.read_parquet(death_proc_files[ag])        
            cases_prep_data[ag] = pd.read_parquet(cases_proc_files[ag])
            
    else:        
        dataset_data = rki_reader[dataset_date_str]
        
        for ag in ALL_AGE_GROUPS:
            
            if isinstance(BL_FILTER, str):
                dblflt = dataset_data.Bundesland == BL_FILTER
            else:
                dblflt = True
            
            death_data0 = dataset_data.loc[
                    (dataset_data.Altersgruppe == ag) &                                 
                    (dataset_data.NeuerTodesfall>=0) &
                    dblflt, 
                    ['AnzahlTodesfall', 'Meldedatum']
                ].groupby('Meldedatum').sum()
                    
            valid_df_indizes = death_data0.index.intersection(nowcast_date_range)
            death_data1 = death_data0.loc[valid_df_indizes]
                        
            death_data2 = pd.DataFrame({
                    'MeldedatumAlter': death_data1.index.map(lambda md: (dataset_date-md).days),
                    'AnzahlTodesfall': death_data1.AnzahlTodesfall
                })
            
            death_data2.index.rename('Meldedatum', inplace = True)
            
            death_prep_data[ag] = death_data2.copy()
        
            death_data2.to_parquet(death_proc_files[ag])
            
            
            if isinstance(BL_FILTER, str):
                cblflt = dataset_data.Bundesland == BL_FILTER
            else:
                cblflt = True
            
            cases_data0 = dataset_data.loc[
                    (dataset_data.Altersgruppe == ag) &                                 
                    (dataset_data.NeuerFall>=0) &
                    cblflt, 
                    ['AnzahlFall', 'Meldedatum']
                ].groupby('Meldedatum').sum()
            
            valid_df_indizes = cases_data0.index.intersection(nowcast_date_range)
            cases_data1 = cases_data0.loc[valid_df_indizes]
                        
            cases_data2 = pd.DataFrame({
                    'MeldedatumAlter': cases_data1.index.map(lambda md: (dataset_date-md).days),
                    'AnzahlFall': cases_data1.AnzahlFall
                })
            
            cases_data2.index.rename('Meldedatum', inplace = True)
            
            cases_prep_data[ag] = cases_data2.copy()
        
            cases_data2.to_parquet(cases_proc_files[ag])
    
    total_max_deaths = 0
    total_max_cases = 0
    for ag in ALL_AGE_GROUPS:
        death_data2 = death_prep_data[ag]
        valid_df_indizes = death_data2.index
        
        death_data3 = death_data2.pivot(columns='MeldedatumAlter', values='AnzahlTodesfall')
        death_data3.fillna(0, inplace=True)
                
        valid_df_columns = death_data3.columns.intersection(days_delay_range)
        death_data4 = death_data3.loc[:, valid_df_columns]
                    
        deaths_input_df[ag].loc[valid_df_indizes, valid_df_columns] += death_data4.loc[valid_df_indizes, valid_df_columns]

        new_max_deaths = death_data2.loc[(death_data2.MeldedatumAlter <= max_days_delay), 'AnzahlTodesfall'].sum()
        total_max_deaths += new_max_deaths
        
        
        cases_data2 = cases_prep_data[ag]
        valid_df_indizes = cases_data2.index
        
        cases_data3 = cases_data2.pivot(columns='MeldedatumAlter', values='AnzahlFall')
        cases_data3.fillna(0, inplace=True)
                
        valid_df_columns = cases_data3.columns.intersection(days_delay_range)
        cases_data4 = cases_data3.loc[:, valid_df_columns]
                    
        cases_input_df[ag].loc[valid_df_indizes, valid_df_columns] += cases_data4.loc[valid_df_indizes, valid_df_columns]

        new_max_cases = cases_data2.loc[(cases_data2.MeldedatumAlter <= max_days_delay), 'AnzahlFall'].sum()
        total_max_cases += new_max_cases        
        
        print('  {:7s}:  {:5.0f}  {:7.0f}  {:7.3f}%'.format(ag, new_max_deaths, new_max_cases, 100*new_max_deaths/new_max_cases))
    
    
    print('  {:7s}:  {:5.0f}  {:7.0f}  {:7.3f}%'.format('A00+', total_max_deaths, total_max_cases, 100*total_max_deaths/total_max_cases))


# %% nowcast helper functions (model, create start parameter, error function, etc.)


def create_empty_xvec(data_mat, r, ag):
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        n = data_mat.shape[0] - MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        n = data_mat.shape[0] - MODEL_PARAM.MIN_POINTS_FOR_FIT
    k = 0
    for p in [MODEL_PARAM.INTERPOL_LAG, 
              MODEL_PARAM.INTERPOL_SHAPE,
              MODEL_PARAM.INTERPOL_SCALE]:
        if (p == 'Spline'):
            k += r+1
        elif (p == 'Full'):
            k += n
        elif (p == 'Const'):
            k += 1
        # else default
        
    if isinstance(MODEL_PARAM.DEFORM_FIRST_N_DAYS, int) and MODEL_PARAM.DEFORM_FIRST_N_DAYS > 0:
        k += MODEL_PARAM.DEFORM_FIRST_N_DAYS
    
    #k = int(MODEL_PARAM.INTERPOL_LAG + MODEL_PARAM.INTERPOL_SHAPE + MODEL_PARAM.INTERPOL_SCALE)
    return np.zeros(n + k)

def split_xvec(xvec, n0, r, ag):
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT
    #m = (n-1) / r
    
    tvec = np.linspace(0, n-1, r+1)
    t0 = np.arange(n)
    
    k0 = 0
    if MODEL_PARAM.INTERPOL_LAG == 'Spline':
        lvec = -(1 + xvec[k0:(k0+r+1)]) # loc
        lvec[lvec > 0.0] = -lvec[lvec > 0.0]
        
        lfnc = intp.interp1d(tvec, lvec)        
        l0 = lfnc(t0)
        
        k0 = k0 + r+1
    elif MODEL_PARAM.INTERPOL_LAG == 'Full':
        l0 = -(1 + xvec[k0:(k0+n)]) # loc
        l0[l0 > 0.0] = -l0[l0 > 0.0]
        
        k0 = k0 + n
    elif MODEL_PARAM.INTERPOL_LAG == 'Const':
        lval = -(1 + xvec[k0])
        
        if lval > 0.0:
            lval = -lval
        l0 = np.full((n), lval)

        k0 = k0 + 1
    else:
        l0 = np.zeros((n)) # default zero
        
    if MODEL_PARAM.INTERPOL_SHAPE == 'Spline':
        cvec = 1.0 + xvec[k0:(k0+r+1)] # loc
        cvec[cvec < SHAPE_MIN] = SHAPE_MIN + (SHAPE_MIN - cvec[cvec < SHAPE_MIN])
        
        cfnc = intp.interp1d(tvec, cvec)        
        c0 = cfnc(t0)
        
        k0 = k0 + r+1        
    elif MODEL_PARAM.INTERPOL_SHAPE == 'Full':
        c0 =  1.0 + xvec[k0:(k0+n)] # loc
        c0[c0 < SHAPE_MIN] = SHAPE_MIN + (SHAPE_MIN - c0[c0 < SHAPE_MIN])
        
        k0 = k0 + n
    elif MODEL_PARAM.INTERPOL_SHAPE == 'Const':
        cval = 1.0 + xvec[k0]
        
        if cval < SHAPE_MIN:
            cval = SHAPE_MIN + (SHAPE_MIN - cval)
        c0 = np.full((n), cval)
        
        k0 = k0 + 1
    else:
        c0 = np.ones((n))  # default one

    if MODEL_PARAM.INTERPOL_SCALE == 'Spline':
        svec = MODEL_PARAM.WB_SCALE_BASE[ag] + xvec[k0:(k0+r+1)] # loc
        svec[svec < 1.0] = 1.0 + (1.0 - svec[svec < 1.0])
        
        sfnc = intp.interp1d(tvec, svec)        
        s0 = sfnc(t0)
                
        k0 = k0 + r+1
        
    elif MODEL_PARAM.INTERPOL_SCALE == 'Full':
        s0 =  MODEL_PARAM.WB_SCALE_BASE[ag] + xvec[k0:(k0+n)] # loc
        s0[s0 < 1.0] = 1.0 + (1.0 - s0[s0 < 1.0])
        
        k0 = k0 + n
    elif MODEL_PARAM.INTERPOL_SCALE == 'Const':
        sval = MODEL_PARAM.WB_SCALE_BASE[ag] + xvec[k0]
        
        if sval < 1.0:
            sval = 1.0 + (1.0 - sval)        
        s0 = np.full((n), sval)
        
        k0 = k0 + 1
    else:
        s0 = np.full((n), MODEL_PARAM.WB_SCALE_BASE[ag]) # default
    
    a0 = 1 + 10 * xvec[k0:k0+n]
    k0 = k0 + n
        
    if isinstance(MODEL_PARAM.DEFORM_FIRST_N_DAYS, int) and MODEL_PARAM.DEFORM_FIRST_N_DAYS > 0:
        d0 = xvec[k0:k0+MODEL_PARAM.DEFORM_FIRST_N_DAYS]
    else:
        d0 = np.empty((0))
    #a0[a0 < 0.0] = -a0[a0 < 0.0]
    
    return l0, c0, s0, a0, d0


def calc_model_mat(xvec, data_mat, r, ag, full = False):
    n0 = data_mat.shape[0] 
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT
    #m = (n-1) / r
    
    l0, c0, s0, a0, d0 = split_xvec(xvec, n0, r, ag)
    
    res_mat = np.empty((n0, n0))
    res_mat[:] = np.nan
    
    if not full:
        for i in range(n):
            swe = stat.weibull_min.cdf(
                np.linspace(1, n0-i, n0-i), 
                c0[i], 
                loc = l0[i],
                scale = s0[i])
            if d0.size > 0:
                swe[:min(d0.size, n0-i)] += d0[:min(d0.size, n0-i)]
                
            res_mat[i, :(n0-i)] = (data_mat[i, n0-1-i] + a0[i]) * swe
    else:
        for i in range(n):
            swe = stat.weibull_min.cdf(
                np.linspace(1, n0, n0), 
                c0[i], 
                loc = l0[i],
                scale = s0[i])
            if d0.size > 0:
                swe[:d0.size] += d0
                
            res_mat[i, :] = (data_mat[i, n0-i-1] + a0[i]) * swe
            
    return res_mat


def calc_model_weib(xvec, data_mat, r, ag, date_range):
    n0 = data_mat.shape[0]
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT
    #m = (n-1) / r
    
    l0, c0, s0, a0, d0 = split_xvec(xvec, n0, r, ag)
    t0 = np.arange(n)
    
    params = pd.DataFrame(
        np.vstack((l0, c0, s0, a0)).transpose(),
        index = t0,
        columns = ['Lag', 'Shape', 'Scale', 'Amplitude']
        )
    
    cdf_dist_mat = np.empty((n0, n))
    pdf_dist_mat = np.empty((n0, n))
    for i in range(n):
        cdf_dist_mat[:, i] = stat.weibull_min.cdf(
                np.linspace(1, n0, n0),
                c0[i], 
                loc = l0[i],
                scale = s0[i])
        if d0.size > 0:
            cdf_dist_mat[:d0.size, i] += d0
            
        pdf_dist_mat[:, i] = np.diff(np.insert(cdf_dist_mat[:, i], 0, 0))
        
    return {
        'Parameter': params,
        'Deformation': pd.DataFrame(
            d0,
            index = pd.RangeIndex(0, d0.size),
            columns = ['Deformation']
            ),
        'CDF': pd.DataFrame(
            cdf_dist_mat,
            index = np.linspace(1, n0, n0),
            columns = date_range
            ),
        'PDF': pd.DataFrame(
            pdf_dist_mat,
            index = np.linspace(1, n0, n0) - 0.5,
            columns = date_range
            )
        }


def calc_estimated_vec(xvec, data_mat, r, ag, date_idx = None):
    n0 = data_mat.shape[0]
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT
    
    if date_idx is None:
        # dataset_date_range = [0, infinity] => CDF always 100%
        l0, c0, s0, a0, d0 = split_xvec(xvec, n0, r, ag)
        return np.asarray([
            data_mat[i, n0-1-i] + a0[i] for i in range(n)])
    else:
        model_mat = calc_model_mat(xvec, data_mat, r, ag, True)
        return model_mat[:n, date_idx]
        
def calc_estimated_mat(xvec, data_mat, r, ag):
    n0 = data_mat.shape[0]
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT

    model_mat = calc_model_mat(xvec, data_mat, r, ag, True)
    return model_mat[:n, :]
        


def calc_original_vec(data_mat):
    n0 = data_mat.shape[0]
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        n = n0 - MODEL_PARAM.MIN_POINTS_FOR_FIT
    return np.asarray([
        data_mat[i, n0-1-i] for i in range(n)])


def err_fnc(xvec, data_mat, r, ag):
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        mpff = MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        mpff = MODEL_PARAM.MIN_POINTS_FOR_FIT
        
    n0 = data_mat.shape[0]
    n = n0 - mpff
    res_mat = calc_model_mat(xvec, data_mat, r, ag)
    err_mat = data_mat - res_mat
    
    err_vec = np.empty(int((n0*n0+n0 - mpff*mpff-mpff)/2))
    
    l0, c0, s0, a0, d0 = split_xvec(xvec, n0, r, ag)
    
    j = 0
    scale = 1.0
    for i in range(n):
        if MODEL_PARAM.SCALE_ERR_VEC:
            scale = (data_mat[i, n0-i-1] + a0[i])
            
        err_vec[j:(j+n0-i)] = (err_mat[i, :(n0-i)])/scale/np.sqrt(n0-i)
        j = j + n0-i
                
    err_fnc.counter = err_fnc.counter + 1
    
    if (err_fnc.counter % xvec.shape[0])==0:
        fmt = lambda y: np.array2string(y, formatter={'float_kind': lambda x: "%14.8f" % x})
        print('Err Fnc Call #{:d}:\nVariance:\n{:s}'.format(
            err_fnc.counter,
            fmt( np.sqrt(np.dot(err_vec,err_vec)) )))
        
        if MODEL_PARAM.INTERPOL_LAG == 'Spline':
            sidz = np.round(np.linspace(0, l0.size-1, MODEL_PARAM.SPLINE_R+1)).astype(int)
            print('Lag (Spline):\n' + fmt(l0[sidz]))
        elif MODEL_PARAM.INTERPOL_LAG == 'Full':
        	print('Lag (Full):\n' + fmt(l0))
        elif MODEL_PARAM.INTERPOL_LAG == 'Const':
        	print('Lag (Const):\n' + fmt(l0[0]))
        else:
        	print('Lag (Default):\n' + fmt(l0[0]))
            
        if MODEL_PARAM.INTERPOL_SHAPE == 'Spline':
            sidz = np.round(np.linspace(0, c0.size-1, MODEL_PARAM.SPLINE_R+1)).astype(int)
            print('Shape (Spline):\n' + fmt(c0[sidz]))
        elif MODEL_PARAM.INTERPOL_SHAPE == 'Full':
        	print('Shape (Full):\n' + fmt(c0))
        elif MODEL_PARAM.INTERPOL_SHAPE == 'Const':
        	print('Shape (Const):\n' + fmt(c0[0]))
        else:
        	print('Shape (Default):\n' + fmt(c0[0]))
            
        if MODEL_PARAM.INTERPOL_SCALE == 'Spline':
            sidz = np.round(np.linspace(0, s0.size-1, MODEL_PARAM.SPLINE_R+1)).astype(int)
            print('Scale (Spline):\n' + fmt(s0[sidz]))
        elif MODEL_PARAM.INTERPOL_SCALE == 'Full':
        	print('Scale (Full):\n' + fmt(s0))
        elif MODEL_PARAM.INTERPOL_SCALE == 'Const':
        	print('Scale (Const):\n' + fmt(s0[0]))
        else:
        	print('Scale (Default):\n' + fmt(s0[0]))
            
        if d0.size > 0:
            print('Deform 1st n days:\n' + fmt(d0))
        
        #print('Amplitudes:\n' + fmt(a0) + '\n')
        print('\n')
        
    return err_vec
    #return err_vec.dot(err_vec)

# %% Nowcast (Calculate, Plot, Save and Load) functions

def do_nowcast(input_data, ag, do_n_days_in_past = 0):    
    if isinstance(MODEL_PARAM.MIN_POINTS_FOR_FIT, dict):
        mpff = MODEL_PARAM.MIN_POINTS_FOR_FIT[ag]
    else:
        mpff = MODEL_PARAM.MIN_POINTS_FOR_FIT
        
    input_matrix = input_data[ag].to_numpy(copy = True)
    input_index = input_data[ag].index
    
    if do_n_days_in_past > 0:
        input_matrix = input_matrix[:-do_n_days_in_past, do_n_days_in_past:]
        input_matrix[np.tril_indices(input_matrix.shape[0], -1)] = np.nan            
        input_index = input_index[:-do_n_days_in_past]
        
    
    averaged_matrix = sig.correlate(input_matrix,
                          1.0/7.0*np.ones((7,1)), 
                          mode='valid',
                          method='direct')
    averaged_matrix = averaged_matrix[:,:5:-1]
    averaged_index = input_index[3:-3]
    nowcast_index = averaged_index[:-mpff]
    
    
    err_fnc.counter = 0
    start_lq_vec = create_empty_xvec(averaged_matrix, MODEL_PARAM.SPLINE_R, ag)
    
    print(err_fnc(start_lq_vec, averaged_matrix, MODEL_PARAM.SPLINE_R, ag))
    lq_result = sopt.least_squares(
        err_fnc, 
        start_lq_vec,
        method = 'lm',
        args = (averaged_matrix, MODEL_PARAM.SPLINE_R, ag))
    # lq_result = sopt.minimize(
    #     fun = err_fnc, 
    #     x0 = start_lq_vec,
    #     args = (averaged_matrix, MODEL_PARAM.SPLINE_R, ag),
    #     method = 'BFGS'     #'Nelder-Mead'
    #     )
    nowcast_fitted_parameters = lq_result.x
        
        
    averaged_data = pd.DataFrame(
        averaged_matrix[:-mpff], #, :-mpff],
        index = nowcast_index,
        columns = [c+1 for c in range(averaged_index.size)])
    
    nowcast_data = pd.DataFrame(
        calc_estimated_mat(
            nowcast_fitted_parameters, 
            averaged_matrix, 
            MODEL_PARAM.SPLINE_R, 
            ag),
        index = nowcast_index,
        columns = [c+1 for c in range(averaged_index.size)])

    result = calc_model_weib(
        nowcast_fitted_parameters, 
        averaged_matrix, 
        MODEL_PARAM.SPLINE_R, 
        ag,
        nowcast_index
        )
    
    result['AveragedData'] = averaged_data
    result['NowcastData'] = nowcast_data
    
    start_date = (input_index[0] + pd.DateOffset(days=1))
    end_date = (input_index[-1] + pd.DateOffset(days=1))
    result['FirstDatasetDate'] = start_date
    result['LastDatasetDate'] = end_date
    result['Agegroup'] = ag
    
    if isinstance(BL_FILTER, str):
        result['Bundesland'] = BL_FILTER
    else:
        result['Bundesland'] = 'Deutschland'
    
    return result
    

## Hacky: using globals 
##   * plot_output_dir
def plot_nowcast(nowcast_result, do_linear = True):
    ag = nowcast_result['Agegroup']
    pdata = nowcast_result['NowcastData']
    
    csh = nowcast_result['Parameter'].Shape.iloc[-1]
    ssc = nowcast_result['Parameter'].Scale.iloc[-1]
    d25 = int(np.ceil(stat.weibull_min.ppf(0.25, csh, scale=ssc)))
    d50 = int(np.ceil(stat.weibull_min.ppf(0.50, csh, scale=ssc)))
    d75 = int(np.ceil(stat.weibull_min.ppf(0.75, csh, scale=ssc)))
    d95 = int(np.ceil(stat.weibull_min.ppf(0.95, csh, scale=ssc)))
    
    plotparts = np.full((4), True)
    
    
    if (not d25 in pdata.columns):
        plotparts[:] = False
    elif (not d50 in pdata.columns):
        plotparts[1:] = False
    elif (not d75 in pdata.columns):
        plotparts[2:] = False
    elif (not d95 in pdata.columns):
        plotparts[3] = False
        
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1, figure=fig, 
                           height_ratios = [5, 1],
                           hspace = 0.1)
    
            
    ax = fig.add_subplot(gs[0, 0])
    
    if isinstance(BL_FILTER, str):
        secsupt = BL_FILTER
    else:
        secsupt = 'Deutschland'
        
    fig.suptitle('COVID-19 - ' + secsupt + ' - Nowcast (' + 
                 (nowcast_result['LastDatasetDate']).strftime('%d.%m.%Y') + 
                 ') der Verstorbenen nach Meldedatum - Altersgruppe ' +
              (ag.replace('A','')),
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=22, color=SLATE, y=0.91)
    
    
    # NOWCAST
    
    d25min = max(min(d25-1, pdata.columns[-1]), 1)
    d50min = max(min(d50-1, pdata.columns[-1]), 1)
    d75min = max(min(d75-1, pdata.columns[-1]), 1)
    d95min = max(min(d95-1, pdata.columns[-1]), 1)
    
    print('d25min = {:d}'.format(d25min))
    print('d50min = {:d}'.format(d50min))
    print('d75min = {:d}'.format(d75min))
    print('d95min = {:d}'.format(d95min))
    
    
    hvecmin = pdata[1]
    ax.bar(x = pdata.index, height = hvecmin, 
           color = (1, 0, 1), edgecolor = (1, 0, 1), width=1)
    
    hvecmax = pdata[d25min]
    ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
            color=(1, 0, 1), edgecolor = (1, 0, 1), width=1)
    
    if plotparts[0]:
        hvecmin = hvecmax
        hvecmax = pdata[d50min]
        ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
                color=(0.8, 0, 0.8), edgecolor = (0.8, 0, 0.8), width=1)
    
    if plotparts[1]:
        hvecmin = hvecmax
        hvecmax = pdata[d75min]
        ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
                color=(0.6, 0, 0.6), edgecolor = (0.6, 0, 0.6), width=1)
    
    if plotparts[2]:
        hvecmin = hvecmax
        hvecmax = pdata[d95min]
        ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
                color=(0.4, 0, 0.4), edgecolor = (0.4, 0, 0.4), width=1)

    if plotparts[3]:    
        hvecmin = hvecmax
        hvecmax = pdata[pdata.columns[-1]]
        ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
                color=(0.2, 0, 0.2), edgecolor = (0.2, 0, 0.2), width=1)
    
    
    # ORIGINAL DATA
    pdata = nowcast_result['AveragedData']
    
    max_idz = pdata.apply(pd.Series.last_valid_index, axis=1) - 1
    
    hvecmin = pdata[1]
    ax.bar(x = pdata.index, height = hvecmin, 
           color = (1, 1, 0), edgecolor = (1, 1, 0), width=1)

        
    hvecmax = np.asarray([
        pdata.iloc[k, min(max_idz[k], d25-1)] 
        for k in range(pdata.index.size)])
    ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
            color = (1, 1, 0), edgecolor = (1, 1, 0), width=1)
        
    if plotparts[0]:
        hvecmin = hvecmax
        hvecmax = np.asarray([
            pdata.iloc[k, min(max_idz[k], d50-1)] 
            for k in range(pdata.index.size)])
        ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
                color = (1, 0.5, 0), edgecolor = (1, 0.5, 0), width=1)

    if plotparts[1]:
        hvecmin = hvecmax
        hvecmax = np.asarray([
            pdata.iloc[k, min(max_idz[k], d75-1)]
            for k in range(pdata.index.size)])
        ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
                color = (1, 0, 0), edgecolor = (1, 0, 0), width=1)
    
    if plotparts[2]:
        hvecmin = hvecmax
        hvecmax = np.asarray([
            pdata.iloc[k, min(max_idz[k], d95-1)] 
            for k in range(pdata.index.size)])
        ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
                color = (0.5, 0, 0), edgecolor = (0.5, 0, 0), width=1)
    
    if plotparts[3]:
        hvecmin = hvecmax
        hvecmax = np.asarray([
            pdata.iloc[k, min(max_idz[k], pdata.shape[1]-1)] 
            for k in range(pdata.index.size)])
        ax.bar(x = pdata.index, height = hvecmax-hvecmin, bottom = hvecmin,
                color = (0, 0, 1), edgecolor = (0, 0, 1), width=1)
    
    
    
    date_form = DateFormatter("%d.%m.\n%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=PLOT_DATE_RANGE_WEEKDAY))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    
    if do_linear:
        ax.set_yscale('linear')
        ylimmax = PLOT_YLIM_LIN_ARR[ag]
        ax.set_ylim([0, PLOT_YLIM_LIN_ARR[ag]])
        
    else:
        ax.set_yscale('log')
        ylimmax = PLOT_YLIM_LOG[ag][1]
        ax.set_ylim(PLOT_YLIM_LOG[ag])
        
    ax.set_xlim([pd.to_datetime(PLOT_DATE_RANGE[0]),
                 pd.to_datetime(PLOT_DATE_RANGE[1])])
    ax.set_ylabel('Verstorbene mit pos. Befund an diesem Meldedatum', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    ax.set_xlabel('Meldedatum des pos. Befund', fontdict={'fontsize': 18}, fontsize=18, color = SLATE, labelpad=14)
    ax.tick_params(axis=u'both', which='major', labelsize=16, labelcolor = SLATE)
    ax.tick_params(axis=u'both', which='minor', length=0, width=0)
    
    
    
    
    
    custom_lines = [
        [
            Line2D([0], [0], color=(1.0, 1.0, 0.0), lw=4),
            Line2D([0], [0], color=(1.0, 0.5, 0.0), lw=4),
            Line2D([0], [0], color=(1.0, 0.0, 0.0), lw=4),
            Line2D([0], [0], color=(0.5, 0.0, 0.0), lw=4),
            Line2D([0], [0], color=(0.0, 0.0, 1.0), lw=4)
        ],
        [
            Line2D([0], [0], color=(1.0, 0.0, 1.0), lw=4),
            Line2D([0], [0], color=(0.8, 0.0, 0.8), lw=4),
            Line2D([0], [0], color=(0.6, 0.0, 0.6), lw=4),
            Line2D([0], [0], color=(0.4, 0.0, 0.4), lw=4),
            Line2D([0], [0], color=(0.2, 0.0, 0.2), lw=4)
        ]
    ]

    custom_legends = [
        [
            'Vorhandene Daten - Innerhalb {:d}-{:d} Tagen aVg*'.format(0,   d25-1),
            'Vorhandene Daten - Innerhalb {:d}-{:d} Tagen aVg*'.format(d25, d50-1),
            'Vorhandene Daten - Innerhalb {:d}-{:d} Tagen aVg*'.format(d50, d75-1),
            'Vorhandene Daten - Innerhalb {:d}-{:d} Tagen aVg*'.format(d75, d95-1),
            'Vorhandene Daten - Nach {:d}+ Tagen aVg*'.format(d95)
        ],
        [
            'Nowcast - Innerhalb {:d}-{:d} Tagen aVg*'.format(0,   d25-1),
            'Nowcast - Innerhalb {:d}-{:d} Tagen aVg*'.format(d25, d50-1),
            'Nowcast - Innerhalb {:d}-{:d} Tagen aVg*'.format(d50, d75-1),
            'Nowcast - Innerhalb {:d}-{:d} Tagen aVg*'.format(d75, d95-1),
            'Nowcast - Nach {:d}+ Tagen aVg*'.format(d95)
        ]
    ]
    
    if not plotparts[0]:
        leg_lines = custom_lines[0][0:1] + custom_lines[1][0:1]
        leg_texts = custom_legends[0][0:1] + custom_legends[1][0:1]
    elif not plotparts[1]:
        leg_lines = custom_lines[0][0:2] + custom_lines[1][0:2]
        leg_texts = custom_legends[0][0:2] + custom_legends[1][0:2]
    elif not plotparts[2]:
        leg_lines = custom_lines[0][0:3] + custom_lines[1][0:3]
        leg_texts = custom_legends[0][0:3] + custom_legends[1][0:3]
    elif not plotparts[3]:
        leg_lines = custom_lines[0][0:4] + custom_lines[1][0:4]
        leg_texts = custom_legends[0][0:4] + custom_legends[1][0:4]
    else:
        leg_lines = custom_lines[0][:] + custom_lines[1][:]
        leg_texts = custom_legends[0][:] + custom_legends[1][:]
        
    #print(leg_lines)
    ax.legend(leg_lines, leg_texts, ncol=PLOT_LEGEND_NCOLS,
              labelcolor=SLATE, loc='upper left', fontsize = 16
              )
    
    eps = 1e-10
    param_df = nowcast_result['Parameter']
    if (param_df.Lag.std()<eps) and (param_df.Shape.std()<eps) and (param_df.Scale.std()<eps):
        ax.text(0.99, 0.98,
                'Weibull:\nk = {:.5f}\nθ = {:.4f}d\nΔ = {:.2f}d'.format(
                    param_df.Shape.iloc[0],
                    param_df.Scale.iloc[0],
                    param_df.Lag.iloc[0]
                    ),
                ha = 'right', va = 'top', multialignment = 'left',
                transform=ax.transAxes,
                color=SLATE, fontsize = 18)
        
            
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.axis('off')
    
    
    
    if nowcast_result['FirstDatasetDate'].year == nowcast_result['LastDatasetDate'].year:
        Datenstand_range_str = (
            nowcast_result['FirstDatasetDate'].strftime('%d.%m.-') + 
            nowcast_result['LastDatasetDate'].strftime('%d.%m.%Y') )
    else:
        Datenstand_range_str = (
            nowcast_result['FirstDatasetDate'].strftime('%d.%m.%y-') + 
            nowcast_result['LastDatasetDate'].strftime('%d.%m.%Y') )
    
    
    plt.text(0, 0.3,
             'Datenquellen:\n' + 
             'Robert Koch-Institut (RKI), NPGEO Corona Hub, https://npgeo-corona-npgeo-de.hub.arcgis.com, ' +
             'Abfragedatum/Datenstand: ' + Datenstand_range_str + '; Datenlizenz by-2-0; ' +
             'eigene Berechnung/eigene Darstellung\n' +
             "NPGEO Direktlink: https://www.arcgis.com/home/item.html?id=f10774f1c63e40168479a1feb6c7ca74",
             fontsize=11.5)
    
    plt.text(0, 0.05, "'Datenlizenz by-2-0': https://www.govdata.de/dl-de/by-2-0\n" +
             "*: 'als Verstorben gemeldet'. Vergangene Tage zwischen pos. Befund & Meldung als gestorben beim RKI. Beinhaltet Meldeketten & Krankheitsverlauf bedingte Verzüge.",
             fontsize=11.5)
    
    
    if do_linear:
        fex = 'LIN'
    else:
        fex = 'LOG'
    
    last_ref_date = nowcast_result['LastDatasetDate'].strftime('%Y-%m-%d')
    
    exp_fname = ('COVID-19_NOWCAST_DEATHS_' + last_ref_date + 
            '_' + ag + '_' + fex)
    
    this_plot_dir =  '{:s}{:s}\\{:s}\\{:s}\\{:s}\\'.format(
        plot_output_dir, INPUT_DATA_RANGE[0],
        to_hash(MODEL_PARAM),
        AGE_DIRS[ag], fex)
    
    Path(this_plot_dir).mkdir(parents=True, exist_ok=True)
    
    exp_full_fname = '{:s}{:s}.png'.format(
        this_plot_dir, exp_fname)
    
    print('Saving ' + exp_fname)
    fig_util.force_fig_size(fig, (1920.0, 1080.0), dpi=100, pad_inches=0.35)
    
    fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
    display(Image(filename=exp_full_fname))
    plt.close()


class NowcastEncoder(json.JSONEncoder):
    PARQUET_DIR = '.'
    
    @staticmethod
    def setup(fdir):
        NowcastEncoder.PARQUET_DIR = fdir
        Path(fdir).mkdir(parents=True, exist_ok=True)
            
    def default(self, v):
        if isinstance(v, pd.DataFrame) and hasattr(v, 'fname'):            
            v.to_parquet(Path(NowcastEncoder.PARQUET_DIR + v.fname))
            
            rval = { 
                    '__class__': 'DataFrame', 
                    'parquet': v.fname
                    }
            if hasattr(v, 'columns_type'):
                rval['columns_type'] = v.columns_type
            if hasattr(v, 'columns_freqname'):
                rval['columns_freqname'] = v.columns_freqname
                
            return rval
        
        elif isinstance(v, pd.Timestamp):
            return { '__class__': 'Timestamp', 'value': v.strftime('%Y-%m-%d') }
        
        else:
            return super().default(v)

NowcastEncoder.setup(proc_data_dir + to_hash(MODEL_PARAM) + '\\')

def NowcastDecoder(d):
    if '__class__' in d:
        if d['__class__'] == 'DataFrame':
            v = pd.read_parquet(Path(NowcastEncoder.PARQUET_DIR + d['parquet']))  
            if 'columns_type' in d:
                if d['columns_type'] == 'DatetimeIndex':
                    if 'columns_freqname' in d:
                        v.columns = pd.DatetimeIndex(v.columns, freq = d['columns_freqname'])
                    else:
                        v.columns = pd.DatetimeIndex(v.columns)
                        
                elif d['columns_type'] == 'Int64Index':
                    v.columns = pd.Int64Index(v.columns.astype('int64'))
                
                elif d['columns_type'] == 'UInt64Index':
                    v.columns = pd.UInt64Index(v.columns.astype('uint64'))
                    
                elif d['columns_type'] == 'TimedeltaIndex':
                    v.columns = pd.TimedeltaIndex(v.columns.astype(pd.Timedelta))
                    
                elif d['columns_type'] == 'RangeIndex':
                    v.columns = pd.RangeIndex(
                        np.int64(d['columns_start']),
                        np.int64(d['columns_stop']),
                        np.int64(d['columns_step']))
            return v
            
        elif d['__class__'] == 'Timestamp':
            return pd.Timestamp(d['value'])
        else:
            raise ValueError('NowcastDecoder: Unexpected __class__ attribute!')
    
    return d


def StoreNowcast(data):
    cdate = data['LastDatasetDate'].strftime('%Y-%m-%d')
    ag = data['Agegroup']
    
    expdata = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            v2 = v.copy()
            v2.fname = 'NOWCAST_{:s}_{:s}_{:s}.parquet'.format(
                cdate, ag, k)
            
            if isinstance(v2.columns, pd.DatetimeIndex):
                v2.columns_type = 'DatetimeIndex'
                if not v2.columns.freq is None:
                    v2.columns_freqname = v2.columns.freq.name
                v2.columns = v2.columns.strftime('%Y-%m-%d')
                    
            elif isinstance(v2.columns, pd.TimedeltaIndex):
                v2.columns_type = 'TimedeltaIndex'
                v2.columns = v2.columns.astype('str')
                
            elif isinstance(v2.columns, pd.RangeIndex):
                v2.columns_type = 'RangeIndex'
                v2.columns_start = v2.columns.start
                v2.columns_stop = v2.columns.stop
                v2.columns_step = v2.columns.step
                v2.columns = v2.columns.astype('str')
                
            elif isinstance(v2.columns, pd.Int64Index):
                v2.columns_type = 'Int64Index'           
                v2.columns = v2.columns.astype('str') 
                    
            elif isinstance(v2.columns, pd.UInt64Index):
                v2.columns_type = 'UInt64Index'
                v2.columns = v2.columns.astype('str')
                
            elif not all(v2.columns.map(lambda s: isinstance(s, str))):
                raise ValueError('StoreNowcast: Unexpected column index datatype!')
            expdata[k] = v2
        else:
            expdata[k] = v

    fname = 'NOWCAST_' + cdate + '_' + ag + ".json"

    p = Path(NowcastEncoder.PARQUET_DIR + fname)
    with p.open("w", encoding ="utf-8") as f:
        json.dump(expdata, f, cls = NowcastEncoder)


def LoadNowcast(ag, cdate):
    fname = 'NOWCAST_' + cdate + '_' + ag + ".json"
    p = Path(NowcastEncoder.PARQUET_DIR + fname)
    
    with p.open("r", encoding ="utf-8") as f:
        data = json.load(f, object_hook = NowcastDecoder)
           
    return data

def ProcNowcastExists(ag, cdate):
    fname = 'NOWCAST_' + cdate + '_' + ag + ".json"
    p = Path(NowcastEncoder.PARQUET_DIR + fname)
    
    return p.is_file()
        

#%% MAIN Script



cur_md5 = to_hash(MODEL_PARAM)
NowcastEncoder.setup(proc_data_dir + cur_md5 + '\\')


if not (('nowcastslib' in globals()) or ('nowcastslib' in locals())):
    nowcastslib = {
        }


if not (('nowcasts' in globals()) or ('nowcasts' in locals())):
    nowcasts = {
        }


if not BL_FILTER in nowcastslib:
    nowcastslib[BL_FILTER] = {}
    
if not cur_md5 in nowcastslib[BL_FILTER]:
    nowcastslib[BL_FILTER][cur_md5] = {}

nowcasts = nowcastslib[BL_FILTER][cur_md5]

    
if DO_ANIMATION_PLOTS:
    LDAY_RANGE = (dataset_date_range[-1] - pd.to_datetime(ANIMATION_START_DATE)).days
else:
    LDAY_RANGE = 0
    
    
this_plot_dir_base =  '{:s}{:s}\\{:s}\\'.format(
        plot_output_dir, INPUT_DATA_RANGE[0],
        to_hash(MODEL_PARAM))
    
Path(this_plot_dir_base).mkdir(parents=True, exist_ok=True)

p = Path(this_plot_dir_base + 'parameter.json')
if not p.is_file():    
    with p.open("w", encoding ="utf-8") as f:
        json.dump(MODEL_PARAM, f)


p = Path(NowcastEncoder.PARQUET_DIR + 'parameter.json')
if not p.is_file():    
    with p.open("w", encoding ="utf-8") as f:
        json.dump(MODEL_PARAM, f)
        

    
for ag in sorted(AGE_GROUPS):

    if not (ag in nowcasts):
        nowcasts[ag] = {}

    for ldays in range(0, LDAY_RANGE+1, ANIMATION_INTERVAL): #range(0,8,7): #range(0,29,7): #range(43):
    
        curdate = dataset_date_range[-1] - pd.DateOffset(days=ldays)
        cdatestr = curdate.strftime('%Y-%m-%d')
        
        
        if not (cdatestr in nowcasts[ag]):
            
            if ProcNowcastExists(ag, cdatestr):
                print('Load {:s} @ {:s}'.format(ag, curdate.strftime('%d.%m.%Y')))
                nowcast_res = LoadNowcast(ag, cdatestr)
                
            else:
                print('Calc {:s} @ {:s}'.format(ag, curdate.strftime('%d.%m.%Y')))
                
                nowcast_res = do_nowcast(deaths_input_df, ag, ldays)
                
                StoreNowcast(nowcast_res)
                
            nowcasts[ag][cdatestr] = nowcast_res
        else:
            print('Reuse {:s} @ {:s}'.format(ag, curdate.strftime('%d.%m.%Y')))
            nowcast_res = nowcasts[ag][cdatestr]
            
            if not ProcNowcastExists(ag, cdatestr):
                StoreNowcast(nowcast_res)
        
        if DO_DEBUG_PLOTS:
            plt.figure(figsize=(16,9))
            ax = plt.subplot(1,1,1)
            plt.plot(nowcasts[ag][cdatestr]['NowcastData'].iloc[-42:].transpose(), color=(0.8,0.8,0.8), linestyle="solid", label='Nowcast Fit')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-42:-28].transpose(),'c', label='Letzte 29-42 Tage')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-28:-21].transpose(),'y', label='Letzte 22-28 Tage')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-21:-14].transpose(),'b', label='Letzte 15-21 Tage')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-14:-7].transpose(),'r', label='Letzte 8-14 Tage')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-7:].transpose(),'k', label='Letzte 7 Tage')
            mva = np.ceil(1.1*nowcasts[ag][cdatestr]['AveragedData'].iloc[-42:,:42].max().max())
            handles, labels = plt.gca().get_legend_handles_labels()
            newLabels, newHandles = [], []
            for handle, label in zip(handles, labels):
                if label not in newLabels:
                    newLabels.append(label)
                    newHandles.append(handle)
            plt.legend(newHandles, newLabels, fontsize=12)
            plt.xlim([0,42])
            plt.ylim([0,mva])
            xstart, xend = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(xstart, xend+1, 7))
            plt.title('Geglättet Eingabedaten & Nowcast Fit für ' + ag + ' @' + cdatestr, fontsize=18, color = SLATE)
            ax.set_ylabel('Geglättete, kommulierte Fälle im Reporting Triangle\ndes beobachteten Meldedatums', fontdict={'fontsize': 14}, fontsize=14, color = SLATE, labelpad=10)
            ax.set_xlabel('Vergangene Tage seit Meldedatum', fontdict={'fontsize': 14}, fontsize=14, color = SLATE, labelpad=10)
            ax.tick_params(axis=u'both', which='major', labelsize=12, labelcolor = SLATE)
            plt.show()
            
            plt.figure(figsize=(16,9))
            ax = plt.subplot(1,1,1)
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-42:-28].transpose()-nowcasts[ag][cdatestr]['NowcastData'].iloc[-42:-28].transpose(),'c', label='Letzte 29-42 Tage')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-28:-21].transpose()-nowcasts[ag][cdatestr]['NowcastData'].iloc[-28:-21].transpose(),'y', label='Letzte 22-28 Tage')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-21:-14].transpose()-nowcasts[ag][cdatestr]['NowcastData'].iloc[-21:-14].transpose(),'b', label='Letzte 15-21 Tage')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-14:-7].transpose()-nowcasts[ag][cdatestr]['NowcastData'].iloc[-14:-7].transpose(),'r', label='Letzte 8-14 Tage')
            plt.plot(nowcasts[ag][cdatestr]['AveragedData'].iloc[-7:].transpose()-nowcasts[ag][cdatestr]['NowcastData'].iloc[-7:].transpose(),'k', label='Letzte 7 Tage')

            handles, labels = plt.gca().get_legend_handles_labels()
            newLabels, newHandles = [], []
            for handle, label in zip(handles, labels):
                if label not in newLabels:
                    newLabels.append(label)
                    newHandles.append(handle)
            plt.legend(newHandles, newLabels, fontsize=12)
            
            plt.xlim([0,42])
            if ag == 'A35-A59':
                plt.ylim([-3,3])
            elif ag == 'A60-A79':
                plt.ylim([-10,10])
            elif ag == 'A80+':
                plt.ylim([-30,30])
            
            xstart, xend = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(xstart, xend+1, 7))
            plt.title('Abweichung geglättete Eingabedaten zu Nowcast Fit für ' + ag + ' @' + cdatestr, fontsize=18, color = SLATE)
            ax.set_ylabel('Abweichung von gefittetem Nowcast in Fällen', fontdict={'fontsize': 14}, fontsize=14, color = SLATE, labelpad=10)
            ax.set_xlabel('Vergangene Tage seit Meldedatum', fontdict={'fontsize': 14}, fontsize=14, color = SLATE, labelpad=10)
            ax.tick_params(axis=u'both', which='major', labelsize=12, labelcolor = SLATE)
            plt.show()
        
        if DO_NOT_OUTPUT_PLOTS:
            print('No plot')
        else:
            if DO_LOG_PLOTS:
                for PLOT_Y_LINEAR in [True, False]:
                    if PLOT_Y_LINEAR:
                        print('Plot linear')
                    else:
                        print('Plot log')
                    plot_nowcast(nowcasts[ag][cdatestr], PLOT_Y_LINEAR)
            else:
                print('Plot linear')
                plot_nowcast(nowcasts[ag][cdatestr], True)
                
           