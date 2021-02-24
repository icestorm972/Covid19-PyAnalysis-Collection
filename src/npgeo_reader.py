# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:56:58 2020

@author: David
"""

import os
import errno
import glob
import inspect
import re
import datetime

from collections import Counter

#import numpy as np
import pandas as pd




class NpgeoReader:
    __order_pattern = (r'(?:\{(year)\:[0-9]+d\})|'
                       r'(?:\{(month)\:[0-9]+d\})|'
                       r'(?:\{(day)\:[0-9]+d\})')
    
    
    @staticmethod
    def kreis_liste(bundeslaender = None):
        if bundeslaender is None:
            bundeslaender = ['Deutschland']
        elif isinstance(bundeslaender, str):
            bundeslaender = [bundeslaender]
        

    @staticmethod
    def __is_pattern_valid(pattern):
        if (pattern is None) or not isinstance(pattern, str):
            return False

        params = Counter([''.join(a) for a in 
                       re.findall(NpgeoReader.__order_pattern, pattern)])
        return (len(params) == 3 and params['year'] == 1 and 
            params['month'] == 1 and params['day'] == 1)
    
    @staticmethod
    def __check_and_normalize_path(input_path, base_path):    
        if input_path[0] == '.':
            input_path = base_path + '/' + input_path
            
        assert os.path.exists(input_path)
        return os.path.abspath(input_path)    
    
    
    @property 
    def filepattern(self):
        return self.__file_pattern
    
    @filepattern.setter
    def filepattern(self, value):
        if value is None:
            value = u'RKI_COVID19_{year:04d}-{month:02d}-{day:02d}.csv'
                    
        assert NpgeoReader.__is_pattern_valid(value)            
        self.__file_pattern = value
        
        
    # lut_path: relative to >script_path< or absolute
    @property 
    def lut_path(self):
        return self.__lut_path
    
    @lut_path.setter
    def lut_path(self, lut_path):
        if lut_path is None:
            lut_path = u'/../LUT'
        
        self.__lut_path = NpgeoReader.__check_and_normalize_path(lut_path, self.__script_path)
    
    
        
    # input_path: relative to >script_path< or absolute
    @property 
    def input_path(self):
        return self.__input_path
    
    @input_path.setter
    def input_path(self, input_path):
        if input_path is None:
            input_path = u'../data/NPGEO'
        
        self.__input_path = NpgeoReader.__check_and_normalize_path(input_path, self.__script_path)
        

    def load_LUT(self):    
        if self.__lut_path is None:
            raise ValueError('NpgeoReader lut_path was not set')
            
        if not os.path.exists(self.__lut_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.__lut_path)
        self.__bundeslaender = pd.read_csv(
            self.__lut_path + '/' + u"Bundesländer.tsv", 
            sep = '\t', decimal = '.', thousands = None, comment = '#', 
            index_col = 'Bundesland')
        
        self.__kreise = pd.read_csv(
            self.__lut_path + '/' + u"Kreise.tsv", 
            sep = '\t', decimal = '.', thousands = None, comment = '#', 
            index_col = 'RKI NPGEO Kreis')
        
        self.__kreise.index.rename('Kreis', inplace=True)
        
        self.__kreis_dict =  { 'Deutschland': self.__kreise.index.copy() }
        for b in self.__bundeslaender.index.to_numpy():
            bflt = self.__kreise.Bundesland == b
            self.__kreis_dict[b] = self.__kreise.loc[bflt].index.copy()
        self.__bundeslaender.loc["Deutschland", ['Abkürzung', 'Id']] = ['DE', 0]

    def __valid_setup(self):
        if ((self.__input_path is None) or (self.__lut_path is None) or
            (self.__file_pattern is None) or
            (self.__bundeslaender is None) or
            (self.__kreise is None)):
            return False
        
        if not os.path.exists(self.__input_path) or not os.path.exists(self.__lut_path):
            return False
        
        try:
            NpgeoReader.__is_pattern_valid(self.__file_pattern)
        except:
            return False
        
        if not isinstance(self.__bundeslaender, pd.DataFrame):
            return False
        
        return True
        
    
    def __init__(self, 
                 input_path = u'../data/NPGEO',
                 lut_path = u'../data/LUT', 
                 file_pattern = u'RKI_COVID19_{year:04d}-{month:02d}-{day:02d}.csv'):
        
        
        # get full path filename of this module
        script_full_filename = inspect.getframeinfo(inspect.currentframe()).filename
        # transform into absolute path (without filename)
        script_path = os.path.dirname(os.path.abspath(script_full_filename))
        # use default relative path to data and transform into absolute path
        self.__script_path = script_path
        

        if not (input_path is None) and isinstance(input_path, str) and len(input_path)>=1:
            # when relative paths => prefix with script path and get absolute
            # and store as member
            input_path = NpgeoReader.__check_and_normalize_path(input_path, script_path)
            self.__input_path = input_path
        else:
            self.__input_path = None
            
            
        if not (input_path is None) and isinstance(input_path, str) and len(input_path)>=1:
            # when relative paths => prefix with script path and get absolute
            # and store as member
            lut_path = NpgeoReader.__check_and_normalize_path(lut_path, script_path)        
            self.__lut_path = lut_path
        else:            
            self.__lut_path = None
            
            
        if not (file_pattern is None) and isinstance(file_pattern, str):
            assert NpgeoReader.__is_pattern_valid(file_pattern)
            self.__file_pattern = file_pattern
        else:
            self.__file_pattern = None


        if not self.__lut_path is None:
            self.load_LUT()
        else:
            self.__bundeslaender = None
            self.__kreise = None
            
        self.rkidata = {}
                        
        
        
    def get_datafile_list(self):
        extract_pattern = re.sub(r'\{(year|month|day)\:[0-9]+d\}', 
                                 r'(\\d*)', 
                                 self.__file_pattern)
        search_pattern = re.sub(r'\{[^}]*\}', 
                                '*', 
                                self.__file_pattern)
        param_order = [''.join(a) for a in 
                       re.findall(NpgeoReader.__order_pattern, 
                                  self.__file_pattern)]
        
        file_list = [os.path.basename(os.path.realpath(s))
                for s in glob.glob(self.input_path + '\\' + search_pattern)]
                
        extract_list = [re.findall(extract_pattern, f) for f in file_list]
        
        valid_list = [i for i, x in enumerate(
                list(map(lambda x: len(x)>0, extract_list))
            ) if x]
        
        file_list = [file_list[i] for i in valid_list]
        
        extract_list = [extract_list[i][0] for i in valid_list]
        
        idz = [
                param_order.index('year'),
                param_order.index('month'), 
                param_order.index('day')
            ]
        normalized_dates = ['{:04d}-{:02d}-{:02d}'.format(
                        int(t[idz[0]]), 
                        int(t[idz[1]]), 
                        int(t[idz[2]])
                    ) for t in extract_list]
        
        return {normalized_dates[i]: file_list[i] 
                for i in range(len(extract_list))}
        
    
    def __get_latest_data_file(self):
        all_data_files = self.get_datafile_list()
        dates = list(all_data_files.keys())
        dates.sort()
        return dates[-1], all_data_files[dates[-1]]
    
                         
    def load(self, data_version_date = None):
        if isinstance(data_version_date, pd.Timestamp):
            data_version_date = data_version_date.strftime('%Y-%m-%d')
            
        dv_arr = [int(a) for a in data_version_date.split('-')]
        fname = self.__file_pattern.format(
            year = dv_arr[0],
            month = dv_arr[1],
            day = dv_arr[2])
        if not isinstance(self.__input_path, str):
            raise ValueError('input_path is not set')
            
        if not os.path.isdir(self.__input_path):
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT), 
                                    fname)
        elif not os.path.isfile(self.__input_path + '/' + fname):
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT), 
                                    self.__input_path + '/' + fname)
            
        if data_version_date is None:
            data_version_date, data_filename = self.__get_latest_data_file()
        else:
            all_data_files = self.get_datafile_list()
            data_filename = all_data_files[data_version_date]
            
        # Try default style of data (after 07.04.2020)
        data = None
        try:
            data = pd.read_csv(
                self.__input_path + '/' + data_filename, 
                parse_dates=['Meldedatum', 'Refdatum'],
                sep=',',
                decimal = '.', thousands = None, comment = '#')
        except ValueError:
            pass
        
        if data is None:                    
            data = pd.read_csv(
                self.__input_path + '/' + data_filename, 
                parse_dates=['Meldedatum'],
                sep=',',
                decimal = '.', thousands = None, comment = '#')
            
            data = data.assign(Refdatum = data.Meldedatum.copy())
        
        # verify _
        data_datenstand = '-'.join(data.Datenstand[0][:10].split('.')[-1::-1])
        data_datenstand2 = '-'.join(data.Datenstand[0][:10].split('/'))
        if (data_version_date != data_datenstand) and (data_version_date != data_datenstand2):
            raise ValueError('Datenstand in file (' + data_datenstand + 
                             ') doesn''t match filename (' +  data_version_date + ')!')
            
        # ihucos timesstamps are utc milliseconds since 1970 check and convert
        if isinstance(data.Meldedatum[0], str):
            data.loc[:, 'Meldedatum'] = data.loc[:, 'Meldedatum'].apply(
                lambda s: datetime.datetime.utcfromtimestamp(float(s[:-3])))
        
        if isinstance(data.Refdatum[0], str):
            data.loc[:, 'Refdatum'] = data.loc[:, 'Refdatum'].apply(
                lambda s: datetime.datetime.utcfromtimestamp(float(s[:-3])))
            
        # Remove "bad" bundesland / Landkreis entries
        # like "LK Göttingen (alt)"
        data = data.loc[data.Bundesland.isin(self.__bundeslaender.index)]
        data = data.loc[data.Landkreis.isin(self.__kreise.index)]
        
        # Consistency check of ids before removing them
        idlist1 = self.__bundeslaender.loc[data.Bundesland, 'Id'].to_numpy()
        idlist2 = data.IdBundesland.to_numpy()
        bundeslaender_ids_ok = all(idlist1 == idlist2)
        try:
            assert bundeslaender_ids_ok
        except AssertionError:
            print('Warning: {:d} Bundesländer ID Error'.format((idlist1 != idlist2).sum()))
            pass
        
        idlist1 = self.__kreise.loc[data.Landkreis, 'Id'].to_numpy()
        idlist2 = data.IdLandkreis.to_numpy()
        kreise_ids_ok = all(idlist1 == idlist2)
        try:
            assert kreise_ids_ok
        except AssertionError:
            print('Warning: {:d} Kreise IDError(s)'.format((idlist1 != idlist2).sum()))
            pass
            
        dropcols = pd.Index(['FID', 'ObjectId', 'Datenstand', 
                    'IdBundesland', 'IdLandkreis'])
        
        data.drop(columns = dropcols.intersection(data.columns), inplace=True)
        
        data['Meldedatum'] = data['Meldedatum'].astype('datetime64[ns]')
        data['Refdatum'] = data['Refdatum'].astype('datetime64[ns]')
                
        self.rkidata[data_version_date] = data
        return data
    
    def index_to_data_version_date(self, idx):
        all_data_files = self.get_datafile_list() 
        
        if isinstance(idx, int):              
            return list(all_data_files.keys())[idx]
        elif isinstance(idx, str):
            if idx in all_data_files.keys():
                return idx
        elif isinstance(idx, pd.Timestamp):
            idx = idx.strftime('%Y-%m-%d')
        raise ValueError('Index "' + str(idx) + '" not found')
        
        
    def __getitem__(self, key):
        if isinstance(key, int):
            all_data_files = self.get_datafile_list()   
            key = list(all_data_files.keys())[key]
        
        if isinstance(key, pd.Timestamp):
            key = key.strftime('%Y-%m-%d')
                
        if not key in self.rkidata:
            # try loading
            self.load(key)
        return self.rkidata[key]
        
    
    @property 
    def bundeslaender(self):
        return self.__bundeslaender.index.copy()
    
    @property 
    def altersgruppen(self):
        kser = self.__kreise.columns.to_series()
        result = kser.filter(regex=r'^A\d{2}([+]|\-A\d{2})$').index.copy()
        return result.append(pd.Index(['A00+']))
    