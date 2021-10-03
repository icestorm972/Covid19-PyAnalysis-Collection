# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 22:43:48 2021

@author: David
"""
import sys
import os
import os.path
import tempfile
import requests
import gzip
import shutil
import pandas as pd
import numpy as np

import urllib.request, json 

# Download URL for arcgis NPGEO Hub RKI Data:
#ARCGIS_RKI_URL = 'https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv'
ARCGIS_RKI_URL = 'https://opendata.arcgis.com/api/v3/datasets/dd4580c810204019a7b8eb3e0b329dd6_0/downloads/data?format=csv&spatialRefId=4326'

# Alternative Download URL
ARCGIS_RKI_URL2 = 'https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data'

# JSON URL for number of entries on arcgis
ARCGIS_JSON_URL1 = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=*&returnCountOnly=true&outSR=4326&f=json'

# JSON URL for summary data on arcgis
ARCGIS_JSON_URL2 = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/rki_key_data_hubv/FeatureServer/0/query?where=(AdmUnitId%3D0)&outFields=*&outSR=4326&f=json'

# Output folder for downloaded data file
DOWNLOAD_OUTPUT_PATH = 'D:\\COVID-19\\data\\NPGEO'

# In case with primary download is something wrong, secondary/mirror can be forced
FORCE_MIRROR = False


# High level download function
# * tries primary URL (faster/smaller due gzip)
# * checks for correct date (either parameter or todays date)
# * if that fails, try to use secondary URL (slow, US AWS server, raw csv)
#
# parameter: 
#   output_path:   str. Default = DOWNLOAD_OUTPUT_PATH
#                  output folder path to save csv files
#   expected_date: str. Default = None.
#                  if None => use today's date for check
#                  otherwise 'YYYY-MM-DD' string for check
#   ignore_first:  bool. Default = False.
#                  if True try directly ARCGIS_RKI_URL2
# returns tuple
#   success: boo.  True = csv was downloaded
#   filename: str. Filename incl. path to csv
def download_RKI_COVID19_csv(output_path = DOWNLOAD_OUTPUT_PATH, 
                             expected_date = None,
                             ignore_first = False
                             ):
    if expected_date is None:
        expected_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # check output path
    op_valid = False
    try:
        if os.path.isdir(output_path):
            op_valid = True
    except:
        pass
    
    if not op_valid:
        print('\nERROR: output path {:s} doesn''t exist!'.format(
                output_path))
        return False, ''
    output_path = os.path.abspath(output_path)
    print('Download folder: {:s}'. format(output_path))

    if not ignore_first:
        print('Trying 1st URL:')
        res, fname = download_arcgis_csv(output_path, ARCGIS_RKI_URL, expected_date)
    else:
        print('Ignoring 1st URL:')
        res = False
        
    if res == False:
        print('Trying 2nd URL:')
        res, fname = download_arcgis_csv(output_path, ARCGIS_RKI_URL2, expected_date)

    return res, fname

# Quellen:
# 1) https://stackoverflow.com/questions/25749345/how-to-download-gz-files-with-requests-in-python-without-decoding-it
# 2) https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_arcgis_csv(output_path = DOWNLOAD_OUTPUT_PATH, url = ARCGIS_RKI_URL, date_filter = None):
    prgc = ['|', '/', '-', '\\']
    
    # start request of download. 
    # Use stream for asynchronous transfer in chunks for progrss bar
    response = requests.get(url, stream=True)
    
    # Get total length of gzip data
    total_length = response.headers.get('content-length')
    
    if response.status_code == 404:
         print('\n  ERROR: Data not available for download (404 error)!\n')
         return False, ''
    
    # Get date/time of file and test if correct
    h_lastmod = response.headers['Last-Modified']
    file_timestamp = pd.to_datetime(h_lastmod)
    file_tstr = file_timestamp.strftime('%Y-%m-%d')
    if not date_filter is None:
        if date_filter != file_tstr:
            print('\n  WARNING: Last Modified {:s} => {:s}, but expected {:s}\n'.format(
                file_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'), 
                file_tstr,
                date_filter))
            return False, ''
    res = True
    
    # Extract filename. This should be RKI_COVID19.csv and is also the fallback
    cdisp = response.headers['Content-Disposition'].split(';')[1].strip().split('=')
    if len(cdisp)>=2 and cdisp[0] == 'filename':
        file_name = cdisp[1].strip('"''')
    else:
        file_name = 'RKI_COVID19.csv'
    print('  Downloading {:s} (Last Updated: {:s})'. format(
        file_name, file_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')))
    
    
    # split file name in basename and extension
    fname_s = file_name.split('.')
    
    if not date_filter is None:
        file_name2 = os.path.abspath('{:s}/{:s}_{:s}.{:s}'.format(
                        output_path,
                        fname_s[0],
                        date_filter,
                        fname_s[1]))
        if os.path.isfile(file_name2):
            base_filename2 = os.path.basename(file_name2)
            print('\n  WARNING: {:s} already exists!\n           Renaming old file to'.format(
                base_filename2))
            
            if not os.path.isfile(file_name2 + '.bak'):
                os.rename(file_name2, file_name2 + '.bak')
                print('           ' + base_filename2 + '.bak\n')
            else:
                cnt = 0
                while os.path.isfile(file_name2 + '.{:03d}.bak'.format(
                        cnt)):
                    cnt += 1
                os.rename(file_name2, file_name2 + '.{:03d}.bak'.format(
                        cnt))
                print('           ' + base_filename2 + '.{:03d}.bak\n'.format(
                        cnt))
    

    # check if raw csv or gzip
    is_gzip = False
    tmp_ext = 'csv'
    if 'Content-Encoding' in response.headers and response.headers['Content-Encoding'] == 'gzip':
        is_gzip = True
        tmp_ext = 'gz'

    # setup download of gzip file in temporary directory
    download_fstr = '{:s}/{:s}_download.{:s}'.format(
        tempfile.gettempdir(), file_name, tmp_ext)
    
    # start download stream
    with open(download_fstr, "wb") as f:    
        if total_length is None: 
            # no content length header => just write the content
            f.write(response.content)
        else:
            # dl => downloaded bytes
            dl = 0
            total_length = int(total_length)
            if is_gzip:
                print('  Compressed download size: {:21,d} bytes'.format(total_length))
            else:
                print('  Uncompressed download/file size: {:14,d} bytes'.format(total_length))
    
            # 256 kile byte per progress bar update
            chunk_size = 256*1024
            
            # Download chunks of max. chunk_size bytes, and don't decode gzip
            # so the content-length is correct
            ccnt = 0
            for chunk in response.raw.stream(chunk_size, decode_content=False):
                if chunk:
                    # write data to disk
                    f.write(chunk)
                    
                    # update progress bar (length 50 characters)
                    dl += len(chunk)                     
                    done = int(51 * dl / total_length)
                    if done < 51:
                        sys.stdout.write("\r  [%s%s%s]" % (
                                '=' * done,
                                prgc[ccnt % 4],
                                ' ' * (51-1-done))
                            )
                    else:
                        sys.stdout.write("\r  [%s]" % ('=' * done) )    
                    sys.stdout.flush()   
                    
                    ccnt += 1
        
            sys.stdout.write("\n")    
            sys.stdout.flush()   
    
    # create dummy filename. The correct date is applied later => rename
    file_name1 = os.path.abspath('{:s}/{:s}_{:s}.{:s}'.format(
                    output_path,
                    fname_s[0],
                    'YYYY_MM_DD',
                    fname_s[1]))

    if is_gzip:
        # unzip downloaded file
        with open(file_name1, 'wb') as f_out:
            with gzip.open(download_fstr, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
        
        print('  Uncompressed file size: {:23,d} bytes'.format(
            os.path.getsize(file_name1)
            ))
    else:
        # just copy csv
        with open(file_name1, 'wb') as f_out:
            with open(download_fstr, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
        
    # sanity checks for 'Datenstand' and extract date for new file name
    csv_data = pd.read_csv(file_name1)
    if '.' in csv_data['Datenstand'].iloc[0][:10]:
        csv_data['Datenstand'] = pd.to_datetime(csv_data.Datenstand.apply(lambda s: '-'.join(s[:10].split('.')[-1::-1])))
    elif '/' in csv_data['Datenstand'].iloc[0][:10]:
        csv_data['Datenstand'] = pd.to_datetime(csv_data.Datenstand.apply(lambda s: '-'.join(s[:10].split('/'))))
    elif '-' in csv_data['Datenstand'].iloc[0][:10]:
        csv_data['Datenstand'] = pd.to_datetime(csv_data.Datenstand.apply(lambda s: s[:10]))
    else:
        print('\n  WARNING: Unrecognized "Datenstand" column format! Example "{:s}"'.format(csv_data['Datenstand'].iloc[0]))
        return False, ''
    
    csv_revision_dates = [np.datetime_as_string(d, unit='D') for d in csv_data.Datenstand.unique()]
    if len(csv_revision_dates) != 1:
        print('\n  WARNING: Datenstand column contains more than one date!')
        for i in range(len(csv_revision_dates)):
            print('    {:d}) {:s}'.format(i+1, csv_revision_dates[i]))
        return False, ''
    else:
        new_base_name = csv_revision_dates[0]
    
    # sanity check over number of lines/entries
    try:
        csv_num_lines = csv_data.shape[0]
        
        with urllib.request.urlopen(ARCGIS_JSON_URL1) as url:
            data = json.loads(url.read().decode())
            arcgis_num_lines = data['count']
        
        if arcgis_num_lines == csv_data.shape[0]:
            print('Checking number of lines: {:d}, seems ok! (at least by arcgis json API result)'.format(
                arcgis_num_lines))
        else:
            print('Checking number of lines: {:d}, but arcgis json API result was {:d} entries!?'.format(
                csv_num_lines, arcgis_num_lines))
            
    except:
        print('Couldn''t check number of lines due to exception!')
        

    # Print some summary infos to console and also try to check with arcgis
    csv_summary = {
        'AnzFall': csv_data.loc[csv_data.NeuerFall>=0,'AnzahlFall'].sum(),
        'AnzFallNeu': csv_data.loc[csv_data.NeuerFall!=0,'AnzahlFall'].sum(),
        'AnzTodesfall': csv_data.loc[csv_data.NeuerTodesfall>=0,'AnzahlTodesfall'].sum(),
        'AnzTodesfallNeu': csv_data.loc[(csv_data.NeuerTodesfall==-1)|(csv_data.NeuerTodesfall==1),'AnzahlTodesfall'].sum()
    }
    
    
    try:
        csv_num_lines = csv_data.shape[0]
        
        with urllib.request.urlopen(ARCGIS_JSON_URL2) as url:
            data = json.loads(url.read().decode())
            arcgis_summary = data['features'][0]['attributes']
            
        print('Summary:')
        print('                {:>10s} | {:>10s} | {:^5}'.format('CSV', 'ARCGIS', 'Check'))
        print('  Total cases:  {:10d} | {:10d} | {:^5}'.format(
            csv_summary['AnzFall'], 
            arcgis_summary['AnzFall'],
            'OK' if csv_summary['AnzFall'] == arcgis_summary['AnzFall'] else 'ERROR'
            ))
        print('  New   cases:  {:10d} | {:10d} | {:^5}'.format(
            csv_summary['AnzFallNeu'],
            arcgis_summary['AnzFallNeu'],
            'OK' if csv_summary['AnzFallNeu'] == arcgis_summary['AnzFallNeu'] else 'ERROR'
            ))
        print('  Total deaths: {:10d} | {:10d} | {:^5}'.format(
            csv_summary['AnzTodesfall'],
            arcgis_summary['AnzTodesfall'],
            'OK' if csv_summary['AnzTodesfall'] == arcgis_summary['AnzTodesfall'] else 'ERROR'
            ))
        print('  New   deaths: {:10d} | {:10d} | {:^5}'.format(
            csv_summary['AnzTodesfallNeu'],
            arcgis_summary['AnzTodesfallNeu'],
            'OK' if csv_summary['AnzTodesfallNeu'] == arcgis_summary['AnzTodesfallNeu'] else 'ERROR'
            ))
        
    except:
        print('Couldn''t check summary due to exception!')
            
        print('CSV-Summary:')
        print('  Total cases:  {:10d}'.format(csv_summary['AnzFall']))
        print('  New   cases:  {:10d}'.format(csv_summary['AnzFallNeu']))
        print('  Total deaths: {:10d}'.format(csv_summary['AnzTodesfall']))
        print('  New   deaths: {:10d}'.format(csv_summary['AnzTodesfallNeu']))

    # # extract "Datenstand" from first line of csv data for new file name
    # csv_datestr = pd.read_csv(file_name1, nrows=1)['Datenstand'][0][:10]
    # if '/' in csv_datestr:
    #     # some had yyyy/mm/dd format
    #     new_base_name = '-'.join(csv_datestr.split('/'))        
    # else:
    #     # normal dd.mm.yyyy format
    #     new_base_name = '-'.join(csv_datestr.split('.')[-1::-1])
    
    # sanity check / assert: compare with file_tstr
    if file_tstr != new_base_name:
        print('\n  WARNING: File time stamp {:s} and Datenstand {:s} are not the same!\n'.format(
            file_tstr, new_base_name))
        res = False
    
    # create final filename
    file_name2 = os.path.abspath('{:s}/{:s}_{:s}.{:s}'.format(
            output_path,
            fname_s[0],
            new_base_name,
            fname_s[1]))
    
    # rename file to final filename
    os.rename(file_name1, file_name2)    
    return res, file_name2

if __name__ == "__main__":
    download_RKI_COVID19_csv(
        output_path = DOWNLOAD_OUTPUT_PATH, 
        expected_date = None,
        ignore_first = FORCE_MIRROR)