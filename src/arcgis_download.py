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

# Download URL for arcgis NPGEO Hub RKI Data:
ARCGIS_RKI_URL = 'https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv'

# Alternative Download URL
ARCGIS_RKI_URL2 = 'https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data'

# Output folder for downloaded data file
DOWNLOAD_OUTPUT_PATH = '.'


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
# returns tuple
#   success: boo.  True = csv was downloaded
#   filename: str. Filename incl. path to csv
def download_RKI_COVID19_csv(output_path = DOWNLOAD_OUTPUT_PATH, expected_date = None):
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

    print('Trying 1st URL:')
    res, fname = download_arcgis_csv(output_path, ARCGIS_RKI_URL, expected_date)
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
        file_name2 = '{:s}/{:s}_{:s}.{:s}'.format(
                output_path,
                fname_s[0],
                date_filter,
                fname_s[1])
        if os.path.isfile(file_name2):
            print('\n  WARNING: {:s} already exists! Renaming old file'.format(
                file_name2))
            
            if not os.path.isfile(file_name2 + '.bak'):
                os.rename(file_name2, file_name2 + '.bak')
                print('           to ' + file_name2 + '.bak\n')
            else:
                cnt = 0
                while os.path.isfile(file_name2 + '.{:03d}.bak'.format(
                        cnt)):
                    cnt += 1
                os.rename(file_name2, file_name2 + '.{:03d}.bak'.format(
                        cnt))
                print('           to ' + file_name2 + '.{:03d}.bak\n'.format(
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
            print('  Total length: {:d} bytes'.format(total_length))
    
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
                    done = int(50 * dl / total_length)
                    if done < 50:
                        sys.stdout.write("\r[%s%s%s]" % (
                                '=' * done,
                                prgc[ccnt % 4],
                                ' ' * (49-done))
                            )
                    else:
                        sys.stdout.write("\r[%s]" % ('=' * done) )    
                    sys.stdout.flush()   
                    
                    ccnt += 1
    
    
    # create dummy filename. The correct date is applied later => rename
    file_name1 = '{:s}/{:s}_{:s}.{:s}'.format(
            output_path,
            fname_s[0],
            'YYYY_MM_DD',
            fname_s[1])

    if is_gzip:
        # unzip downloaded file
        with open(file_name1, 'wb') as f_out:
            with gzip.open(download_fstr, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
    else:
        # just copy csv
        with open(file_name1, 'wb') as f_out:
            with open(download_fstr, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
        
    
    # extract "Datenstand" from first line of csv data for new file name
    csv_datestr = pd.read_csv(file_name1, nrows=1)['Datenstand'][0][:10]
    if '/' in csv_datestr:
        # some had yyyy/mm/dd format
        new_base_name = '-'.join(csv_datestr.split('/'))        
    else:
        # normal dd.mm.yyyy format
        new_base_name = '-'.join(csv_datestr.split('.')[-1::-1])
    
    # sanity check / assert: compare with file_tstr
    if file_tstr != new_base_name:
        print('\n  WARNING: File time stamp {:s} and Datenstand {:s} are not the same!\n'.format(
            file_tstr, new_base_name))
        res = False
    
    # create final filename
    file_name2 = '{:s}/{:s}_{:s}.{:s}'.format(
            output_path,
            fname_s[0],
            new_base_name,
            fname_s[1])
    
    # rename file to final filename
    os.rename(file_name1, file_name2)    
    return res, file_name2

if __name__ == "__main__":
    download_RKI_COVID19_csv(
        output_path = DOWNLOAD_OUTPUT_PATH, 
        expected_date = None)