# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 22:43:48 2021

@author: David
"""
import sys
import os
import tempfile
import requests
import gzip
import shutil
import pandas as pd

ARCGIS_RKI_URL = 'https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv'
DOWNLOAD_OUTPUT_PATH = '.'

# Quellen:
# 1) https://stackoverflow.com/questions/25749345/how-to-download-gz-files-with-requests-in-python-without-decoding-it
# 2) https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_arcgis_csv(file_name = None, url = ARCGIS_RKI_URL):
    response = requests.get(url, stream=True)
    total_length = response.headers.get('content-length')
    
    cdisp = response.headers['Content-Disposition'].split(';')[1].strip().split('=')
    if file_name is None and len(cdisp)>=2 and cdisp[0] == 'filename':
        file_name = cdisp[1].strip('"''')
    
    print("Downloading %s" % file_name)
    
    download_fstr = tempfile.gettempdir() + '\\ARCGIS_RKI_download.gz'
    
    with open(download_fstr, "wb") as f:    
        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            print('Total length: {:d} bytes'.format(total_length))
            for chunk in response.raw.stream(1024, decode_content=False):
                if chunk:
                    f.write(chunk)
                    
                    dl += len(chunk)                     
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                    sys.stdout.flush()

    with open(DOWNLOAD_OUTPUT_PATH + '\\' + file_name, 'wb') as f_out:
        with gzip.open(download_fstr, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)
            
    csv_datestr = pd.read_csv('RKI_COVID19.csv', nrows=1)['Datenstand'][0][:10]
    if '/' in csv_datestr:
        new_base_name = '-'.join(csv_datestr.split('/'))
        
    else:
        new_base_name = '-'.join(csv_datestr.split('.')[-1::-1])
        
    fname_s = file_name.split('.')
    file_name2 = '{:s}_{:s}.{:s}'.format(
            fname_s[0],
            new_base_name,
            fname_s[1])
    
    os.rename(DOWNLOAD_OUTPUT_PATH + '\\' + file_name, 
              DOWNLOAD_OUTPUT_PATH + '\\' + file_name2)    
    