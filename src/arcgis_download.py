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

# Download URL for arcgis NPGEO Hub RKI Data:
ARCGIS_RKI_URL = 'https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv'

# Output folder for downloaded data file
DOWNLOAD_OUTPUT_PATH = '.'


# Quellen:
# 1) https://stackoverflow.com/questions/25749345/how-to-download-gz-files-with-requests-in-python-without-decoding-it
# 2) https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_arcgis_csv(url = ARCGIS_RKI_URL):
    # start request of download. 
    # Use stream for asynchronous transfer in chunks for progrss bar
    response = requests.get(url, stream=True)
    # Get total length of gzip data
    total_length = response.headers.get('content-length')
    
    # Extract filename. This should be RKI_COVID19.csv and is also the fallback
    cdisp = response.headers['Content-Disposition'].split(';')[1].strip().split('=')
    if len(cdisp)>=2 and cdisp[0] == 'filename':
        file_name = cdisp[1].strip('"''')
    else:
        file_name = 'RKI_COVID19.csv'
    print("Downloading %s" % file_name)

    # setup download of gzip file in temporary directory    
    download_fstr = tempfile.gettempdir() + '\\' + file_name + '_download.gz'
    
    # start download stream
    with open(download_fstr, "wb") as f:    
        if total_length is None: 
            # no content length header => just write the content
            f.write(response.content)
        else:
            # dl => downloaded bytes
            dl = 0
            total_length = int(total_length)
            print('Total length: {:d} bytes'.format(total_length))
            
            # Download chunks of max. 1024 bytes, and don't decode gzip
            # so the content-length is correct
            for chunk in response.raw.stream(1024, decode_content=False):
                if chunk:
                    # write data to disk
                    f.write(chunk)
                    
                    # update progress bar (length 50 characters)
                    dl += len(chunk)                     
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                    sys.stdout.flush()                    
    
    # split file name in basename and extension
    fname_s = file_name.split('.')
    
    # create dummy filename. The correct date is applied later => rename
    file_name1 = '{:s}\\{:s}_{:s}.{:s}'.format(
            DOWNLOAD_OUTPUT_PATH,
            fname_s[0],
            'YYYY_MM_DD',
            fname_s[1])

    # unzip downloaded file
    with open(file_name1, 'wb') as f_out:
        with gzip.open(download_fstr, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)
    
    # extract "Datenstand" from first line of csv data for new file name
    csv_datestr = pd.read_csv(file_name1, nrows=1)['Datenstand'][0][:10]
    if '/' in csv_datestr:
        # some had yyyy/mm/dd format
        new_base_name = '-'.join(csv_datestr.split('/'))        
    else:
        # normal dd.mm.yyyy format
        new_base_name = '-'.join(csv_datestr.split('.')[-1::-1])
      
    # create final filename
    file_name2 = '{:s}\\{:s}_{:s}.{:s}'.format(
            DOWNLOAD_OUTPUT_PATH,
            fname_s[0],
            new_base_name,
            fname_s[1])
    
    # rename file to final filename
    os.rename(file_name1, file_name2)    
    