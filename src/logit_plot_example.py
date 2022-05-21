# -*- coding: utf-8 -*-
"""
Created on Sat May 21 16:37:50 2022

@author: David
"""

import os
import warnings
from datetime import datetime as dt

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import Locator, LogitLocator
from matplotlib.ticker import NullFormatter, PercentFormatter
from matplotlib.colors import PowerNorm
from matplotlib.dates import WeekdayLocator, DayLocator
from matplotlib.dates import ConciseDateFormatter, date2num
from matplotlib.patches import Rectangle

import seaborn as sns

from IPython.display import display, Image

import fig_util


# modified version of AutoMinorLocator of matplotlib.ticker
# adjusted for use with LogitLocator as major locator
class AutoLogitMinorLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks. The scale must be linear with major ticks evenly spaced.
    """
    def __init__(self):
        pass

    def __call__(self):
        """Return the locations of the ticks."""
        if self.axis.get_scale() != 'logit':
            warnings.warn('AutoLogitMinorLocator does not work with '
                                 'non-logit scale')
            return []
        #print("AutoLogitMinorLocator.__call__")

        majorlocs = self.axis.get_majorticklocs()
        #print("maj: "+str(majorlocs))
        try:
            majorstep = majorlocs[1] / majorlocs[0]
            if not np.isclose(majorstep, [10.0]).any():
                warnings.warn('AutoLogitMinorLocator for major locs '
                                     'other than 1:10 are not implemented')
                return []
                
        except IndexError:
            # Need at least two major ticks to find minor tick locations
            # TODO: Figure out a way to still be able to display minor
            # ticks without two major ticks visible. For now, just display
            # no ticks at all.
            return []


        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        
        locs = []
        for ma, mb in np.vstack((majorlocs[:-1], majorlocs[1:])).T:
            if mb <= 0.5:
                if np.isclose(mb/ma, [10.0]).any():
                    locs.append(ma * np.arange(2, 10, 1))
                elif np.isclose(mb/ma, [5.0]).any():
                    locs.append(ma * np.arange(2, 5, 1))
            else:
                mbb = 1-mb
                maa = 1-ma
                if np.isclose(maa/mbb, [10.0]).any():
                    locs.append(1 - mbb * np.arange(9, 1, -1))
                elif np.isclose(maa/mbb, [5.0]).any():
                    locs.append(1 - mbb * np.arange(4, 1, -1))
        locs = np.hstack((*locs,))
        #print(locs)

        return self.raise_if_exceeds(locs)

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))
        
# some constants for plot
LABEL_COL = (0.15, 0.15, 0.15)
GRID_COL = {
    'major': (0.85,0.85,0.85), 
    'minor': (0.95,0.95,0.95)
    }

FONT_SCALE = 3/1.5

PLT_YMIN = 0.001
PLT_YMAX = 0.999

PLT_XMIN = '2022-01-01'
PLT_XMAX = dt.now().strftime('%Y-%m-%d')

GRAYED_OUT_DAYS = 14
GRAYED_OUT_COLOR = (0.5, 0.2, 0)
GRAYED_OUT_OPACITY = 0.1
GRAYED_OUT_FONT_COLOR = 'r' # red
GRAYED_OUT_TEXT = "signifikante Anzahl\nNachmeldungen\nzu erwarten"

DATE_DOT_STR = dt.now().strftime('%d.%m.%Y')

# needed change to settings, so that gridlines are BEHIND the plot
plt.rc('axes', axisbelow=True)

############################################################################################
# extra constants
INCLUDE_XREASON = True
USE_FIG_UTIL = True
INPUT_FILEPATH = 'the_path_to_the_input_csv'
OUTPUT_FILEPATH = 'the_path_where_the_png_is_saved'

############################################################################################

# note: use LaTeX math string for supscript * => $^*$
if INCLUDE_XREASON:
    SEQ_REASON_STR = 'Labor unbekannt (X) oder kein spezieller angegeben (N) = (um Unbekannte) "erweiterte" Stichprobe'
    SEQ_REASON_TITLE = 'NX-Sequenzdaten$^*$'
else:
    SEQ_REASON_STR = 'kein spezieller angegeben (N) = reine repräsentative Stichprobe'
    SEQ_REASON_TITLE = 'N-Sequenzdaten$^*$'
    


# read data
plt_data = pd.read_csv(f'{INPUT_FILEPATH}{os.sep}PLT_DATA_NX.CSV')


# create figure & axis, adjust size of axis inside figure
fig, ax = plt.subplots(num=None, figsize=(6.75*2, 4*2), facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.3)

# Scatter plot, use data in form of pandas dataframe
# 'Proben-Datum', ... are the names of the data columns
# sizes: min/max size of dots of scatter plot
# size_norm: how to scale the dots
# ax: use already existing axis
sns.scatterplot(data = plt_data, 
                x = 'Proben-Datum', 
                y = 'Anteil (Logit-Skala)', 
                size = 'Proben-Anzahl', 
                hue = 'Sublinie/VoC',
                sizes = (2,200), 
                size_norm = PowerNorm(gamma=1.0), 
                ax = ax)

# set y axis to logistic scale
ax.set_yscale('logit')

# set x/y axis limits
ax.set_xlim([pd.to_datetime(PLT_XMIN), pd.to_datetime(PLT_XMAX)])
ax.set_ylim([PLT_YMIN, PLT_YMAX])


# set locator (where ticks/gridlines occur) and formatter (their labels)
# for y axis

# major: logistic positions for ticks
ax.yaxis.set_major_locator(LogitLocator())
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
# minor: no labels (NullFormater) but in between gridlines
ax.yaxis.set_minor_locator(AutoLogitMinorLocator())
ax.yaxis.set_minor_formatter(NullFormatter())

# for x axis
# WeekdayLocator: mark every week at second day (Wednesday)
# ConciseDateFormatter: minimalist/generalist formatter
xmaj_locator = WeekdayLocator(interval=1, byweekday=2)
xmaj_formatter = ConciseDateFormatter(xmaj_locator)
ax.xaxis.set_major_locator(xmaj_locator)
ax.xaxis.set_major_formatter(xmaj_formatter)
# gridline for every day
ax.xaxis.set_minor_locator(DayLocator(interval=1))

# no ticks marks for minor ticks
ax.tick_params(which='minor', length=0, width=0)
# set fontsize and color of major tick labels
ax.tick_params(labelsize=10*FONT_SCALE, labelcolor = LABEL_COL)

# activate and format gridline
ax.grid(True, which='major', axis='both', linestyle='-', color=GRID_COL['major'])
ax.grid(True, which='minor', axis='both', linestyle='-', color=GRID_COL['minor'])

# set and format axis labels
ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize': 12*FONT_SCALE}, 
              color = LABEL_COL, labelpad=18*FONT_SCALE)
ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': 12*FONT_SCALE}, 
              color = LABEL_COL, labelpad=14*FONT_SCALE)

# extra due to concise formatter
# this has year/month/... as extra text. To set fontsize we have to do this:
ax.xaxis.get_offset_text().set_size(10*FONT_SCALE)

# unused but useful for other plots:
# rotate x axis tick labels (e.g. for long dates)
# plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

# set and format plot title (using formatted strings with constants)
ax.set_title(f'COVID-19 - DE - Delta & Omikron Anteile in {SEQ_REASON_TITLE} - {PLT_XMAX}', 
             fontdict={'fontsize': 14*FONT_SCALE}, color = LABEL_COL, 
             pad=12*FONT_SCALE)


# draw legend
leg = plt.legend(loc = 'upper left', 
                ncol = 1,
                fontsize = 12*FONT_SCALE, 
                markerscale = 2,
                labelcolor = LABEL_COL)
# also set font color of legend title:
plt.setp(leg.get_title(), color=LABEL_COL)


# calculate area for grayed out data
endDate = pd.to_datetime(PLT_XMAX)
startDate = endDate - pd.DateOffset(days=GRAYED_OUT_DAYS)
rectStart = date2num(startDate)
rectEnd = date2num(endDate)
rectWidth = rectEnd - rectStart

# draw filled rectangle for grayed out data
ax.add_patch(Rectangle((rectStart, PLT_YMIN), rectWidth, PLT_YMAX-PLT_YMIN, 
                       fc=GRAYED_OUT_COLOR, alpha=GRAYED_OUT_OPACITY))
fig.text(0.89, 0.89, GRAYED_OUT_TEXT,
         color = GRAYED_OUT_FONT_COLOR,
         size = 8*FONT_SCALE, 
         va = "top", ha = "right") # vertical/horizontal alignment

# add footnotes
fig.text(0.15, 0.07, 
         "Datenquelle:\n" +
         "    Robert Koch-Institut. (2021). SARS-CoV-2 Sequenzdaten aus Deutschland, \n" +
         f"    Datenstand: {DATE_DOT_STR}, DOI: https://doi.org/10.5281/zenodo.5139363 ; "+
         "eigene Berechnung/Darstellung\n" +
         "Skript abgeleitet von Cornelius Römer https://github.com/corneliusroemer/\n" +
         f"$^*$ Sequenzierungsgrund ist {SEQ_REASON_STR}", 
         size=8*FONT_SCALE, va="bottom", ha="left")



if not USE_FIG_UTIL:
    plt.show()
else:
    reason_str = 'NX' if INCLUDE_XREASON else 'N'
    
    exp_full_fname = f'{OUTPUT_FILEPATH}{os.sep}Omicron_{PLT_XMAX}_{reason_str}.png'
    
    print('Saving ' + exp_full_fname)
    # adjust image size to get correct width/height in pixels and resolution
    fig_util.force_fig_size(fig, (1920.0/100.0, 1080.0/100.0), dpi=100, pad_inches=0.35)
    # save file
    fig.savefig(exp_full_fname, dpi=100, bbox_inches='tight', pad_inches=0.35)
    # load file and display it (in all frontend types)
    display(Image(filename=exp_full_fname))
    # close plot
    plt.close()