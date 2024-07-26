#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:42:54 2024

@author: kjmayer
"""

import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
from scipy.stats import binom
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as c
from cartopy import config
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([])
#%%
season = 'NDJFM'
# season = 'MJJAS'
DIR_FIG = '/Volumes/Elements_PhD/ARISE/global_analysis/plots/finalfigures/'

if season == 'NDJFM':
    DIR_LOAD = '/Volumes/Elements_PhD/ARISE/global_analysis/SST_confcorr/NDJFM/'
    locations = ['ssa','amazon','africa','russia']
elif season == 'MJJAS': 
    DIR_LOAD = '/Volumes/Elements_PhD/ARISE/global_analysis/SST_confcorr/MJJAS/'
    locations = ['amazon','africa','australia']

DIR = '/Volumes/Elements_PhD/ARISE/data/processed/ensmean1-10_detrend/'
xFINAME = 'control_ens1_SST_2015-2069_detrended_ensmean1-10_2.5x2.5.nc'
X = xr.open_dataarray(DIR+xFINAME)[0]
X = X.where((X.lat <= 70) & (X.lat >= -70),drop=True)
X = xr.where(np.isnan(X),x=X,y=1)
# ------
RUN = 'SAI'

ilatlon = 0

FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SAI_negssa = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SAI_posssa = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 1

FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SAI_negamazon = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SAI_posamazon = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 2
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SAI_negafrica = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SAI_posafrica = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 3
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SAI_negother = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SAI_posother = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

# ------
RUN = 'control'
ilatlon = 0
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SSP_negssa = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SSP_posssa = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 1
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SSP_negamazon = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SSP_posamazon = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 2
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SSP_negafrica = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SSP_posafrica = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 3
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SSP_negother = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
SSP_posother = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

# ------
RUN = 'baseline'

ilatlon = 0
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
base_negssa = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
base_posssa = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 1
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
base_negamazon = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
base_posamazon = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 2
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
base_negafrica = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
base_posafrica = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X

ilatlon = 3
FI_LOAD = 'SST_'+RUN+'_confcorrneg_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
base_negother = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X
FI_LOAD = 'SST_'+RUN+'_confcorrpos_'+locations[ilatlon]+'_'+season+'_60S-70Nx0-360E.nc'
base_posother = xr.open_dataarray(DIR_LOAD+FI_LOAD).mean(['mem','seed'],skipna=True)*X


#%%
# Plotting:
lat = base_negamazon.lat
lon = base_negamazon.lon
# -------------------------------------------------------------------------
# -------------------------- NDJFM ----------------------------------------
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(80,27))
ax = fig.subplot_mosaic('''
                        ADGJ
                        BEHK
                        CFIL
                        ''',subplot_kw={'projection':ccrs.PlateCarree(180)})


fig.subplots_adjust(wspace=0.075,hspace=0.075)
for loc in ['A','B','C','D','E','F','G','H','I','J','K','L']:
    ax[loc].coastlines(resolution='50m', color='dimgrey', linewidth=1)
    ax[loc].axis("off")
    ax[loc].set_ylim(-62.5,72.5)#72.5)

    gl = ax[loc].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

cmap = 'RdBu_r'
csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(-1, 1.1, .1),csm.N)

if season == 'NDJFM':
    ax['A'].text(x=-180,y=110,s='Boreal Winter',fontsize=75,color='dimgrey')
elif season == 'MJJAS':
    ax['A'].text(x=-180,y=110,s='Boreal Summer',fontsize=75,color='dimgrey')
                 
# ax['A'].text(x=-180,y=110,s='Amazon:',fontsize=75,color='dimgrey')


ax['A'].set_title('(a) BASE (+)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['A'].pcolor(lon,lat,base_posamazon,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['A'].plot(lon[114],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['B'].set_title('(b) SAI (+)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['B'].pcolor(lon,lat,SAI_posamazon,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['B'].plot(lon[114],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['C'].set_title('(c) SSP2-4.5 (+)' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['C'].pcolor(lon,lat,SSP_posamazon,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['C'].plot(lon[114],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
# ----

ax['D'].set_title('(d) BASE (-)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['D'].pcolor(lon,lat,base_negamazon,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['D'].plot(lon[114],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['E'].set_title('(e) SAI (-)' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['E'].pcolor(lon,lat,SAI_negamazon,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['E'].plot(lon[114],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['F'].set_title('(f) SSP2-4.5 (-)' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['F'].pcolor(lon,lat,SSP_negamazon,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['F'].plot(lon[114],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

## --------

# ax['G'].text(x=-180,y=110,s='Africa:',fontsize=75,color='dimgrey')

if season == 'NDJFM':
    ax['G'].set_title('(g) BASE (+)' ,fontsize=60,color='dimgrey',loc='left')
    c1 = ax['G'].pcolor(lon,lat,base_posafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['G'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
    ax['H'].set_title('(h) SAI (+)' ,fontsize=60,color='dimgrey',loc='left')
    c1 = ax['H'].pcolor(lon,lat,SAI_posafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['H'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
    ax['I'].set_title('(i) SSP2-4.5 (+)' ,fontsize=60,color='dimgrey',loc='left')
    c2 = ax['I'].pcolor(lon,lat,SSP_posafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['I'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    # ----
    
    ax['J'].set_title('(j) BASE (-)' ,fontsize=60,color='dimgrey',loc='left')
    c1 = ax['J'].pcolor(lon,lat,base_negafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['J'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
    ax['K'].set_title('(k) SAI (-)' ,fontsize=60,color='dimgrey',loc='left')
    c2 = ax['K'].pcolor(lon,lat,SAI_negafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['K'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
    ax['L'].set_title('(l) SSP2-4.5 (-)' ,fontsize=60,color='dimgrey',loc='left')
    c3 = ax['L'].pcolor(lon,lat,SSP_negafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['L'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
elif season == 'MJJAS':
    ax['G'].set_title('(g) BASE (+)' ,fontsize=60,color='dimgrey',loc='left')
    c1 = ax['G'].pcolor(lon,lat,base_posother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['G'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
    ax['H'].set_title('(h) SAI (+)' ,fontsize=60,color='dimgrey',loc='left')
    c1 = ax['H'].pcolor(lon,lat,SAI_posother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['H'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
    ax['I'].set_title('(i) SSP2-4.5 (+)' ,fontsize=60,color='dimgrey',loc='left')
    c2 = ax['I'].pcolor(lon,lat,SSP_posother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['I'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    # ----
    
    ax['J'].set_title('(j) BASE (-)' ,fontsize=60,color='dimgrey',loc='left')
    c1 = ax['J'].pcolor(lon,lat,base_negother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['J'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
    ax['K'].set_title('(k) SAI (-)' ,fontsize=60,color='dimgrey',loc='left')
    c2 = ax['K'].pcolor(lon,lat,SAI_negother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['K'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    
    ax['L'].set_title('(l) SSP2-4.5 (-)' ,fontsize=60,color='dimgrey',loc='left')
    c3 = ax['L'].pcolor(lon,lat,SSP_negother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
    ax['L'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

cax2 = plt.axes([.34,0.09,0.34,0.03])
cbar2 = plt.colorbar(c3,cax=cax2,orientation = 'horizontal',fraction=0.04, ticks=np.arange(-1,1.5,.5))
cbar2.ax.tick_params(size=0,labelsize=50)
cbar2.outline.set_visible(False)
cbar2.ax.set_xticklabels(np.round(np.arange(-1,1.5,.5),2),color='darkgrey')
cbar2.ax.set_xlabel('SST ($^{\circ}$C)',fontsize=60,color='darkgrey')

plt.show()
# if season == 'NDJFM':
#     plt.savefig(DIR_FIG+'SST_confcorr_NDJFM.png',bbox_inches='tight',dpi=300)
# elif season == 'MJJAS':
#     plt.savefig(DIR_FIG+'SST_confcorr_MJJAS.png',bbox_inches='tight',dpi=300)

#%%
# Plotting:
lat = base_negother.lat
lon = base_negother.lon
# -------------------------------------------------------------------------
# -------------------------- NDJFM ----------------------------------------
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(40,27))
ax = fig.subplot_mosaic('''
                        AD
                        BE
                        CF
                        ''',subplot_kw={'projection':ccrs.PlateCarree(180)})


fig.subplots_adjust(wspace=0.075,hspace=0.075)
for loc in ['A','B','C','D','E','F']:
    ax[loc].coastlines(resolution='50m', color='dimgrey', linewidth=1)
    ax[loc].axis("off")
    ax[loc].set_ylim(-62.5,72.5)#72.5)

    gl = ax[loc].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

cmap = 'RdBu_r'
csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(-1, 1.1, .1),csm.N)

if season == 'NDJFM':
    ax['A'].text(x=-180,y=110,s='Boreal Winter',fontsize=75,color='dimgrey')
    # ax['A'].text(x=-180,y=110,s='Russia:',fontsize=75,color='dimgrey')
    ax['A'].plot(lon[42],lat[50],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['B'].plot(lon[42],lat[50],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['C'].plot(lon[42],lat[50],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['D'].plot(lon[42],lat[50],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['E'].plot(lon[42],lat[50],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['F'].plot(lon[42],lat[50],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
elif season == 'MJJAS':
    ax['A'].text(x=-180,y=110,s='Boreal Summer',fontsize=75,color='dimgrey')
    # ax['A'].text(x=-180,y=110,s='Australia:',fontsize=75,color='dimgrey')
    ax['A'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['B'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['C'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['D'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['E'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    ax['F'].plot(lon[54],lat[20],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
                 

ax['A'].set_title('(a) BASE (+)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['A'].pcolor(lon,lat,base_posother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['B'].set_title('(b) SAI (+)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['B'].pcolor(lon,lat,SAI_posother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['C'].set_title('(c) SSP2-4.5 (+)' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['C'].pcolor(lon,lat,SSP_posother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)

# ----

ax['D'].set_title('(d) BASE (-)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['D'].pcolor(lon,lat,base_negother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['E'].set_title('(e) SAI (-)' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['E'].pcolor(lon,lat,SAI_negother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['F'].set_title('(f) SSP2-4.5 (-)' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['F'].pcolor(lon,lat,SSP_negother,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


cax2 = plt.axes([.34,0.09,0.34,0.03])
cbar2 = plt.colorbar(c3,cax=cax2,orientation = 'horizontal',fraction=0.04, ticks=np.arange(-1,1.5,.5))
cbar2.ax.tick_params(size=0,labelsize=50)
cbar2.outline.set_visible(False)
cbar2.ax.set_xticklabels(np.round(np.arange(-1,1.5,.5),2),color='darkgrey')
cbar2.ax.set_xlabel('SST ($^{\circ}$C)',fontsize=60,color='darkgrey')

plt.show()
# if season == 'NDJFM':
#     plt.savefig(DIR_FIG+'SST_confcorr_Russia_NDJFM.png',bbox_inches='tight',dpi=300)
# elif season == 'MJJAS':
#     plt.savefig(DIR_FIG+'SST_confcorr_Australia_MJJAS.png',bbox_inches='tight',dpi=300)
    
#%%
# Plotting only Western Africa:
lat = base_negamazon.lat
lon = base_negamazon.lon
# -------------------------------------------------------------------------
# -------------------------- NDJFM ----------------------------------------
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(65,40))
ax = fig.subplot_mosaic('''
                        ABC
                        .GH
                        DEF
                        .IJ
                        ''',subplot_kw={'projection':ccrs.PlateCarree(-60)})


fig.subplots_adjust(wspace=0.075,hspace=0.075)
for loc in ['A','B','C','D','E','F','G','H','I','J']:
    ax[loc].coastlines(resolution='50m', color='dimgrey', linewidth=1)
    ax[loc].axis("off")
    ax[loc].set_ylim(-62.5,72.5)

    gl = ax[loc].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

cmap = 'RdBu_r'
csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(-1, 1.1, .1),csm.N)

ax['A'].set_title('(a) BASE (+)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['A'].pcolor(lon,lat,base_posafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['A'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['B'].set_title('(b) SAI (+)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['B'].pcolor(lon,lat,SAI_posafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['B'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['C'].set_title('(c) SSP2-4.5 (+)' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['C'].pcolor(lon,lat,SSP_posafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['C'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
# ----

ax['D'].set_title('(d) BASE (-)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['D'].pcolor(lon,lat,base_negafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['D'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['E'].set_title('(e) SAI (-)' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['E'].pcolor(lon,lat,SAI_negafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['E'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['F'].set_title('(f) SSP2-4.5 (-)' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['F'].pcolor(lon,lat,SSP_negafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['F'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
# -----   

ax['G'].set_title('(g) SAI - BASE (+)' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['G'].pcolor(lon,lat,SAI_posafrica - base_posafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['G'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['H'].set_title('(h) SSP2-4.5 - BASE (+)' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['H'].pcolor(lon,lat,SSP_posafrica - base_posafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['H'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
# -----   

ax['I'].set_title('(i) SAI - BASE (-)' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['I'].pcolor(lon,lat,SAI_negafrica - base_negafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['I'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['J'].set_title('(j) SSP2-4.5 - BASE (-)' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['J'].pcolor(lon,lat,SSP_negafrica - base_negafrica,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['J'].plot(lon[6],lat[28],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
    

# cax2 = plt.axes([.34,0.09,0.34,0.03])
# cbar2 = plt.colorbar(c3,cax=cax2,orientation = 'horizontal',fraction=0.04, ticks=np.arange(-1,1.5,.5))
# cbar2.ax.tick_params(size=0,labelsize=50)
# cbar2.outline.set_visible(False)
# cbar2.ax.set_xticklabels(np.round(np.arange(-1,1.5,.5),2),color='darkgrey')
# cbar2.ax.set_xlabel('SST ($^{\circ}$C)',fontsize=60,color='darkgrey')

plt.show()
# plt.savefig(DIR_FIG+'SST_confcorr_NDJFM_wafrica.png',bbox_inches='tight',dpi=300)

#%%
# Plotting only southern south america:
lat = base_negamazon.lat
lon = base_negamazon.lon
# -------------------------------------------------------------------------
# -------------------------- NDJFM ----------------------------------------
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(65,20))
ax = fig.subplot_mosaic('''
                        ABC
                        DEF
                        ''',subplot_kw={'projection':ccrs.PlateCarree(-60)})


fig.subplots_adjust(wspace=0.075,hspace=0.075)
for loc in ['A','B','C','D','E','F']:
    ax[loc].coastlines(resolution='50m', color='dimgrey', linewidth=1)
    ax[loc].axis("off")
    ax[loc].set_ylim(-62.5,72.5)

    gl = ax[loc].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

cmap = 'RdBu_r'
csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(-1, 1.1, .1),csm.N)

ax['A'].set_title('(a) BASE (+)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['A'].pcolor(lon,lat,base_posssa,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['A'].plot(lon[116],lat[10],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['B'].set_title('(b) SAI (+)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['B'].pcolor(lon,lat,SAI_posssa,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['B'].plot(lon[116],lat[10],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['C'].set_title('(c) SSP2-4.5 (+)' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['C'].pcolor(lon,lat,SSP_posssa,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['C'].plot(lon[116],lat[10],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
# ----

ax['D'].set_title('(d) BASE (-)' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['D'].pcolor(lon,lat,base_negssa,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['D'].plot(lon[116],lat[10],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['E'].set_title('(e) SAI (-)' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['E'].pcolor(lon,lat,SAI_negssa,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['E'].plot(lon[116],lat[10],'X',markersize=20,color='k',transform=ccrs.PlateCarree())

ax['F'].set_title('(f) SSP2-4.5 (-)' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['F'].pcolor(lon,lat,SSP_negssa,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['F'].plot(lon[116],lat[10],'X',markersize=20,color='k',transform=ccrs.PlateCarree())
# -----   
    

cax2 = plt.axes([.34,0.09,0.34,0.03])
cbar2 = plt.colorbar(c3,cax=cax2,orientation = 'horizontal',fraction=0.04, ticks=np.arange(-1,1.5,.5))
cbar2.ax.tick_params(size=0,labelsize=50)
cbar2.outline.set_visible(False)
cbar2.ax.set_xticklabels(np.round(np.arange(-1,1.5,.5),2),color='darkgrey')
cbar2.ax.set_xlabel('SST ($^{\circ}$C)',fontsize=60,color='darkgrey')

# plt.show()
plt.savefig(DIR_FIG+'SST_confcorr_NDJFM_ssa.png',bbox_inches='tight',dpi=300)