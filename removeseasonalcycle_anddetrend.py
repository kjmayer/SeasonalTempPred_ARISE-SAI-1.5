#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:24:10 2022

@author: kmayer

Using the regrided SST data (2.5x2.5)


1) Calculate ensemble mean and
2) Simultaneously detrend & remove seasonal cycle

"""
import xarray as xr
import numpy as np
from numpy.polynomial import polynomial
import datetime as dt

import matplotlib.pyplot as plt

#%% Variables
RUN = 'control' # 'SAI', 'control'
VAR = 'T2m'#'SST' # 'T2m'
# if VAR == 'SST':
DIR = '/Volumes/Elements_External_HD/PhD/data/ARISE/raw/concat/regrid/'
# if VAR == 'T2m':
#     DIR = '/Volumes/Elements_External_HD/PhD/data/ARISE/raw/concat/'

#%% Calculate ensemble mean
'''
Combine ensemble members 1-10 (using all members for ensemble mean)
--> used for trend calculation
'''
mem = [1,2,3,4,5,6,7,8,9,10]

for im,m in enumerate(mem):
    print(m)
    if VAR == 'SST':
        FINAME = RUN+'_ens'+str(m)+'_'+VAR+'_2015-2069_2.5x2.5.nc'
        print(FINAME)
    elif VAR == 'T2m':
        FINAME = RUN+'_ens'+str(m)+'_'+VAR+'_2015-2069_5x5.nc'
    ens = xr.open_dataarray(DIR+FINAME)
    
    if im == 0:
        if VAR == 'SST':
            all_ens = xr.DataArray( np.zeros(shape=(10,660,73,144),dtype='float'),
                                    name=VAR,
                                    dims=('ens','time','lat','lon'),
                                    coords=[('ens',mem),
                                            ('time',ens.time),
                                            ('lat',ens.lat),('lon',ens.lon)])   
        elif VAR == 'T2m':
            all_ens = xr.DataArray( np.zeros(shape=(10,660,37,72),dtype='float'),
                                    name=VAR,
                                    dims=('ens','time','lat','lon'),
                                    coords=[('ens',mem),
                                            ('time',ens.time),
                                            ('lat',ens.lat),('lon',ens.lon)]) 
    
    all_ens[im] = ens

# Mean over ensemble members
ens_mean = all_ens.mean('ens',skipna=True)  

plt.plot(ens[:,28,55],'teal')
plt.plot(ens_mean[:,28,55],'k')
plt.show()

# Save ensemble member mean
if VAR == 'SST':
    ens_mean.to_netcdf('/Volumes/Elements_External_HD/PhD/data/ARISE/processed/ensmean/'+RUN+'_ensmean1-10_'+VAR+'_2015-2069_2.5x2.5.nc',
                        encoding={VAR: {'dtype': 'float32', '_FillValue': -900,'zlib': True, 'complevel': 1},
                                  'time':{'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1},
                                  'lat': {'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1},
                                  'lon': {'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1}}
                      )
elif VAR == 'T2m':
    ens_mean.to_netcdf('/Volumes/Elements_External_HD/PhD/data/ARISE/processed/ensmean/'+RUN+'_ensmean1-10_'+VAR+'_2015-2069_5x5.nc',
                        encoding={VAR: {'dtype': 'float32', '_FillValue': -900,'zlib': True, 'complevel': 1},
                                  'time':{'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1},
                                  'lat': {'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1},
                                  'lon': {'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1}}
                      )


#%% Detrend ensemble members individually
'''
FOR BOTH CONTROL & SAI:
1) Detrend 2015-2034 (before SAI)
2) Detrend 2035-2069 (after SAI)
3) Re-combine
'''

if VAR == 'SST':
    FINAME_ensmean = RUN+'_ensmean1-10_'+VAR+'_2015-2069_2.5x2.5.nc'
    ensmean = xr.open_dataarray('/Volumes/Elements_External_HD/PhD/data/ARISE/processed/ensmean/'+FINAME_ensmean)
    inan = np.isnan(ensmean)
    ensmean = xr.where(inan,x=0,y=ensmean)
elif VAR == 'T2m':
    FINAME_ensmean = RUN+'_ensmean1-10_'+VAR+'_2015-2069_5x5.nc'
    ensmean = xr.open_dataarray('/Volumes/Elements_External_HD/PhD/data/ARISE/processed/ensmean/'+FINAME_ensmean)
    inan = np.isnan(ensmean)
    ensmean = xr.where(inan,x=0,y=ensmean)

ensmean_before = ensmean.where(ensmean.time.dt.year <= 2034, drop=True)
ensmean_after  = ensmean.where(ensmean.time.dt.year >= 2035, drop=True)

ensmean_before = ensmean_before.stack(z=('lat','lon'))
ensmean_after  = ensmean_after.stack(z=('lat','lon'))

for m in np.arange(1,11):
    print(m)
    if VAR == 'SST':
        FINAME_ens = RUN+'_ens'+str(m)+'_'+VAR+'_2015-2069_2.5x2.5.nc'
        X = xr.open_dataarray(DIR+FINAME_ens)
        X = xr.where(inan,x=0,y=X)
        print(FINAME_ens)
    elif VAR == 'T2m':
        FINAME_ens = RUN+'_ens'+str(m)+'_'+VAR+'_2015-2069_5x5.nc'
        X = xr.open_dataarray(DIR+FINAME_ens)
    
    X_before = X.where(X.time.dt.year <= 2034, drop=True)
    X_after  = X.where(X.time.dt.year >= 2035, drop=True)
    
    
    X_before = X_before.stack(z=('lat', 'lon'))
    X_after  = X_after.stack(z=('lat', 'lon'))
    
    temp_before = X_before['time.month']
    temp_after  = X_after['time.month']
    
    ##########################################################################
    
    # detrend 2015-2034 (before SAI)
    detrend_before = []
    for label,ensmean_before_group in ensmean_before.groupby('time.month'):
        #print(label)
        Xgroup_before = X_before.where(temp_before == label, drop = True)
        
        curve_before = polynomial.polyfit(np.arange(0,ensmean_before_group.shape[0]),ensmean_before_group,3)
        trend_before = polynomial.polyval(np.arange(0,ensmean_before_group.shape[0]),curve_before,tensor=True)
        trend_before = np.swapaxes(trend_before,0,1)
        
        diff_before = Xgroup_before - trend_before
        detrend_before.append(diff_before)
        
        # plt.plot(ensmean_before_group[:,-1000],'grey')
        # plt.plot(Xgroup_before[:,-1000],'teal')
        # plt.plot(trend_before[:,-1000],'k')
        # plt.show()
    
    detrend_before_xr = xr.concat(detrend_before,dim='time').unstack()
    detrend_before_xr = detrend_before_xr.sortby('time')
    
    # detrend 2035-2069  (after SAI)  
    detrend_after = []
    for label,ensmean_after_group in ensmean_after.groupby('time.month'):
        #print(label)
        Xgroup_after = X_after.where(temp_after == label, drop = True)
        
        curve_after = polynomial.polyfit(np.arange(0,ensmean_after_group.shape[0]),ensmean_after_group,3)
        trend_after = polynomial.polyval(np.arange(0,ensmean_after_group.shape[0]),curve_after,tensor=True)
        trend_after = np.swapaxes(trend_after,0,1)
        
        diff_after = Xgroup_after - trend_after
        detrend_after.append(diff_after)
        
        # plt.plot(ensmean_after_group[:,-1000],'grey')
        # plt.plot(Xgroup_after[:,-1000],'teal')
        # plt.plot(trend_after[:,-1000],'k')
        # plt.show()
    
    detrend_after_xr = xr.concat(detrend_after,dim='time').unstack()
    detrend_after_xr = detrend_after_xr.sortby('time')
    
    
    ##########################################################################
    
    # combine the detrended before and after
    mem_detrended = xr.DataArray(np.zeros(np.shape(X),dtype='float'),
                                 name=VAR,
                                 dims=('time','lat','lon'),
                                 coords=[('time',X.time),('lat',X.lat),('lon',X.lon)])
    
    mem_detrended[:240] = detrend_before_xr
    mem_detrended[240:] = detrend_after_xr
    
    mem_detrended = xr.where(inan,x=np.nan,y=mem_detrended)
    mem_detrended[240].plot(levels=np.arange(-5,5.1,.1))
    plt.show()
    
    # save
    if VAR == 'SST':
        OUTFILE = RUN+'_ens'+str(m)+'_'+VAR+'_2015-2069_detrended_ensmean1-10_2.5x2.5.nc'
    elif VAR == 'T2m':
        OUTFILE = RUN+'_ens'+str(m)+'_'+VAR+'_2015-2069_detrended_ensmean1-10_5x5.nc'
        
    mem_detrended.to_netcdf('/Volumes/Elements_External_HD/PhD/data/ARISE/processed/ensmean1-10_detrend/'+OUTFILE,
                            encoding={VAR: {'dtype': 'float32', '_FillValue': -900,'zlib': True, 'complevel': 1},
                                      'time':{'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1},
                                      'lat': {'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1},
                                      'lon': {'dtype': 'double','_FillValue': -900,'zlib': True, 'complevel': 1}}
                            )
    
    ##########################################################################
    # plt.plot(mem_detrended[:100].mean(('lat','lon')))
    # plt.ylim(-1,1)
    # plt.show()
    
    # mem_detrended[101].plot(levels=np.arange(-5,5.1,.1))
    # plt.plot(X[:100].mean(('lat','lon')))
    # plt.ylim(4,6)
    # plt.show()