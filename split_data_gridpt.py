#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:23:09 2022

@author: kmayer
"""
import numpy as np
import xarray as xr
import datetime as dt
import warnings
warnings.simplefilter("ignore") 
#%%
def is_month(data, months):
    i_timedim = np.where(np.asarray(data.dims) == 'time')[0][0]
    if i_timedim == 0:
        data = data[data.time.dt.month.isin(months)]
    elif i_timedim == 1:
        data = data[:,data.time.dt.month.isin(months)]
    return data

#-----------------------------------------------------------------------------
def train(DIR, RUN, LEAD, TRAINmem, MEMstr, ilat, ilon, Xmonths = [8,9,10,11,12], Ymonths = [11,12,1,2,3]):
    
    for t, trainmem in enumerate(TRAINmem):
        #print('Loading X training member '+str(trainmem))
        xFINAME = RUN+'_ens'+str(trainmem)+'_SST_2015-2069_detrended_ensmean'+MEMstr+'_2.5x2.5.nc'
        X = xr.open_dataarray(DIR+xFINAME)
        X = X.where(X.time.dt.year >= 2050, drop=True)
        X = X[:-1*LEAD]
        X = is_month(X,Xmonths)
        X = X.where((X.lat <= 70) & (X.lat >= -70),drop=True)
        
        if t == 0:
            Xall = xr.DataArray(np.zeros(shape=(len(TRAINmem),np.shape(X)[0],np.shape(X)[1],np.shape(X)[2]),dtype='float')+np.nan,
                                    name='SST',
                                    dims=('ens','time','lat','lon'),
                                    coords=[('ens',TRAINmem),
                                            ('time',X.time.data),
                                            ('lat',X.lat.data),('lon',X.lon.data)])
            
        Xall[t] = X

    Xall_stacked  = Xall.stack(time_all = ('ens','time'))
    Xall_stackedT = Xall_stacked.transpose('time_all','lat','lon')

    Xtrain_std    = np.nanstd(Xall_stackedT,axis=0)
    Xtrain_mean   = np.nanmean(Xall_stackedT, axis=0)
    
    Xtrain  = (Xall_stackedT - Xtrain_mean) / Xtrain_std
    Xtrain  = Xtrain.stack(z=('lat','lon'))
    
    ##########################################################################
    
    for t, trainmem in enumerate(TRAINmem):
        #print('Loading Y training member '+str(trainmem))

        yFINAME = RUN+'_ens'+str(trainmem)+'_T2m_2015-2069_detrended_ensmean'+MEMstr+'_5x5.nc'
        Y = xr.open_dataarray(DIR+yFINAME) 
        Y = Y.where(Y.time.dt.year >= 2050, drop=True)
        Y = Y[LEAD:,ilat,ilon]
        Y = is_month(Y,Ymonths)

        if t == 0:
            Yall = xr.DataArray(np.zeros(shape=(len(TRAINmem),np.shape(Y)[0]),dtype='float')+np.nan,
                                    name='T2m',
                                    dims=('ens','time'),
                                    coords=[('ens',TRAINmem),
                                            ('time',Y.time.data)])
        
    
        Yall[t] = Y

    Yall_stacked = Yall.stack(time_all=('ens','time'))
    
    Ytrain_median = np.median(Yall_stacked)
    Ytrain = Yall_stacked - Ytrain_median
    Ytrain[np.where(Ytrain>=0)[0]] = 1
    Ytrain[np.where(Ytrain<0)[0]] = 0
    
    return Xtrain, Ytrain, Xtrain_mean, Xtrain_std, Ytrain_median
 

#-----------------------------------------------------------------------------
def val(DIR, RUN, LEAD, VALmem, MEMstr, Xtrain_mean, Xtrain_std, Ytrain_median, ilat, ilon, Xmonths = [8,9,10,11,12], Ymonths = [11,12,1,2,3]):
    
    '''
    Load X (SST) & Y (T2M) data from base run (2015-2034 = no SAI or SSP2-4.5) and return
    validation data.
    
    Variables
    ----------
    LEAD: int
        the montly lead time between SST and T2M (e.g. LEAD = 2 --> Jan SST predicts March T2M)
    DIR: str
        directory path where SST and T2M data is located
    VALmem: list
        The member(s) used for validation (e.g. 3 = member 3)
        --> this/these are the member(s) loaded in this fuction
    MEMstr: str
        The members used for detrending and removing the seasonal cycle (TRAINstr + validation member)
        --> purely an informational str about how each member was preprocessed
    
    Xtrain_mean: numpy array
        Lat x Lon array of training members' SST means
    Xtrain_std: numpy array
        Lat x Lon array of training members' SST standard deviations
    Ytrain_median: float
        Median of T2M training members at location lat,lon
    lat,lon: int
        Index of the lat and lon location of T2M being predicted
        
    Returns:
    --------
    X_val: numpy array
        SST data used for validation (many members)
    
    Y_val: numpy array
        T2M data used for validation at LEAD = LEAD (many members)
        Converted into 0s and 1s
        
    lattxt,lontxt: str
        lat and lon str for location being predicted
        --> Used in saving model file
    '''
    
    xFINAME = RUN+'_ens'+str(VALmem)+'_SST_2015-2069_detrended_ensmean'+MEMstr+'_2.5x2.5.nc'
    X = xr.open_dataarray(DIR+xFINAME)
    X = X.where(X.time.dt.year >= 2050, drop=True)
    X = X[:-1*LEAD]
    X = is_month(X,Xmonths)
    X = X.where((X.lat <= 70) & (X.lat >= -70),drop=True)
    X = (X - Xtrain_mean) / Xtrain_std
    
    Xval = X.stack(z=('lat','lon'))
    
    ##########################################################################
    yFINAME = RUN+'_ens'+str(VALmem)+'_T2m_2015-2069_detrended_ensmean'+MEMstr+'_5x5.nc'
    Y = xr.open_dataarray(DIR+yFINAME) 
    Y = Y.where(Y.time.dt.year >= 2050, drop=True)
    Y = Y[LEAD:,ilat,ilon]
    Y = is_month(Y,Ymonths)
    
    Yval = Y - Ytrain_median
    Yval[np.where(Yval>=0)] = 1
    Yval[np.where(Yval<0)] = 0
        
    return Xval, Yval

#-----------------------------------------------------------------------------
def test(LEAD, TESTmem, DIR, RUN, MEMstr, Xtrain_mean, Xtrain_std, Ytrain_median, ilat, ilon, Xmonths = [8,9,10,11,12], Ymonths = [11,12,1,2,3]):

    '''
    Load X (SST) & Y (T2M) data from control or SAI run (>2035) and return
    base run testing data. These are the member(s) not used in training and validation for 2015-2034.
    
    Variables
    ----------
    LEAD: int
        the montly lead time between SST and T2M (e.g. LEAD = 2 --> Jan SST predicts March T2M)
    NUMMEMS: int
        number of members to loop through and load into a single array
    DIR: str
        directory path where SST and T2M data is located
    RUN: str
        Either 'control' or 'SAI' to denote whether to load in data following SSP2-4.5 or SAI-1.5 scenarios
    MEMstr: std
        The members used for detrending and removing the seasonal cycle (TRAINstr + validation member)
        --> purely an informational str about how each member was preprocessed
    
    Xtrain_mean: numpy array
        Lat x Lon array of training members' SST means
    Xtrain_std: numpy array
        Lat x Lon array of training members' SST standard deviations
    Ytrain_median: float
        Median of T2M training members at location lat,lon
    lat,lon: int
        Index of the lat and lon location of T2M being predicted
        
    Returns:
    --------
    X_test: numpy array
        SST data used for testing (many members)
    
    Y_test: numpy array
        T2M data used for testing at LEAD = LEAD (many members)
        Converted into 0s and 1s
        
    lattxt,lontxt: str
        lat and lon str for location being predicted
        --> Used in saving files
    '''
    xFINAME = RUN+'_ens'+str(TESTmem)+'_SST_2015-2069_detrended_ensmean'+MEMstr+'_2.5x2.5.nc'
    X = xr.open_dataarray(DIR+xFINAME)
    X = X.where(X.time.dt.year >= 2050, drop=True)
    X = X[:-1*LEAD]
    X = is_month(X,Xmonths)
    X = X.where((X.lat <= 70) & (X.lat >= -70),drop=True)
    X = (X - Xtrain_mean) / Xtrain_std
    
    yFINAME = RUN+'_ens'+str(TESTmem)+'_T2m_2015-2069_detrended_ensmean'+MEMstr+'_5x5.nc'
    Y = xr.open_dataarray(DIR+yFINAME) 
    Y = Y.where(Y.time.dt.year >= 2050, drop=True)
    Y = Y[LEAD:,ilat,ilon]
    Y = is_month(Y,Ymonths)
    
    Ytest = Y - Ytrain_median
    Ytest[np.where(Ytest>=0)] = 1
    Ytest[np.where(Ytest<0)] = 0
  
    ##########################################################################
    Xtest = X.stack(z=('lat','lon'))

    return Xtest, Ytest



#-----------------------------------------------------------------------------
#-------------------- B A S E L I N E ----------------------------------------
#-----------------------------------------------------------------------------
def train_baseline(DIR, LEAD, TRAINmem, MEMstr, ilat, ilon, Xmonths = [8,9,10,11,12], Ymonths = [11,12,1,2,3]):
    
    for t, trainmem in enumerate(TRAINmem):
        #print('Loading X training member '+str(trainmem))
        xFINAME = 'control_ens'+str(trainmem)+'_SST_2015-2069_detrended_ensmean'+MEMstr+'_2.5x2.5.nc'
        X = xr.open_dataarray(DIR+xFINAME)
        X = X.where(X.time.dt.year <= 2034, drop=True)
        X = X[:-1*LEAD]
        X = is_month(X,Xmonths)
        X = X.where((X.lat <= 70) & (X.lat >= -70),drop=True)
        
        if t == 0:
            Xall = xr.DataArray(np.zeros(shape=(len(TRAINmem),np.shape(X)[0],np.shape(X)[1],np.shape(X)[2]),dtype='float')+np.nan,
                                    name='SST',
                                    dims=('ens','time','lat','lon'),
                                    coords=[('ens',TRAINmem),
                                            ('time',X.time.data),
                                            ('lat',X.lat.data),('lon',X.lon.data)])
            
        Xall[t] = X

    Xall_stacked  = Xall.stack(time_all = ('ens','time'))
    Xall_stackedT = Xall_stacked.transpose('time_all','lat','lon')

    Xtrain_std    = np.nanstd(Xall_stackedT,axis=0)
    Xtrain_mean   = np.nanmean(Xall_stackedT, axis=0)
    
    Xtrain  = (Xall_stackedT - Xtrain_mean) / Xtrain_std
    Xtrain  = Xtrain.stack(z=('lat','lon'))
    
    ##########################################################################
    
    for t, trainmem in enumerate(TRAINmem):
        #print('Loading Y training member '+str(trainmem))

        yFINAME = 'control_ens'+str(trainmem)+'_T2m_2015-2069_detrended_ensmean'+MEMstr+'_5x5.nc'
        Y = xr.open_dataarray(DIR+yFINAME) 
        Y = Y.where(Y.time.dt.year <= 2034, drop=True)
        Y = Y[LEAD:,ilat,ilon]
        Y = is_month(Y,Ymonths)
        
        if t == 0:
            Yall = xr.DataArray(np.zeros(shape=(len(TRAINmem),np.shape(Y)[0]),dtype='float')+np.nan,
                                    name='T2m',
                                    dims=('ens','time'),
                                    coords=[('ens',TRAINmem),
                                            ('time',Y.time.data)])
        
    
        Yall[t] = Y

    Yall_stacked = Yall.stack(time_all=('ens','time'))
    
    Ytrain_median = np.median(Yall_stacked)
    Ytrain = Yall_stacked - Ytrain_median
    Ytrain[np.where(Ytrain>=0)[0]] = 1
    Ytrain[np.where(Ytrain<0)[0]] = 0
    
    return Xtrain, Ytrain, Xtrain_mean, Xtrain_std, Ytrain_median
 

#-----------------------------------------------------------------------------
def val_baseline(DIR, LEAD, VALmem, MEMstr, Xtrain_mean, Xtrain_std, Ytrain_median, ilat, ilon, Xmonths = [8,9,10,11,12], Ymonths = [11,12,1,2,3]):
    
    '''
    Load X (SST) & Y (T2M) data from base run (2015-2034 = no SAI or SSP2-4.5) and return
    validation data.
    
    Variables
    ----------
    LEAD: int
        the montly lead time between SST and T2M (e.g. LEAD = 2 --> Jan SST predicts March T2M)
    DIR: str
        directory path where SST and T2M data is located
    VALmem: list
        The member(s) used for validation (e.g. 3 = member 3)
        --> this/these are the member(s) loaded in this fuction
    MEMstr: str
        The members used for detrending and removing the seasonal cycle (TRAINstr + validation member)
        --> purely an informational str about how each member was preprocessed
    
    Xtrain_mean: numpy array
        Lat x Lon array of training members' SST means
    Xtrain_std: numpy array
        Lat x Lon array of training members' SST standard deviations
    Ytrain_median: float
        Median of T2M training members at location lat,lon
    lat,lon: int
        Index of the lat and lon location of T2M being predicted
        
    Returns:
    --------
    X_val: numpy array
        SST data used for validation (many members)
    
    Y_val: numpy array
        T2M data used for validation at LEAD = LEAD (many members)
        Converted into 0s and 1s
        
    lattxt,lontxt: str
        lat and lon str for location being predicted
        --> Used in saving model file
    '''
    
    xFINAME = 'control_ens'+str(VALmem)+'_SST_2015-2069_detrended_ensmean'+MEMstr+'_2.5x2.5.nc'
    X = xr.open_dataarray(DIR+xFINAME)
    X = X.where(X.time.dt.year <= 2034, drop=True)
    X = X[:-1*LEAD]
    X = is_month(X,Xmonths)
    X = X.where((X.lat <= 70) & (X.lat >= -70),drop=True)
    X = (X - Xtrain_mean) / Xtrain_std
    
    Xval = X.stack(z=('lat','lon'))
    
    ##########################################################################
    yFINAME = 'control_ens'+str(VALmem)+'_T2m_2015-2069_detrended_ensmean'+MEMstr+'_5x5.nc'
    Y = xr.open_dataarray(DIR+yFINAME) 
    Y = Y.where(Y.time.dt.year <= 2034, drop=True)
    Y = Y[LEAD:,ilat,ilon]
    Y = is_month(Y,Ymonths)
    
    Yval = Y - Ytrain_median
    Yval[np.where(Yval>=0)] = 1
    Yval[np.where(Yval<0)] = 0
        
    return Xval, Yval

#-----------------------------------------------------------------------------
def test_baseline(LEAD, TESTmem, DIR, MEMstr, Xtrain_mean, Xtrain_std, Ytrain_median, ilat, ilon, Xmonths = [8,9,10,11,12], Ymonths = [11,12,1,2,3]):

    '''
    Load X (SST) & Y (T2M) data from control or SAI run (>2035) and return
    base run testing data. These are the member(s) not used in training and validation for 2015-2034.
    
    Variables
    ----------
    LEAD: int
        the montly lead time between SST and T2M (e.g. LEAD = 2 --> Jan SST predicts March T2M)
    NUMMEMS: int
        number of members to loop through and load into a single array
    DIR: str
        directory path where SST and T2M data is located
    RUN: str
        Either 'control' or 'SAI' to denote whether to load in data following SSP2-4.5 or SAI-1.5 scenarios
    MEMstr: std
        The members used for detrending and removing the seasonal cycle (TRAINstr + validation member)
        --> purely an informational str about how each member was preprocessed
    
    Xtrain_mean: numpy array
        Lat x Lon array of training members' SST means
    Xtrain_std: numpy array
        Lat x Lon array of training members' SST standard deviations
    Ytrain_median: float
        Median of T2M training members at location lat,lon
    lat,lon: int
        Index of the lat and lon location of T2M being predicted
        
    Returns:
    --------
    X_test: numpy array
        SST data used for testing (many members)
    
    Y_test: numpy array
        T2M data used for testing at LEAD = LEAD (many members)
        Converted into 0s and 1s
        
    lattxt,lontxt: str
        lat and lon str for location being predicted
        --> Used in saving files
    '''
    xFINAME = 'control_ens'+str(TESTmem)+'_SST_2015-2069_detrended_ensmean'+MEMstr+'_2.5x2.5.nc'
    X = xr.open_dataarray(DIR+xFINAME)
    X = X.where(X.time.dt.year <= 2034, drop=True)
    X = X[:-1*LEAD]
    X = is_month(X,Xmonths)
    X = X.where((X.lat <= 70) & (X.lat >= -70),drop=True)
    X = (X - Xtrain_mean) / Xtrain_std
    
    yFINAME = 'control_ens'+str(TESTmem)+'_T2m_2015-2069_detrended_ensmean'+MEMstr+'_5x5.nc'
    Y = xr.open_dataarray(DIR+yFINAME) 
    Y = Y.where(Y.time.dt.year <= 2034, drop=True)
    Y = Y[LEAD:,ilat,ilon]
    Y = is_month(Y,Ymonths)
    
    Ytest = Y - Ytrain_median
    Ytest[np.where(Ytest>=0)] = 1
    Ytest[np.where(Ytest<0)] = 0
  
    ##########################################################################
    Xtest = X.stack(z=('lat','lon'))

    return Xtest, Ytest




#-----------------------------------------------------------------------------
def balance_classes(data):
    # Make validation classes balanced (2 classes)
    n_zero = np.shape(np.where(data==0)[0])[0]
    n_one  = np.shape(np.where(data==1)[0])[0]
    i_zero = np.where(data==0)[0]
    i_one  = np.where(data==1)[0]
    
    if n_one > n_zero:
        isubset_one = np.random.choice(i_one,size=n_zero,replace=False)
        i_new = np.sort(np.append(i_zero,isubset_one))
    elif n_one < n_zero:
        isubset_zero = np.random.choice(i_zero,size=n_one,replace=False)
        i_new = np.sort(np.append(isubset_zero,i_one))
    else:
        i_new = np.arange(0,len(data))
    
    return i_new
    
    
