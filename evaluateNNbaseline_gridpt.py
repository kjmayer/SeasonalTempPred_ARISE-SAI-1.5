#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 08:13:32 2023

@author: kmayer
"""

#%% IMPORT FUNCTIONS
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import sys
sys.path.append('/Users/kmayer/Documents/Scripts/PhDResearch/ARISE/seasonal_prediction/SAI_impact/ANN/global_lowres_NDJFM/functions/')
from ANN import defineNN, plot_results
from split_data_gridpt import train_baseline, balance_classes, test_baseline
#from hyperparams import get_params
#%% DEFINE MACHINE LEARNING FUNCTIONS & PARAMETERS
# ---------------- SET PARAMETERS ----------------
NLABEL     = 2                 
PATIENCE   = 20           
GLOBAL_SEED = 2147483648
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)


#%% DEFINE VARIABLE PARAMETERS
DIR = '/Volumes/Elements_External_HD/PhD/data/ARISE/processed/ensmean1-10_detrend/'
# DIR_MODEL = '/Users/kmayer/Documents/Scripts/PhDResearch/ARISE/seasonal_prediction/SAI_impact/ANN/global_lowres_NDJFM/model_weights/gridpts/baseline/'
DIR_MODEL = '/Users/kmayer/Documents/Scripts/PhDResearch/ARISE/seasonal_prediction/SAI_impact/ANN/global_lowres_NDJFM/model_weights/gridpts/baseline/MJJAS/'
# DIR_SAVE = '/Users/kmayer/Documents/Scripts/PhDResearch/ARISE/seasonal_prediction/SAI_impact/ANN/global_lowres_NDJFM/accvsconf/gridpts/baseline/'
DIR_SAVE = '/Users/kmayer/Documents/Scripts/PhDResearch/ARISE/seasonal_prediction/SAI_impact/ANN/global_lowres_NDJFM/accvsconf/gridpts/baseline/MJJAS/'

LEAD = 3 # 2 = 1 month inbetween (e.g. January --> March)

MEMstr = '1-10'
TRAINmem = [[1,2,3,4,5,6,7,8],
            [2,3,4,5,6,7,8,9],
            [3,4,5,6,7,8,9,10],
            [1,4,5,6,7,8,9,10],
            [1,2,5,6,7,8,9,10],
            [1,2,3,6,7,8,9,10],
            [1,2,3,4,7,8,9,10],
            [1,2,3,4,5,8,9,10],
            [1,2,3,4,5,6,9,10],
            [1,2,3,4,5,6,7,10]]
  
VALmem  = [9,10,1,2,3,4,5,6,7,8]          
TESTmem = [10,1,2,3,4,5,6,7,8,9]

#%% Params: 

# 25Sx120E   
experiments = {
    'baseline': {
        'hidden_layers':[8],
        'LR': 0.001,
        'L2': 0.25,
        'N_batch': 32
        },
    }

xmonths = [2,3,4,5,6]
ymonths = [5,6,7,8,9]


# 65Nx205E    
# experiments = {
#     'baseline': {
#         'hidden_layers':[32,8],
#         'LR': 0.001,
#         'L2': 0.5,
#         'N_batch': 32,
#         },
#     }

# xmonths = [8,9,10,11,12]
# ymonths = [11,12,1,2,3]


ilatrange = np.arange(6,33) # 60S-70N #np.arange(3,34)= 75S-75N #np.arange(23,34) #np.arange(13,14)
ilonrange = np.arange(0,72) # #np.arange(24,25)

#%%  CALCULATE & SAVE ACCURACY VS CONFIDENCE
RUN = 'baseline'

for m, mems in enumerate(TRAINmem): 
    acc = np.zeros(shape=(3,2,len(ilatrange),len(ilonrange))) + np.nan
    for la,ilat in enumerate(ilatrange): #np.arange(3,34)):
        for lo,ilon in enumerate(ilonrange): #np.arange(0,72)): 
        
            # ---------------- GET TESTING DATA ----------------
            # print('RUN: '+RUN+'\nTEST MEMBER: '+str(TESTmem[m]))
    
            _, _, Xtrain_mean, Xtrain_std, Ytrain_median = train_baseline(DIR = DIR,
                                                                LEAD = LEAD,
                                                                TRAINmem = mems,
                                                                MEMstr = MEMstr,
                                                                ilat = ilat,
                                                                ilon = ilon,
                                                                Xmonths = xmonths,
                                                                Ymonths = ymonths)
            
            # print('... LOAD TESTING DATA '+str(TESTmem[m])+' ...')
            Xtest, Ytest = test_baseline(DIR = DIR,
                                LEAD = LEAD,
                                TESTmem = TESTmem[m],
                                MEMstr = MEMstr,
                                Xtrain_mean = Xtrain_mean,
                                Xtrain_std = Xtrain_std,
                                Ytrain_median = Ytrain_median,
                                ilat = ilat,
                                ilon = ilon,
                                Xmonths = xmonths,
                                Ymonths = ymonths)
            
            if np.all(Ytest.isnull()):
                print('nans - skip location')
                continue # if Yval is nans, skip it
                
            X_test = np.asarray(Xtest,dtype='float')
            X_test[np.isnan(X_test)] = 0.
            Y_test = np.asarray(Ytest)
            
            i_new = balance_classes(data = Y_test)     
            Y_test = Y_test[i_new]
            X_test = X_test[i_new]
            
            # -------- GET PARAMETERS --------
            print(RUN+'_TEST MEMBER: '+str(TESTmem[m])+'_'+str(Ytest.lat.values)+'_'+str(Ytest.lon.values))
            params = experiments[RUN]
            #print(params)
            N_EPOCHS   = 1000 
            HIDDENS    = params['hidden_layers']
            LR_INIT    = params['LR']    
            RIDGE      = params['L2']
            BATCH_SIZE = params['N_batch']  
            
            #% ------------------------ EVALUATE NN ------------------------
            
            for NETWORK_SEED in np.arange(0,3):  
                # print(NETWORK_SEED)
                # ----- Define NN Architecture -----
                tf.keras.backend.clear_session() 
                model = defineNN(HIDDENS,
                                 input_shape = X_test.shape[1],
                                 output_shape=NLABEL,
                                 ridge_penalty=RIDGE,
                                 act_fun='relu',
                                 network_seed=NETWORK_SEED)
                # ----- Compile NN -----
                METRICS = [tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy", dtype=None)]
                LOSS_FUNCTION = tf.keras.losses.SparseCategoricalCrossentropy()
                OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_INIT)
                
                model.compile(optimizer = OPTIMIZER, loss = LOSS_FUNCTION, metrics = METRICS)   
                
                # fi = 'baseline_T2m'+str(Ytest.lat.values)+'N_'+str(Ytest.lon.values)+'E_SST70S-70N_valmem'+str(VALmem[m])+'_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_seed'+str(NETWORK_SEED)+'.h5'
                fi = 'baseline_MJJAS_T2m'+str(Ytest.lat.values)+'N_'+str(Ytest.lon.values)+'E_SST70S-70N_valmem'+str(VALmem[m])+'_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_seed'+str(NETWORK_SEED)+'.h5'
                model.load_weights(DIR_MODEL+fi)
               
                # ----- EVALUATE MODEL -----  
                conf_pred = model.predict(X_test)           # softmax output
                
                cat_pred  = np.argmax(conf_pred, axis = -1) # categorical output
                max_conf  = np.max(conf_pred, axis = -1)   # predicted category confidence
                rand_prob = np.random.normal(loc=0.0,scale=1.0,size=np.shape(max_conf)[0])*np.min(max_conf)*10**-5
                max_conf += rand_prob #add small random noise to account for when the network uses the same confidence for many predictions
                
                # all predictions
                icorr  = np.where(cat_pred == Y_test)[0]
                if np.shape(Ytest)[0] == 0:
                    acc[NETWORK_SEED,0,la,lo] = 0.
                else:
                    acc[NETWORK_SEED,0,la,lo] = (len(icorr)/np.shape(Ytest)[0]) * 100
                
                # 20% most confident
                i_cover = np.where(max_conf >= np.percentile(max_conf,80))[0]
                icorr   = np.where(cat_pred[i_cover] == Y_test[i_cover])[0]
                if len(i_cover) == 0:
                    acc[NETWORK_SEED,1,la,lo] = 0.
                else:
                    acc[NETWORK_SEED,1,la,lo] = (len(icorr)/len(i_cover)) * 100
                    
                
                if NETWORK_SEED == 0 and m == 0 and la == 0 and lo == 0:
                    print(acc[NETWORK_SEED,:,la,lo])
                    print(np.shape(Ytest)[0])
    
    acc_xr = xr.DataArray(acc, 
                          dims=['seed','percentile','lat','lon'], 
                          coords = [('seed',np.arange(0,3)),
                                    ('percentile', [0,80]),
                                    ('lat',np.arange(-60,75,5)),#np.arange(-75,80,5)),
                                    ('lon',np.arange(0,360,5))])
    
            
    # ----- SAVE ACC VS CONF -----
    # FI_SAVE = 'accvsconf_baseline_75S-75Nx0-360E_seeds0-2_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc'
    FI_SAVE = 'accvsconf_baseline_MJJAS_60S-70Nx0-360E_seeds0-2_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc'

    acc_xr.to_netcdf(DIR_SAVE+FI_SAVE)