#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:46:07 2024

@author: kjmayer
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
sys.path.append('/Volumes/Elements_PhD/ARISE/global_analysis/functions/')
from ANN import defineNN, plot_results
from split_data_gridpt import train, balance_classes, test
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
DIR = '/Volumes/Elements_PhD/ARISE/data/processed/ensmean1-10_detrend/'
# DIR_MODEL = '/Volumes/Elements_PhD/ARISE/global_analysis/model_weights/gridpts/MJJAS/'
# DIR_SAVE = '/Volumes/Elements_PhD/ARISE/global_analysis/accvsconf/gridpts/MJJAS/'
DIR_MODEL = '/Volumes/Elements_PhD/ARISE/global_analysis/model_weights/gridpts/'
DIR_SAVE = '/Volumes/Elements_PhD/ARISE/global_analysis/accvsconf/gridpts/'

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
# experiments = {
#     'SAI': {
#         'hidden_layers':[32],
#         'LR': 0.001,
#         'L2': 1.0,
#         'N_batch': 32
#         },
#     'control': {
#         'hidden_layers':[32,8],
#         'LR': 0.001,
#         'L2': 5.0,
#         'N_batch': 32
#         },
#     }


# 65Nx205E    
experiments = {
    'SAI': {
        'hidden_layers':[32,8],#[8],
        'LR': 0.001,#0.001,
        'L2': 0.25,#1.0,
        'N_batch': 32,#32
        },
    'control': {
        'hidden_layers':[64,8],#[16],
        'LR': 0.001,#0.001,#0.01,
        'L2': 1.0,##0.5,
        'N_batch': 32,#32
        },
    }

#%%  CALCULATE & SAVE ACCURACY VS CONFIDENCE
ilatrange = np.arange(6,33)
ilonrange = np.arange(0,72)

for RUN in ['control','SAI']:
    for m, mems in enumerate(TRAINmem): 
        acc = np.zeros(shape=(3,len(ilatrange),len(ilonrange))) + np.nan
        for la,ilat in enumerate(ilatrange): #np.arange(3,34)):
            for lo,ilon in enumerate(ilonrange): #np.arange(0,72)): 
            
                # ---------------- GET TESTING DATA ----------------
                # print('RUN: '+RUN+'\nTEST MEMBER: '+str(TESTmem[m]))
        
                _, _, Xtrain_mean, Xtrain_std, Ytrain_median = train(DIR = DIR,
                                                                    RUN = RUN, 
                                                                    LEAD = LEAD,
                                                                    TRAINmem = mems,
                                                                    MEMstr = MEMstr,
                                                                    ilat = ilat,
                                                                    ilon = ilon,
                                                                    Xmonths = [8,9,10,11,12], #[2,3,4,5,6], #[8,9,10,11,12],
                                                                    Ymonths = [11,12,1,2,3]) #[5,6,7,8,9]) #[11,12,1,2,3])
                
                # print('... LOAD TESTING DATA '+str(TESTmem[m])+' ...')
                Xtest, Ytest = test(DIR = DIR,
                                    RUN = RUN,
                                    LEAD = LEAD,
                                    TESTmem = TESTmem[m],
                                    MEMstr = MEMstr,
                                    Xtrain_mean = Xtrain_mean,
                                    Xtrain_std = Xtrain_std,
                                    Ytrain_median = Ytrain_median,
                                    ilat = ilat,
                                    ilon = ilon,
                                    Xmonths = [8,9,10,11,12], #[2,3,4,5,6], #[8,9,10,11,12],
                                    Ymonths = [11,12,1,2,3]) #[5,6,7,8,9]) #[11,12,1,2,3]))
                
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
                    
                    fi = RUN+'_T2m'+str(Ytest.lat.values)+'N_'+str(Ytest.lon.values)+'E_SST70S-70N_valmem'+str(VALmem[m])+'_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_seed'+str(NETWORK_SEED)+'.h5'
                    # fi = RUN+'_MJJAS_T2m'+str(Ytest.lat.values)+'N_'+str(Ytest.lon.values)+'E_SST70S-70N_valmem'+str(VALmem[m])+'_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_seed'+str(NETWORK_SEED)+'.h5'
                    model.load_weights(DIR_MODEL+fi)
                   
                    # ----- EVALUATE MODEL -----  
                    conf_pred = model.predict(X_test)           # softmax output
                    
                    cat_pred  = np.argmax(conf_pred, axis = -1) # categorical output
                    max_conf  = np.max(conf_pred, axis = -1)   # predicted category confidence
                    rand_prob = np.random.normal(loc=0.0,scale=1.0,size=np.shape(max_conf)[0])*np.min(max_conf)*10**-5
                    max_conf += rand_prob #add small random noise to account for when the network uses the same confidence for many predictions
                    
                    # 20% least confident
                    i_cover = np.where(max_conf <= np.percentile(max_conf,20))[0]
                    icorr   = np.where(cat_pred[i_cover] == Y_test[i_cover])[0]
                    if len(i_cover) == 0:
                        acc[NETWORK_SEED,la,lo] = 0.
                    else:
                        acc[NETWORK_SEED,la,lo] = (len(icorr)/len(i_cover)) * 100
                        
                    
                    if NETWORK_SEED == 0 and m == 0 and la == 0 and lo == 0:
                        print(acc[NETWORK_SEED,la,lo])
                        print(np.shape(Ytest)[0])
        
        acc_xr = xr.DataArray(acc, 
                              dims=['seed','lat','lon'], 
                              coords = [('seed',np.arange(0,3)),
                                        ('lat',np.arange(-60,75,5)),#np.arange(-75,80,5)),
                                        ('lon',np.arange(0,360,5))])
        
                
        # ----- SAVE ACC VS CONF -----
        FI_SAVE = 'accvs20leastconf_'+RUN+'_75S-75Nx0-360E_seeds0-2_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc'
        # FI_SAVE = 'accvs20leastconf_'+RUN+'_MJJAS_60S-70Nx0-360E_seeds0-2_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc'

        acc_xr.to_netcdf(DIR_SAVE+FI_SAVE)