#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:34:42 2022

@author: kmayer
"""

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
from split_data_gridpt import train, balance_classes, val

#%% DEFINING MACHINE LEARNING FUNCTIONS & PARAMETERS

# ---------------- LEARNING RATE CALLBACK FUNCTION ----------------
def scheduler(epoch, lr):
    # This function keeps the initial learning rate for the first ten epochs
    # and decreases it exponentially after that.
    if epoch < 10:
        return lr
    else:
        return lr * tf.constant(.9,dtype=tf.float32)

# ---------------- SET PARAMETERS ----------------
NLABEL     = 2                 
PATIENCE   = 20         
GLOBAL_SEED = 2147483648
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

#%% VARIABLE PARAMETERS
DIR = '/Volumes/Elements_External_HD/PhD/data/ARISE/processed/ensmean1-10_detrend/'
DIR_MODEL = '/Users/kmayer/Documents/Scripts/PhDResearch/ARISE/seasonal_prediction/SAI_impact/ANN/global_lowres_NDJFM/model_weights/gridpts/MJJAS/'
LEAD = 3 # 3 = 2 month inbetween (e.g. Septmeber --> December)

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
    'SAI': {
        'hidden_layers':[32],
        'LR': 0.001,
        'L2': 1.0,
        'N_batch': 32
        },
    'control': {
        'hidden_layers':[32,8],
        'LR': 0.001,
        'L2': 5.0,
        'N_batch': 32
        },
    }


# 65Nx205E    
# experiments = {
#     'SAI': {
#         'hidden_layers':[32,8],#[8],
#         'LR': 0.001,#0.001,
#         'L2': 0.25,#1.0,
#         'N_batch': 32,#32
#         },
#     'control': {
#         'hidden_layers':[64,8],#[16],
#         'LR': 0.001,#0.001,#0.01,
#         'L2': 1.0,##0.5,
#         'N_batch': 32,#32
#         },
#     }

# 25N - 75N = 23 - 33 (Y.lat[23:34])
#ilat = 31 #65N
# 205E = 41, 125E = 25 = NE Russia

ilatrange = np.arange(6,33) # 60S-70N #np.arange(3,34)= 75S-75N #np.arange(23,34) #np.arange(13,14)
ilonrange = np.arange(0,72) # #np.arange(24,25)
#%% 
for RUN in ['control','SAI']:
    print('RUN: '+RUN)#
    for m, mems in enumerate(TRAINmem):
        for ilat in ilatrange:
            for ilon in ilonrange:        
                # ---------------- GET TRAINING & VALIDATION DATA ----------------
                #print('RUN: '+RUN+'\nTEST MEMBER: '+str(TESTmem[m]))
        
                print('... LOAD TRAINING DATA ...')
                Xtrain, Ytrain, Xtrain_mean, Xtrain_std, Ytrain_median = train(DIR = DIR,
                                                                               RUN = RUN, 
                                                                               LEAD = LEAD,
                                                                               TRAINmem = mems,
                                                                               MEMstr = MEMstr,
                                                                               ilat = ilat,
                                                                               ilon = ilon,
                                                                               Xmonths = [2,3,4,5,6], #[8,9,10,11,12],
                                                                               Ymonths = [5,6,7,8,9]) #[11,12,1,2,3])
        
                
                print('... LOAD VALIDATION MEMBER '+str(VALmem[m])+' ...')
                Xval, Yval = val(DIR = DIR,
                                 RUN = RUN,
                                 LEAD = LEAD,
                                 VALmem = VALmem[m],
                                 MEMstr = MEMstr,
                                 Xtrain_mean = Xtrain_mean,
                                 Xtrain_std = Xtrain_std,
                                 Ytrain_median = Ytrain_median,
                                 ilat = ilat,
                                 ilon = ilon,
                                 Xmonths = [2,3,4,5,6], #[8,9,10,11,12],
                                 Ymonths = [5,6,7,8,9]) #[11,12,1,2,3])
                
                if np.all(Yval.isnull()):
                    print('nans - skip location')
                    continue # if Yval is nans, skip it
                
                
                X_train = np.asarray(Xtrain,dtype='float')
                X_train[np.isnan(X_train)] = 0.
                Y_train = np.asarray(Ytrain)
                
                X_val = np.asarray(Xval,dtype='float')
                X_val[np.isnan(X_val)] = 0.
                Y_val = np.asarray(Yval)
                    
                i_new = balance_classes(data = Y_val)     
                Y_val = Y_val[i_new]
                X_val = X_val[i_new]
                
                # -------- GET PARAMETERS --------
                print(RUN+'_'+str(Yval.lat.values)+'_'+str(Yval.lon.values))
                params = experiments[RUN]
                #print(params)
                N_EPOCHS   = 1000 
                HIDDENS    = params['hidden_layers']
                LR_INIT    = params['LR']    
                RIDGE      = params['L2']
                BATCH_SIZE = params['N_batch'] 
                
                #% ------------------------ TRAIN NN ------------------------
                for NETWORK_SEED in np.arange(0,3):  
                    # ----- Define NN Architecture -----
                    tf.keras.backend.clear_session() 
                    model = defineNN(HIDDENS,
                                     input_shape = X_train.shape[1:],
                                     output_shape = NLABEL,
                                     ridge_penalty = RIDGE,
                                     act_fun = 'relu',
                                     network_seed = NETWORK_SEED)
                    # ----- Compile NN -----
                    METRICS = [tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy", dtype=None)]
                    LOSS_FUNCTION = tf.keras.losses.SparseCategoricalCrossentropy()
                    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_INIT)
        
                    model.compile(optimizer = OPTIMIZER, loss = LOSS_FUNCTION, metrics = METRICS)   
                    
                    # ----- Callbacks -----
                    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=PATIENCE,
                                                                    mode='auto',
                                                                    restore_best_weights=True,
                                                                    verbose=0)  
                    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=0)
                    callbacks = [es_callback,lr_callback]
                    
                    # ----- TRAINING NETWORK -----
                    start_time = time.time()
                    history = model.fit(X_train, Y_train,
                                        validation_data=(X_val, Y_val),
                                        batch_size=BATCH_SIZE,
                                        epochs=N_EPOCHS,
                                        shuffle=True,
                                        verbose=0,
                                        callbacks=callbacks,
                                       )
                    stop_time = time.time()
                    tf.print(f"Elapsed time during fit = {stop_time - start_time:.2f} seconds\n")
                    
                    if NETWORK_SEED == 0 and ilat == 13 and ilon == 24: #31 and ilon == 41:
                        #----- PLOT THE RESULTS -----
                        plot_results(
                            history,
                            exp_info=(N_EPOCHS, HIDDENS, LR_INIT, BATCH_SIZE, NETWORK_SEED, PATIENCE, RIDGE),
                            showplot=True
                        )   
                        
                        # ----- PRINT THE RESULTS -----
                        predictions = np.argmax(model.predict(X_val),axis=-1)
                        confusion = tf.math.confusion_matrix(labels=Y_val, predictions=predictions)
                        
                        '''# RECALL/Conditional Acc = correct predictions of a class / total number of true occurances of that class
                        # zero_recall  = (np.sum(confusion[0,0])/np.sum(confusion[0,:])) * 100
                        # one_recall   = (np.sum(confusion[1,1])/np.sum(confusion[1,:])) * 100'''
                        
                        # PRECISION = correct predictions of a class / total predictions for that class
                        zero_precision  = (np.sum(confusion[0,0])/np.sum(confusion[:,0])) * 100
                        one_precision   = (np.sum(confusion[1,1])/np.sum(confusion[:,1])) * 100
                            
                        # Number of times network predicts a given class
                        zero_predictions  = (np.shape(np.where(predictions==0))[1]/predictions.shape[0])* 100
                        one_predictions   = (np.shape(np.where(predictions==1))[1]/predictions.shape[0])* 100
                        
                        print('Zero prediction accuracy: '+str(zero_precision)[:2]+'%')
                        print('Zero: '+str(zero_predictions)[:3]+'% of predictions')
                        print('One prediction accuracy: '+str(one_precision)[:2]+'%')
                        print('One: '+str(one_predictions)[:3]+'% of predictions')
                    
                    # #----- SAVE MODEL -----  
                    # fi = RUN+'_T2m'+str(Yval.lat.values)+'N_'+str(Yval.lon.values)+'E_SST70S-70N_valmem'+str(VALmem[m])+'_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_seed'+str(NETWORK_SEED)+'.h5'
                    fi = RUN+'_MJJAS_T2m'+str(Yval.lat.values)+'N_'+str(Yval.lon.values)+'E_SST70S-70N_valmem'+str(VALmem[m])+'_testmem'+str(TESTmem[m])+'_layers'+str(HIDDENS[0])+'_ADAM'+str(LR_INIT)+'_L2'+str(RIDGE)+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_seed'+str(NETWORK_SEED)+'.h5'
                    model.save_weights(DIR_MODEL+fi)
        
                
                
