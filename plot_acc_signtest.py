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
def diff_sign_test(x1, x2):

    diff_sign = x2 - x1
    diff_sign = xr.where(diff_sign>=0, 1, diff_sign)
    diff_sign = xr.where(diff_sign<0, np.nan, diff_sign)

    N = diff_sign.shape[0] * diff_sign.shape[1]
    prob = binom.cdf(k=np.sum(diff_sign, axis=(0,1)), n=N, p=0.5)

    # make it a two-sided test
    prob = np.where(prob>0.5, 1-prob, prob)

    # nan out the land
    nan_grid = xr.where(np.isnan(np.sum((x2-x1).values, axis=(0,1))), np.nan, 1.)
    prob = prob * nan_grid

    print(N, prob.shape)
    return np.sum(diff_sign, axis=(0,1)) * nan_grid / N, prob


#%% VARIABLES
DIR_NDJFM = '/Volumes/Elements_PhD/ARISE/global_analysis/accvsconf/gridpts/'
DIR_MJJAS = '/Volumes/Elements_PhD/ARISE/global_analysis/accvsconf/gridpts/MJJAS/'

DIR_base_NDJFM = '/Volumes/Elements_PhD/ARISE/global_analysis/accvsconf/gridpts/baseline/'
DIR_base_MJJAS = '/Volumes/Elements_PhD/ARISE/global_analysis/accvsconf/gridpts/baseline/MJJAS/'

DIR_FIG = '/Volumes/Elements_PhD/ARISE/global_analysis/plots/finalfigures/'

MEMstr = '1-10'
LEAD = 3

   
experiments = {
    
    # 25Sx120E
    'SAI_MJJAS': {
        'hidden_layers':[32],
        'LR': 0.001,
        'L2': 1.0,
        'N_batch': 32
        },
    'control_MJJAS': {
        'hidden_layers':[32,8],
        'LR': 0.001,
        'L2': 5.0,
        'N_batch': 32
        },
    'baseline_MJJAS': {
        'hidden_layers':[8],
        'LR': 0.001,
        'L2': 0.25,
        'N_batch': 32
        },
    
    # 65Nx205E
    'SAI_NDJFM': {
            'hidden_layers':[32,8],#[8],
            'LR': 0.001,#0.001,
            'L2': 0.25,#1.0,
            'N_batch': 32,#32
            },
    'control_NDJFM': {
            'hidden_layers':[64,8],#[16],
            'LR': 0.001,#0.001,#0.01,
            'L2': 1.0,##0.5,
            'N_batch': 32,#32
            },
    'baseline_NDJFM': {
            'hidden_layers':[32,8],
            'LR': 0.001,
            'L2': 0.5,
            'N_batch': 32,
            },
    
    }

N_EPOCHS   = 1000
BATCH_SIZE = 32 
PATIENCE   = 20

TESTmem = [10,1,2,3,4,5,6,7,8,9]
#%% LOAD
for m, mem in enumerate(TESTmem):
    if m == 0:
        cont_NDJFM = xr.open_dataarray(DIR_NDJFM+'accvsconf_control_75S-75Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['control_NDJFM']['hidden_layers'][0])+'_ADAM'+str(experiments['control_NDJFM']['LR'])+'_L2'+str(experiments['control_NDJFM']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        sai_NDJFM  = xr.open_dataarray(DIR_NDJFM+'accvsconf_SAI_75S-75Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['SAI_NDJFM']['hidden_layers'][0])+'_ADAM'+str(experiments['SAI_NDJFM']['LR'])+'_L2'+str(experiments['SAI_NDJFM']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        base_NDJFM = xr.open_dataarray(DIR_base_NDJFM+'accvsconf_baseline_75S-75Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['baseline_NDJFM']['hidden_layers'][0])+'_ADAM'+str(experiments['baseline_NDJFM']['LR'])+'_L2'+str(experiments['baseline_NDJFM']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')

        cont_MJJAS = xr.open_dataarray(DIR_MJJAS+'accvsconf_control_MJJAS_60S-70Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['control_MJJAS']['hidden_layers'][0])+'_ADAM'+str(experiments['control_MJJAS']['LR'])+'_L2'+str(experiments['control_MJJAS']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        sai_MJJAS = xr.open_dataarray(DIR_MJJAS+'accvsconf_SAI_MJJAS_60S-70Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['SAI_MJJAS']['hidden_layers'][0])+'_ADAM'+str(experiments['SAI_MJJAS']['LR'])+'_L2'+str(experiments['SAI_MJJAS']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        base_MJJAS = xr.open_dataarray(DIR_base_MJJAS+'accvsconf_baseline_MJJAS_60S-70Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['baseline_MJJAS']['hidden_layers'][0])+'_ADAM'+str(experiments['baseline_MJJAS']['LR'])+'_L2'+str(experiments['baseline_MJJAS']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        
    else:
        cont_NDJFM_temp = xr.open_dataarray(DIR_NDJFM+'accvsconf_control_75S-75Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['control_NDJFM']['hidden_layers'][0])+'_ADAM'+str(experiments['control_NDJFM']['LR'])+'_L2'+str(experiments['control_NDJFM']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        sai_NDJFM_temp  = xr.open_dataarray(DIR_NDJFM+'accvsconf_SAI_75S-75Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['SAI_NDJFM']['hidden_layers'][0])+'_ADAM'+str(experiments['SAI_NDJFM']['LR'])+'_L2'+str(experiments['SAI_NDJFM']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        base_NDJFM_temp = xr.open_dataarray(DIR_base_NDJFM+'accvsconf_baseline_75S-75Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['baseline_NDJFM']['hidden_layers'][0])+'_ADAM'+str(experiments['baseline_NDJFM']['LR'])+'_L2'+str(experiments['baseline_NDJFM']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        
        cont_MJJAS_temp = xr.open_dataarray(DIR_MJJAS+'accvsconf_control_MJJAS_60S-70Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['control_MJJAS']['hidden_layers'][0])+'_ADAM'+str(experiments['control_MJJAS']['LR'])+'_L2'+str(experiments['control_MJJAS']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        sai_MJJAS_temp = xr.open_dataarray(DIR_MJJAS+'accvsconf_SAI_MJJAS_60S-70Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['SAI_MJJAS']['hidden_layers'][0])+'_ADAM'+str(experiments['SAI_MJJAS']['LR'])+'_L2'+str(experiments['SAI_MJJAS']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        base_MJJAS_temp = xr.open_dataarray(DIR_base_MJJAS+'accvsconf_baseline_MJJAS_60S-70Nx0-360E_seeds0-2_testmem'+str(mem)+'_layers'+str(experiments['baseline_MJJAS']['hidden_layers'][0])+'_ADAM'+str(experiments['baseline_MJJAS']['LR'])+'_L2'+str(experiments['baseline_MJJAS']['L2'])+'_patience'+str(PATIENCE)+'_batch'+str(BATCH_SIZE)+'_mem1-10.nc')
        
        
        cont_NDJFM = xr.concat([cont_NDJFM,cont_NDJFM_temp],dim='ens')
        sai_NDJFM  = xr.concat([sai_NDJFM,sai_NDJFM_temp],dim='ens')
        base_NDJFM = xr.concat([base_NDJFM,base_NDJFM_temp],dim='ens')
        
        cont_MJJAS = xr.concat([cont_MJJAS,cont_MJJAS_temp],dim='ens')
        sai_MJJAS  = xr.concat([sai_MJJAS,sai_MJJAS_temp],dim='ens')
        base_MJJAS = xr.concat([base_MJJAS,base_MJJAS_temp],dim='ens')

#%%
# Calculate mean and standard deviation across ensemble members (N=10) and network seed (N=3)

# select confidence level (all/100% or 20%):
iper = 0 # 0 = all predictions; 1 = 20% most confident predictions

# ----- NDJFM -----
cont_NDJFM_mean = cont_NDJFM.mean(['ens','seed'],skipna=True)[iper]
sai_NDJFM_mean  = sai_NDJFM.mean(['ens','seed'],skipna=True)[iper]
base_NDJFM_mean = base_NDJFM.mean(['ens','seed'],skipna=True)[iper]

base_NDJFM_std = base_NDJFM.std(['ens','seed'],skipna=True)[iper]
sai_NDJFM_std  = sai_NDJFM.std(['ens','seed'],skipna=True)[iper]
cont_NDJFM_std = cont_NDJFM.std(['ens','seed'],skipna=True)[iper]

# diff_saibase_NDJFM  = sai_NDJFM_mean - base_NDJFM_mean
# diff_contbase_NDJFM = cont_NDJFM_mean - base_NDJFM_mean

# ----- MJJAS -----
cont_MJJAS_mean = cont_MJJAS.mean(['ens','seed'],skipna=True)[iper]
sai_MJJAS_mean  = sai_MJJAS.mean(['ens','seed'],skipna=True)[iper]
base_MJJAS_mean = base_MJJAS.mean(['ens','seed'],skipna=True)[iper]

base_MJJAS_std = base_MJJAS.std(['ens','seed'],skipna=True)[iper]
sai_MJJAS_std  = sai_MJJAS.std(['ens','seed'],skipna=True)[iper]
cont_MJJAS_std = cont_MJJAS.std(['ens','seed'],skipna=True)[iper]

# diff_saibase_MJJAS  = sai_MJJAS_mean - base_MJJAS_mean
# diff_contbase_MJJAS = cont_MJJAS_mean - base_MJJAS_mean


#%%
# SIGN TEST:
# LOOK AT SIGN ACROSS MEMBERS AND SEEDS
ds_saibase_NDJFM, prob_saibase_NDJFM = diff_sign_test(base_NDJFM, sai_NDJFM)
ds_contbase_NDJFM, prob_contbase_NDJFM = diff_sign_test(base_NDJFM, cont_NDJFM)
ds_saibase_MJJAS, prob_saibase_MJJAS = diff_sign_test(base_MJJAS, sai_MJJAS)
ds_contbase_MJJAS, prob_contbase_MJJAS = diff_sign_test(base_MJJAS, cont_MJJAS)

# AVERAGE OVER SEEDS FIRST
# ds_saibase_NDJFM, prob_saibase_NDJFM = diff_sign_test(base_NDJFM.mean(axis=1, keepdims=True), sai_NDJFM.mean(axis=1, keepdims=True))
# ds_contbase_NDJFM, prob_contbase_NDJFM = diff_sign_test(base_NDJFM.mean(axis=1, keepdims=True), cont_NDJFM.mean(axis=1, keepdims=True))
# ds_saibase_MJJAS, prob_saibase_MJJAS = diff_sign_test(base_MJJAS.mean(axis=1, keepdims=True), sai_MJJAS.mean(axis=1, keepdims=True))
# ds_contbase_MJJAS, prob_contbase_MJJAS = diff_sign_test(base_MJJAS.mean(axis=1, keepdims=True), cont_MJJAS.mean(axis=1, keepdims=True))

# GRAB ONLY ONE SEED
# seed = 1
# ds_saibase_NDJFM, prob_saibase_NDJFM = diff_sign_test(base_NDJFM[:,seed:seed+1,:,:,:], sai_NDJFM[:,seed:seed+1,:,:,:])
# ds_contbase_NDJFM, prob_contbase_NDJFM = diff_sign_test(base_NDJFM[:,seed:seed+1,:,:,:], cont_NDJFM[:,seed:seed+1,:,:,:])
# ds_saibase_MJJAS, prob_saibase_MJJAS = diff_sign_test(base_MJJAS[:,seed:seed+1,:,:,:], sai_MJJAS[:,seed:seed+1:,:,:])
# ds_contbase_MJJAS, prob_contbase_MJJAS = diff_sign_test(base_MJJAS[:,seed:seed+1,:,:,:], cont_MJJAS[:,seed:seed+1,:,:,:])


# ------------------------------------------------------------------------------------------------------------

# CHOOSE THE CONFIDENCE CUTOFF
ds_saibase_NDJFM, prob_saibase_NDJFM = ds_saibase_NDJFM[iper,:,:], prob_saibase_NDJFM[iper,:,:]
ds_contbase_NDJFM, prob_contbase_NDJFM = ds_contbase_NDJFM[iper,:,:], prob_contbase_NDJFM[iper,:,:]
ds_saibase_MJJAS, prob_saibase_MJJAS = ds_saibase_MJJAS[iper,:,:], prob_saibase_MJJAS[iper,:,:]
ds_contbase_MJJAS, prob_contbase_MJJAS = ds_contbase_MJJAS[iper,:,:], prob_contbase_MJJAS[iper,:,:]

# USE KIRSTEN'S NAMES
diff_saibase_NDJFM = ds_saibase_NDJFM
diff_contbase_NDJFM = ds_contbase_NDJFM
diff_saibase_MJJAS = ds_saibase_MJJAS
diff_contbase_MJJAS = ds_contbase_MJJAS

#%%
# code for wilks FDR & associated lat-lon meshgrid for plotting significance dots on figure
alpha = 0.1

#%%

# ---------------------- NDJFM ----------------------

# ----------- SAI - BASE -----------
Pvals_saibase_NDJFM = np.sort(prob_saibase_NDJFM.flatten())
y = (np.arange(1,len(Pvals_saibase_NDJFM)+1,1)/len(Pvals_saibase_NDJFM))*alpha
k = np.where(Pvals_saibase_NDJFM > y)[0][0] # find first instance where Pvals > y

longrid_saibase_NDJFM,latgrid_saibase_NDJFM = np.meshgrid(diff_saibase_NDJFM.lon,diff_saibase_NDJFM.lat)
if k == 0:
   fdrP_saibase_NDJFM = 0
   siglon_saibase_NDJFM = np.asarray([np.nan])
   siglat_saibase_NDJFM = np.asarray([np.nan])
else:
    fdrP_saibase_NDJFM = Pvals_saibase_NDJFM[k-1] # use the index right before Pvals > y -> all Pvals should be less than fdrPcutoff to pass

    siglon_saibase_NDJFM = longrid_saibase_NDJFM[np.asarray(prob_saibase_NDJFM<=fdrP_saibase_NDJFM)]
    siglat_saibase_NDJFM = latgrid_saibase_NDJFM[np.asarray(prob_saibase_NDJFM<=fdrP_saibase_NDJFM)]

# ----------- SSP (control) - BASE -----------
Pvals_contbase_NDJFM = np.sort(prob_contbase_NDJFM.flatten())
y = (np.arange(1,len(Pvals_contbase_NDJFM)+1,1)/len(Pvals_contbase_NDJFM))*alpha
k = np.where(Pvals_contbase_NDJFM > y)[0][0] # find first instance where Pvals > y

longrid_contbase_NDJFM,latgrid_contbase_NDJFM = np.meshgrid(diff_contbase_NDJFM.lon,diff_contbase_NDJFM.lat)
if k == 0:
    fdrP_contbase_NDJFM = 0
    siglon_contbase_NDJFM = np.asarray([np.nan])
    siglat_contbase_NDJFM = np.asarray([np.nan])
else:
    fdrP_contbase_NDJFM = Pvals_contbase_NDJFM[k-1] # use the index right before Pvals > y -> all Pvals should be less than fdrPcutoff to pass
    siglon_contbase_NDJFM = longrid_contbase_NDJFM[np.asarray(prob_contbase_NDJFM<=fdrP_contbase_NDJFM)]
    siglat_contbase_NDJFM = latgrid_contbase_NDJFM[np.asarray(prob_contbase_NDJFM<=fdrP_contbase_NDJFM)]

#%%
# ---------------------- MJJAS ----------------------

# ----------- SAI- BASE -----------
Pvals_saibase_MJJAS = np.sort(prob_saibase_MJJAS.flatten())
y = (np.arange(1,len(Pvals_saibase_MJJAS)+1,1)/len(Pvals_saibase_MJJAS))*alpha
k = np.where(Pvals_saibase_MJJAS > y)[0][0] # find first instance where Pvals > y

longrid_saibase_MJJAS,latgrid_saibase_MJJAS = np.meshgrid(diff_saibase_MJJAS.lon,diff_saibase_MJJAS.lat)
if k == 0:
   fdrP_saibase_MJJAS = 0
   siglon_saibase_MJJAS = np.asarray([np.nan])
   siglat_saibase_MJJAS = np.asarray([np.nan])
else:
    fdrP_saibase_MJJAS = Pvals_saibase_MJJAS[k-1] # use the index right before Pvals > y -> all Pvals should be less than fdrPcutoff to pass

    siglon_saibase_MJJAS = longrid_saibase_MJJAS[np.asarray(prob_saibase_MJJAS<=fdrP_saibase_MJJAS)]
    siglat_saibase_MJJAS = latgrid_saibase_MJJAS[np.asarray(prob_saibase_MJJAS<=fdrP_saibase_MJJAS)]


# ----------- SSP - BASE -----------
Pvals_contbase_MJJAS = np.sort(prob_contbase_MJJAS.flatten())
y = (np.arange(1,len(Pvals_contbase_MJJAS)+1,1)/len(Pvals_contbase_MJJAS))*alpha
k = np.where(Pvals_contbase_MJJAS > y)[0][0] # find first instance where Pvals > y

longrid_contbase_MJJAS,latgrid_contbase_MJJAS = np.meshgrid(diff_contbase_MJJAS.lon,diff_contbase_MJJAS.lat)
if k == 0:
    fdrP_contbase_MJJAS = 0
    siglon_contbase_MJJAS = np.asarray([np.nan])
    siglat_contbase_MJJAS = np.asarray([np.nan])
else:
    fdrP_contbase_MJJAS = Pvals_contbase_MJJAS[k-1] # use the index right before Pvals > y -> all Pvals should be less than fdrPcutoff to pass
    siglon_contbase_MJJAS = longrid_contbase_MJJAS[np.asarray(prob_contbase_MJJAS<=fdrP_contbase_MJJAS)]
    siglat_contbase_MJJAS = latgrid_contbase_MJJAS[np.asarray(prob_contbase_MJJAS<=fdrP_contbase_MJJAS)]

#%%
# Plotting BS & BW together:

# -------------------------------------------------------------------------
# -------------------------- NDJFM ----------------------------------------
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(80,25))
ax = fig.subplot_mosaic('''
                        A..H
                        BDFI
                        CEGJ
                        ''',subplot_kw={'projection':ccrs.PlateCarree()})


fig.subplots_adjust(wspace=0.075,hspace=0.075)
for loc in ['A','B','C','D','E','F','G','H','I','J']:
    ax[loc].coastlines(resolution='50m', color='dimgrey', linewidth=1)
    ax[loc].axis("off")
    ax[loc].set_ylim(-62.5,72.5)#72.5)

    gl = ax[loc].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

cmap = 'bone_r'
csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(50, 101, 1),csm.N)

if iper == 1:
    ax['A'].text(x=-180,y=150,s='Confident Prediction Accuracy',fontsize=75,color='dimgrey')
else:
    ax['A'].text(x=-180,y=150,s='All Prediction Accuracy',fontsize=75,color='dimgrey')

ax['A'].text(x=-180,y=110,s='Boreal Summer:',fontsize=75,color='dimgrey')
ax['H'].text(x=-180,y=110,s='Boreal Winter:',fontsize=75,color='dimgrey')


ax['A'].set_title('(a) BASE' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['A'].pcolor(base_MJJAS.lon,base_MJJAS.lat,base_MJJAS_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['B'].set_title('(b) SAI' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['B'].pcolor(sai_MJJAS.lon,sai_MJJAS.lat,sai_MJJAS_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['C'].set_title('(c) SSP2-4.5' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['C'].pcolor(cont_MJJAS.lon,cont_MJJAS.lat,cont_MJJAS_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)

# ----

ax['H'].set_title('(h) BASE' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['H'].pcolor(base_NDJFM.lon,base_NDJFM.lat,base_NDJFM_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['I'].set_title('(i) SAI' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['I'].pcolor(sai_NDJFM.lon,sai_NDJFM.lat,sai_NDJFM_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['J'].set_title('(j) SSP2-4.5' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['J'].pcolor(cont_NDJFM.lon,cont_NDJFM.lat,cont_NDJFM_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)

cax = plt.axes([.34,0.85,0.34,0.03])# [.55,0.84,0.34,0.02])
cbar = fig.colorbar(c1,cax=cax,orientation = 'horizontal', ticks=np.arange(50,110,10),shrink=0.5, pad=0.1)
cbar.ax.tick_params(size=0,labelsize=50)
cbar.outline.set_visible(False)
cbar.ax.set_xticklabels(np.arange(50,110,10),color='darkgrey')
cbar.ax.set_xlabel('accuracy (a, b, c, h, i, & j)',fontsize=60,color='darkgrey')

# ------- PLOT DIFF -------

cmap = 'RdBu_r'

csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(0, 1.025, .025),csm.N)

ax['D'].set_title('(d) SAI - BASE' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['D'].pcolor(diff_saibase_MJJAS.lon,diff_saibase_MJJAS.lat,diff_saibase_MJJAS,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['D'].scatter(siglon_saibase_MJJAS,siglat_saibase_MJJAS,marker='.',color='w',edgecolors='k',transform=ccrs.PlateCarree(),s=300,alpha=1)

ax['E'].set_title('(e) SSP2-4.5 - BASE' ,fontsize=60,color='dimgrey',loc='left')
c4 = ax['E'].pcolor(diff_contbase_MJJAS.lon,diff_contbase_MJJAS.lat,diff_contbase_MJJAS,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['E'].scatter(siglon_contbase_MJJAS,siglat_contbase_MJJAS,marker='.',color='w',edgecolors='k',transform=ccrs.PlateCarree(),s=300,alpha=1)



ax['F'].set_title('(f) SAI - BASE' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['F'].pcolor(diff_saibase_NDJFM.lon,diff_saibase_NDJFM.lat,diff_saibase_NDJFM,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['F'].scatter(siglon_saibase_NDJFM,siglat_saibase_NDJFM,marker='.',color='w',edgecolors='k',transform=ccrs.PlateCarree(),s=300,alpha=1)

ax['G'].set_title('(g) SSP2-4.5 - BASE' ,fontsize=60,color='dimgrey',loc='left')
c4 = ax['G'].pcolor(diff_contbase_NDJFM.lon,diff_contbase_NDJFM.lat,diff_contbase_NDJFM,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['G'].scatter(siglon_contbase_NDJFM,siglat_contbase_NDJFM,marker='.',color='w',edgecolors='k',transform=ccrs.PlateCarree(),s=300,alpha=1)


cax2 = plt.axes([.34,0.74,0.34,0.03])#[.55,0.74,0.34,0.02])
cbar2 = plt.colorbar(c4,cax=cax2,orientation = 'horizontal',fraction=0.04, ticks=np.arange(0,1.1,.1))
cbar2.ax.tick_params(size=0,labelsize=50)
cbar2.outline.set_visible(False)
cbar2.ax.set_xticklabels(np.round(np.arange(0,1.1,.1),2),color='darkgrey')
cbar2.ax.set_xlabel('sign frequency (d, e, f, & g)',fontsize=60,color='darkgrey')

plt.show()
# if iper == 1:
#     plt.savefig(DIR_FIG+'NNconfidentaccc_signtest_wilks.png',bbox_inches='tight',dpi=300)
# else:
#     plt.savefig(DIR_FIG+'NNallaccc_signtest_wilks.png',bbox_inches='tight',dpi=300)

#%%
# Plotting BS:

fig = plt.figure(figsize=(65,20))
ax = fig.subplot_mosaic('''
                        ABC
                        DE.
                        ''',subplot_kw={'projection':ccrs.PlateCarree()})


fig.subplots_adjust(wspace=0.075,hspace=0.075)
for loc in ['A','B','C','D','E']:
    ax[loc].coastlines(resolution='50m', color='dimgrey', linewidth=1)
    ax[loc].axis("off")
    ax[loc].set_ylim(-62.5,72.5)#72.5)

    gl = ax[loc].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

cmap = 'bone_r'
csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(50, 101, 1),csm.N)

# if iper == 1:
#     ax['A'].text(x=-180,y=150,s='Confident Prediction Accuracy',fontsize=75,color='dimgrey')
# else:
#     ax['A'].text(x=-180,y=150,s='All Prediction Accuracy',fontsize=75,color='dimgrey')

ax['A'].text(x=-180,y=110,s='Boreal Summer:',fontsize=75,color='dimgrey')


ax['A'].set_title('(a) BASE' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['A'].pcolor(base_MJJAS.lon,base_MJJAS.lat,base_MJJAS_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['B'].set_title('(b) SAI' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['B'].pcolor(sai_MJJAS.lon,sai_MJJAS.lat,sai_MJJAS_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['C'].set_title('(c) SSP2-4.5' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['C'].pcolor(cont_MJJAS.lon,cont_MJJAS.lat,cont_MJJAS_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


cax = plt.axes([.65,0.425,0.25,0.03]) # [left-right, up-down, length, width]
cbar = fig.colorbar(c1,cax=cax,orientation = 'horizontal', ticks=np.arange(50,110,10),shrink=0.5, pad=0.1)
cbar.ax.tick_params(size=0,labelsize=50)
cbar.outline.set_visible(False)
cbar.ax.set_xticklabels(np.arange(50,110,10),color='darkgrey')
cbar.ax.set_xlabel('accuracy (a, b, c)',fontsize=60,color='darkgrey')

# ------- PLOT DIFF -------

cmap = 'RdBu_r'

csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(0, 1.025, .025),csm.N)

ax['D'].set_title('(d) SAI - BASE' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['D'].pcolor(diff_saibase_MJJAS.lon,diff_saibase_MJJAS.lat,diff_saibase_MJJAS,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['D'].scatter(siglon_saibase_MJJAS,siglat_saibase_MJJAS,marker='.',color='w',edgecolors='k',transform=ccrs.PlateCarree(),s=300,alpha=1)

ax['E'].set_title('(e) SSP2-4.5 - BASE' ,fontsize=60,color='dimgrey',loc='left')
c4 = ax['E'].pcolor(diff_contbase_MJJAS.lon,diff_contbase_MJJAS.lat,diff_contbase_MJJAS,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['E'].scatter(siglon_contbase_MJJAS,siglat_contbase_MJJAS,marker='.',color='w',edgecolors='k',transform=ccrs.PlateCarree(),s=300,alpha=1)

cax2 = plt.axes([.65,0.275,0.25,0.03]) # [left-right, up-down, length, width]
cbar2 = plt.colorbar(c4,cax=cax2,orientation = 'horizontal',fraction=0.04, ticks=np.arange(0,1.1,.1))
cbar2.ax.tick_params(size=0,labelsize=50)
cbar2.outline.set_visible(False)
cbar2.ax.set_xticklabels(np.round(np.arange(0,1.1,.1),2),color='darkgrey')
cbar2.ax.set_xlabel('sign frequency (d, e)',fontsize=60,color='darkgrey')

# plt.show()
if iper == 1:
    plt.savefig(DIR_FIG+'NNconfidentaccc_signtest_wilks_borealsummer.png',bbox_inches='tight',dpi=300)
else:
    plt.savefig(DIR_FIG+'NNallaccc_signtest_wilks_borealsummer.png',bbox_inches='tight',dpi=300)

#%%
# Plotting BW:

fig = plt.figure(figsize=(65,20))
ax = fig.subplot_mosaic('''
                        ABC
                        DE.
                        ''',subplot_kw={'projection':ccrs.PlateCarree()})


fig.subplots_adjust(wspace=0.075,hspace=0.075)
for loc in ['A','B','C','D','E']:
    ax[loc].coastlines(resolution='50m', color='dimgrey', linewidth=1)
    ax[loc].axis("off")
    ax[loc].set_ylim(-62.5,72.5)#72.5)

    gl = ax[loc].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

cmap = 'bone_r'
csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(50, 101, 1),csm.N)

# if iper == 1:
#     ax['A'].text(x=-180,y=150,s='Confident Prediction Accuracy',fontsize=75,color='dimgrey')
# else:
#     ax['A'].text(x=-180,y=150,s='All Prediction Accuracy',fontsize=75,color='dimgrey')

ax['A'].text(x=-180,y=110,s='Boreal Winter:',fontsize=75,color='dimgrey')


ax['A'].set_title('(a) BASE' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['A'].pcolor(base_NDJFM.lon,base_NDJFM.lat,base_NDJFM_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['B'].set_title('(b) SAI' ,fontsize=60,color='dimgrey',loc='left')
c1 = ax['B'].pcolor(sai_NDJFM.lon,sai_NDJFM.lat,sai_NDJFM_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


ax['C'].set_title('(c) SSP2-4.5' ,fontsize=60,color='dimgrey',loc='left')
c2 = ax['C'].pcolor(cont_NDJFM.lon,cont_NDJFM.lat,cont_NDJFM_mean,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)


cax = plt.axes([.65,0.425,0.25,0.03]) # [left-right, up-down, length, width]
cbar = fig.colorbar(c1,cax=cax,orientation = 'horizontal', ticks=np.arange(50,110,10),shrink=0.5, pad=0.1)
cbar.ax.tick_params(size=0,labelsize=50)
cbar.outline.set_visible(False)
cbar.ax.set_xticklabels(np.arange(50,110,10),color='darkgrey')
cbar.ax.set_xlabel('accuracy (a, b, c)',fontsize=60,color='darkgrey')

# ------- PLOT DIFF -------

cmap = 'RdBu_r'

csm=plt.get_cmap(cmap)
norm = c.BoundaryNorm(np.arange(0, 1.025, .025),csm.N)

ax['D'].set_title('(d) SAI - BASE' ,fontsize=60,color='dimgrey',loc='left')
c3 = ax['D'].pcolor(diff_saibase_NDJFM.lon,diff_saibase_NDJFM.lat,diff_saibase_NDJFM,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['D'].scatter(siglon_saibase_NDJFM,siglat_saibase_NDJFM,marker='.',color='w',edgecolors='k',transform=ccrs.PlateCarree(),s=300,alpha=1)

ax['E'].set_title('(e) SSP2-4.5 - BASE' ,fontsize=60,color='dimgrey',loc='left')
c4 = ax['E'].pcolor(diff_contbase_NDJFM.lon,diff_contbase_NDJFM.lat,diff_contbase_NDJFM,cmap=csm,transform=ccrs.PlateCarree(),norm=norm)
ax['E'].scatter(siglon_contbase_NDJFM,siglat_contbase_NDJFM,marker='.',color='w',edgecolors='k',transform=ccrs.PlateCarree(),s=300,alpha=1)

cax2 = plt.axes([.65,0.275,0.25,0.03]) # [left-right, up-down, length, width]
cbar2 = plt.colorbar(c4,cax=cax2,orientation = 'horizontal',fraction=0.04, ticks=np.arange(0,1.1,.1))
cbar2.ax.tick_params(size=0,labelsize=50)
cbar2.outline.set_visible(False)
cbar2.ax.set_xticklabels(np.round(np.arange(0,1.1,.1),2),color='darkgrey')
cbar2.ax.set_xlabel('sign frequency (d, e)',fontsize=60,color='darkgrey')

# plt.show()
if iper == 1:
    plt.savefig(DIR_FIG+'NNconfidentaccc_signtest_wilks_borealwinter.png',bbox_inches='tight',dpi=300)
else:
    plt.savefig(DIR_FIG+'NNallaccc_signtest_wilks_borealwinter.png',bbox_inches='tight',dpi=300)
