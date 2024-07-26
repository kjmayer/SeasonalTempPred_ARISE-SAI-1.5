# SeasonalTempPred_ARISE-SAI-1.5
Associated code for paper "Future Seasonal Surface Temperature Predictability with and without ARISE-Stratospheric Aerosol Injection-1.5"

## Main Paper Scripts:
### Preprocessing
removeseasonalcycle_anddetrend.py

### Training Functions
ANN.py \
split_data_gridpt.py

### Train and Test Networks
trainNN_gridpt.py \
trainNNbaseline_gridpt.py \
evaluateNN_gridpt.py \
evaluateNNbaseline_gridpt.py

### Figures 2 and 3
global_analysis/plot_scripts/plot_acc_signtest.py

### Figures 3 and 4
global_analysis/plot_scripts/plot_confcorr_SST.py

## Scripts for supplemental figures
evaluateNN_gridpt_leastconf.py \
evaluateNNbaseline_gridpt_leastconf.py \
plot_acc20leastconf_signtest.py
