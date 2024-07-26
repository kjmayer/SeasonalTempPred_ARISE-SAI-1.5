# Future Seasonal Surface Temperature Predictability with and without ARISE-Stratospheric Aerosol Injection-1.5
Paper in preparation for Earth's Future

* __project-name__: SeasonalTempPred_ARISE-SAI-1.5
* __authors__: Kirsten J. Mayer, Elizabeth A. Barnes, James Hurrell
* __date__: July 26, 2024

To help reduce anthropogenic climate change impacts, various forms of solar radiation modification have been proposed to reduce the rate of warming. One method to intentionally reflect sunlight into space is through the introduction of reflective particles into the stratosphere, known as stratospheric aerosol injection (SAI). Previous research has shown that SAI implementation could lead to future climate impacts beyond surface temperature, including changes in the distribution of future tropical precipitation. This response has the potential to modulate midlatitude variability and predictability through atmospheric teleconnections. Here, we explore possible differences in surface temperature predictability under a future with and without SAI implementation, using the ARISE-SAI-1.5 simulations. We find significant future predictability changes in both boreal summer and winter under both SSP2-4.5 with and without SAI. However, during boreal winter, some of the increases in future predictability under SS2-4.5 are mitigated by SAI, particularly in regions impacted by ENSO teleconnections. 

## Data Preprocessing
removeseasonalcycle_anddetrend.py

## Training Functions
ANN.py \
split_data_gridpt.py

## Train and Test Networks
trainNN_gridpt.py \
trainNNbaseline_gridpt.py \
evaluateNN_gridpt.py \
evaluateNNbaseline_gridpt.py

## Figures 2 and 3
plot_scripts/plot_acc_signtest.py

## Figures 3 and 4
plot_scripts/plot_confcorr_SST.py

## Scripts for Supplemental:
evaluateNN_gridpt_leastconf.py \
evaluateNNbaseline_gridpt_leastconf.py \
plot_acc20leastconf_signtest.py
