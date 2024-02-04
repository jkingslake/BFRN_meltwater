%% Script to load buttressing reponse number results from Reese et al., 2016 and melt data from Trusel et al,, 2015


%% 1. Load buttressing response number output
load('ButtressingFluxResponseNumbers')

B = GRIDobj(x,y,BFRN);


%% 2. Load melt data

M = GRIDobj('Truseletal2015_2091-2100_Melt\tas_CLM4_ensembleMean_rcp85_Mar2015_maskedBiasCorrectedProjectedMeltFlux_Mar2015_2091-2100Mean_rectified_cutline_lowMeltMasked.tif');



