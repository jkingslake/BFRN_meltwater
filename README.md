# BFRN_meltwater
Project to compute the direct impact of meltwater flow across ice shelf surfaces on buttressing. See WAIS abstract

The master branch does the flow routing in Matlab. This branch is aimed at converting this to python using richdem and pangeo. 

Old version (currently the master branch) requires Topotoolbox

Uses a merged REMA dem <https://www.pgc.umn.edu/data/rema/> over Amery Ice Shelf

FillandMergeBasins_v2.m is the main function for simply computing flow across the REMA DEM by filling the depressions and joining up basins. THis function calls AddWaterDepthsTohs.m and Hypsometry_of_all_basins.m and uses the subset of REMA contained inteh GRIDObj stored in ClippedDEMforPartialFillingAnalysis.mat. It could use any DEM though. 

Simply running FillandMergeBasins_v2.m should result in the DE being progressivly filled until all depressions are filled and the process will stop. 

FillandMergeBasins_v2.m does not resolve wate flow explicitly, it just assumes instantaneous arival at the lowest point in each basin. 
Advantages:  this appraoch is very fast compared to modelling the flow. This code can be used for looking at nonlinearities associated with filling if depressions. 
Limitations: this approach is not going to be that useful for including incision and percolation because it doesnt reslove the flow. 

