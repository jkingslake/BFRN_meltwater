# Matlab code for routing water across a non-filled DEM
This is an early version of the code to route water across a non-filled DEM written in matlab. It requires Topotoolbox. Schwanghart, W., Scherler, D. (2014): TopoToolbox 2 – MATLAB-based software for topographic analysis and modeling in Earth surface sciences. Earth Surface Dynamics, 2, 1-7. DOI: 10.5194/esurf-2-1-2014
Thank yo to the developers of Topotoolbox for making this code available.

This code uses a merged REMA (Reference Elevation Model of Antarctica) DEM <https://www.pgc.umn.edu/data/rema/> over Amery Ice Shelf.

FillandMergeBasins_v2.m is the main function for simply computing flow across the REMA DEM by filling the depressions and joining up basins. This function calls AddWaterDepthsTohs.m and Hypsometry_of_all_basins.m and uses the subset of REMA contained in the GRIDObj stored in ClippedDEMforPartialFillingAnalysis.mat. It could use any DEM though. 

Simply running FillandMergeBasins_v2.m should result in the DEM being progressivly filled until all depressions are filled and the process will stop. 

FillandMergeBasins_v2.m does not resolve water flow explicitly, it just assumes instantaneous arrival at the lowest point in each basin. 
Advantages:  This appraoch is very fast compared to modelling the flow. This code can be used for looking at nonlinearities associated with filling of depressions. 
Limitations: This approach is not going to be that useful for modelling incision and percolation because it doesnt reslove the flow. 

