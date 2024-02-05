# Grounding-line flux changes due to flow of surface meltwater across Antarctic ice shelves 

## Introduction 
Meltwater produced on the surface of the Antarctic Ice Sheet flows can flow across the surface of the ice for 10's or 100's of kilometers, but rarely reaches the ocean. Instead, most of it refreezes after draining some distance. This redistributes mass. Ice shelves provide buttressing to inland ice. Different areas of ice sheets provide different amounts of buttressing. Therefore, if the buttressing provided by the ice shelf in the location is different than the buttressing provided by the ice shelf in the location where the meltwater refreezes, the net result of melting, drainage and refreezing will be a change in ice flux across the grounding line.

This project aims to quantify the impact of meltwater flow on ice flux across the grounding line via the change that this mass redistribution has on buttressing.

For more details see Poster_abstract_Kingslake_final.docx, which is a poster abstract for the WAIS conference in 2020.

## Approach
- setup an algorithm that can take as input a DEM and a map of meltwater production and compute the final distribution of meltwater after it has flowed across the surface, potentially filling and overtopping depressions. 
- systematically apply this algorithm across all Antarctic ice shelves and adjacent regions, applying meltwater to grid cells and recording the distribution of meltwater, 
- combine this map of meltwater distribution with the loss of mass corresponding to the melting in the grid cell to produce many "redistribution maps"
- multiply the redistribution maps by the buttressing flux response numbers (BFRN) from Reese et al. 2018 to get a map of the change in ice flux across the grounding line due to the redistribution of mass. 

Assuming the buttressing response is linear (as Reese et al did), it doesn't matter if the mass change in any one spot is negative or positive, and it doesn't matter if there is variability spatially, the net change in ice flux across the grounding line is the sum of the changes associated with each the mass change in each individual grid cell.

Reese, R., Gudmundsson, G.H., Levermann, A. and Winkelmann, R., 2018. The far reach of ice-shelf thinning in Antarctica. Nature Climate Change, 8(1), pp.53-57.

## Code

The /matlab directory contains code written in 2019/2020 in matlab to compute the flow of meltwater across digital elevation models with depressions. This was used extensively by Julian Spergel in his PhD thesis (https://academiccommons.columbia.edu/doi/10.7916/swez-dp81; https://doi.org/10.7916/swez-dp81) to examine the propensity of different ice shelf areas to transport water. 

The /python directory contains code using a third-party library called fill-spill-merge to do a similar computation that the matlab code was designed to perform. fill-spill-merge is written in C++ and is well tested. Future work, at least for now, will be based around fill-spill-merge, as it is probably more reliable and faster than the Matlab code. The directory contains a number of Jupyter notebooks that use python to call fill-spill-merge and analyze and plot the results. 

Barnes, R., Callaghan, K.L. and Wickert, A.D., 2020. Computing water flow through complex landscapes, Part 3: Fill-Spill-Merge: Flow routing in depression hierarchies. Earth Surface Dynamics Discussions, 2020, pp.1-22.